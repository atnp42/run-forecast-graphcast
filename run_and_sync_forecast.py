import os
import time
import shutil
import threading
import subprocess
from datetime import datetime, timedelta

import xarray as xr
import pygrib
import dropbox
import warnings

# === Config ===
ACCESS_TOKEN = os.getenv("DROPBOX_TOKEN")
dbx = dropbox.Dropbox(ACCESS_TOKEN)

DROPBOX_ASSETS_PATH = "/run_graphcast/assets"
LOCAL_ASSETS_PATH = "/workspace/assets"
DROPBOX_RESULTS_PATH = "/graphcast_results"
LOCAL_RESULTS_PATH = "/workspace/results"
TEMP_NC_DIR = "/workspace/temp_nc"

os.makedirs(LOCAL_ASSETS_PATH, exist_ok=True)
os.makedirs(LOCAL_RESULTS_PATH, exist_ok=True)
os.makedirs(TEMP_NC_DIR, exist_ok=True)

# === CONUS Bounding Box ===
lat_min, lat_max = 20.0, 55.0
lon_min, lon_max = -130.0, -60

# === Upload Tracking ===
uploaded = set()
pending_uploads = []
uploads_done = threading.Event()

warnings.filterwarnings("ignore", category=FutureWarning)

# === Dropbox ===
def download_folder(dbx, dropbox_path, local_path):
    entries = dbx.files_list_folder(dropbox_path).entries
    for entry in entries:
        dp = f"{dropbox_path}/{entry.name}"
        lp = f"{local_path}/{entry.name}"

        if isinstance(entry, dropbox.files.FileMetadata):
            print(f"[ASSETS] Downloading asset: {dp}")
            with open(lp, "wb") as f:
                _, res = dbx.files_download(dp)
                f.write(res.content)
        elif isinstance(entry, dropbox.files.FolderMetadata):
            os.makedirs(lp, exist_ok=True)
            download_folder(dbx, dp, lp)

# === Uploading ===
def is_file_stable(path, min_size_bytes=50_000_000, idle_seconds=30):
    try:
        stat = os.stat(path)
        return stat.st_size >= min_size_bytes and (time.time() - stat.st_mtime) >= idle_seconds
    except FileNotFoundError:
        return False

def upload_result_to_dropbox(local_file):
    file_name = os.path.basename(local_file)
    dropbox_target_path = f"{DROPBOX_RESULTS_PATH}/{file_name}"
    file_size = os.path.getsize(local_file)
    size_mb = file_size / (1024 * 1024)

    print(f"[UPLOAD] Start: {file_name} ({size_mb:.2f} MB)")
    CHUNK_SIZE = 100 * 1024 * 1024

    with open(local_file, "rb") as f:
        if file_size <= CHUNK_SIZE:
            dbx.files_upload(f.read(), dropbox_target_path, mode=dropbox.files.WriteMode("overwrite"))
        else:
            session_start = dbx.files_upload_session_start(f.read(CHUNK_SIZE))
            cursor = dropbox.files.UploadSessionCursor(session_id=session_start.session_id, offset=f.tell())
            commit = dropbox.files.CommitInfo(path=dropbox_target_path, mode=dropbox.files.WriteMode("overwrite"))

            while f.tell() < file_size:
                chunk = f.read(CHUNK_SIZE)
                if f.tell() < file_size:
                    dbx.files_upload_session_append_v2(chunk, cursor)
                    cursor.offset = f.tell()
                else:
                    dbx.files_upload_session_finish(chunk, cursor, commit)

    os.remove(local_file)
    print(f"[UPLOAD] Done and deleted: {file_name}\n")

def upload_worker():
    print("[UPLOAD] Background uploader started.")
    while not uploads_done.is_set() or pending_uploads:
        for fname in list(pending_uploads):
            local_path = os.path.join(LOCAL_RESULTS_PATH, fname)

            if fname in uploaded or not os.path.isfile(local_path):
                continue

            if is_file_stable(local_path):
                try:
                    upload_result_to_dropbox(local_path)
                    uploaded.add(fname)
                    pending_uploads.remove(fname)
                except Exception as e:
                    print(f"[UPLOAD] Error uploading {fname}: {e}")
        time.sleep(5)
    print("[UPLOAD] All uploads completed. Uploader thread exiting.")

# === Spatial Subsetting ===
def subset_grib_to_conus(grib_path, output_path):
    try:
        with pygrib.open(grib_path) as grbs:
            unique_vars = sorted(set((grb.shortName, grb.typeOfLevel) for grb in grbs))

        saved_files = []
        for shortName, typeOfLevel in unique_vars:
            try:
                ds = xr.open_dataset(
                    grib_path,
                    engine="cfgrib",
                    filter_by_keys={'shortName': shortName, 'typeOfLevel': typeOfLevel},
                    backend_kwargs={'indexpath': ''},
                )
                if ds.longitude.max() > 180:
                    ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
                    ds = ds.sortby('longitude')
                ds_subset = ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
                ds_subset.load()

                temp_file = os.path.join(TEMP_NC_DIR, f"{shortName}_{typeOfLevel}.nc")
                ds_subset.to_netcdf(temp_file)
                saved_files.append(temp_file)
            except Exception as e:
                print(f"[SUBSET] Skipping {shortName} ({typeOfLevel}): {e}")

        if saved_files:
            datasets = [xr.open_dataset(f, decode_times=True) for f in saved_files]
            merged = xr.merge(datasets, compat="override")
            merged.to_netcdf(output_path)
            for f in saved_files:
                os.remove(f)
            print(f"[SUBSET] Subset saved: {output_path}")
        else:
            print(f"[SUBSET] No valid variables found in {grib_path}")

        os.remove(grib_path)
    except Exception as e:
        print(f"[SUBSET] Failed to process {grib_path}: {e}")

# === Forecast and Process ===
def run_forecast_and_subset(date_str, time_str="1200", lead_time=168, model="graphcast"):
    raw_grib = os.path.join(LOCAL_RESULTS_PATH, f"{model}_{date_str}_{time_str}_{lead_time}h_gpu.grib")
    subset_nc = os.path.join(LOCAL_RESULTS_PATH, f"{model}_conus_{date_str}.nc")

    print(f"[FORECAST] Running for {date_str}")
    command = [
        "ai-models",
        "--assets", LOCAL_ASSETS_PATH,
        "--path", raw_grib,
        "--input", "cds",
        "--date", date_str,
        "--time", time_str,
        "--lead-time", str(lead_time),
        model
    ]

    subprocess.run(command)
    print(f"[FORECAST] Done: {date_str}")

    subset_grib_to_conus(raw_grib, subset_nc)
    pending_uploads.append(os.path.basename(subset_nc))

# === Main Loop ===
if __name__ == "__main__":
    print("[INIT] Downloading assets...")
    download_folder(dbx, DROPBOX_ASSETS_PATH, LOCAL_ASSETS_PATH)
    print("[INIT] Assets ready.\n")

    uploader_thread = threading.Thread(target=upload_worker, daemon=True)
    uploader_thread.start()

    start_date = datetime(2023, 1, 3)
    end_date = datetime(2023, 2, 3)

    forecast_threads = []
    while start_date <= end_date:
        date_str = start_date.strftime("%Y%m%d")
        t = threading.Thread(target=run_forecast_and_subset, args=(date_str,))
        t.start()
        forecast_threads.append(t)
        start_date += timedelta(days=1)

    for t in forecast_threads:
        t.join()

    print("[MAIN] Forecasts done. Waiting for uploads...")
    uploads_done.set()
    uploader_thread.join()
    print("[MAIN] All uploads completed.")
