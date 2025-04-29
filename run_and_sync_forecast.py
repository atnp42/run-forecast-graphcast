import os
import time
import dropbox
import subprocess
import threading
from datetime import datetime, timedelta
import xarray as xr
import numpy as np
import glob
import warnings
from eccodes import codes_grib_new_from_file, codes_get, codes_release

# === Dropbox Setup ===
ACCESS_TOKEN = os.getenv("DROPBOX_TOKEN")
dbx = dropbox.Dropbox(ACCESS_TOKEN)

DROPBOX_ASSETS_PATH = "/run_graphcast/assets"
LOCAL_ASSETS_PATH = "/workspace/assets"

DROPBOX_RESULTS_PATH = "/graphcast_results"
LOCAL_RESULTS_PATH = "/workspace/results"
TEMP_SUBSET_DIR = "/workspace/results/temp"
FINAL_SUBSET_DIR = "/workspace/results/final"

os.makedirs(LOCAL_RESULTS_PATH, exist_ok=True)
os.makedirs(LOCAL_ASSETS_PATH, exist_ok=True)
os.makedirs(TEMP_SUBSET_DIR, exist_ok=True)
os.makedirs(FINAL_SUBSET_DIR, exist_ok=True)

# === Upload Tracking ===
uploaded = set()
pending_uploads = []
uploads_done = threading.Event()

# === Hilfsfunktionen ===
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

def is_file_stable(path, min_size_bytes=300_000_000, idle_seconds=30):
    try:
        stat = os.stat(path)
        file_size = stat.st_size
        mtime = stat.st_mtime
        time_since_mod = time.time() - mtime
        return file_size >= min_size_bytes and time_since_mod >= idle_seconds
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
            print("[UPLOAD] File is small. Uploading in one request.")
            dbx.files_upload(f.read(), dropbox_target_path, mode=dropbox.files.WriteMode("overwrite"))
            print("[UPLOAD] Upload complete.")
        else:
            total_chunks = (file_size + CHUNK_SIZE - 1) // CHUNK_SIZE
            print(f"[UPLOAD] Large file detected. Uploading in {total_chunks} chunks.")

            session_start = dbx.files_upload_session_start(f.read(CHUNK_SIZE))
            cursor = dropbox.files.UploadSessionCursor(session_id=session_start.session_id, offset=f.tell())
            commit = dropbox.files.CommitInfo(path=dropbox_target_path, mode=dropbox.files.WriteMode("overwrite"))

            chunk_idx = 1
            while True:
                chunk = f.read(CHUNK_SIZE)
                if not chunk:
                    break

                if f.tell() < file_size:
                    dbx.files_upload_session_append_v2(chunk, cursor)
                    print(f"[UPLOAD] Chunk {chunk_idx}/{total_chunks} appended.")
                    cursor.offset = f.tell()
                else:
                    dbx.files_upload_session_finish(chunk, cursor, commit)
                    print(f"[UPLOAD] Chunk {chunk_idx}/{total_chunks} uploaded and committed.")
                    break

                chunk_idx += 1

    os.remove(local_file)
    print(f"[UPLOAD] File upload complete and local file deleted: {file_name}\n")

def upload_worker():
    print("[UPLOAD] Background uploader started.")
    while not uploads_done.is_set() or pending_uploads:
        for fname in list(pending_uploads):
            local_path = os.path.join(FINAL_SUBSET_DIR, fname)

            if fname in uploaded or not os.path.isfile(local_path):
                continue

            try:
                upload_result_to_dropbox(local_path)
                uploaded.add(fname)
                pending_uploads.remove(fname)
            except Exception as e:
                print(f"[UPLOAD] Error uploading {fname}: {e}")
        time.sleep(5)

    print("[UPLOAD] All uploads completed. Uploader thread exiting.")

# === Zuschneiden ===
warnings.filterwarnings("ignore", category=FutureWarning, module="cfgrib")

def subset_grib(input_file, date_str):
    lat_min, lat_max = 15, 50
    lon_min, lon_max = -130, -66

    known_levels = {
        "surface": [0],
        "heightAboveGround": [2, 10],
        "meanSea": [0],
        "isobaricInhPa": [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    }

    for type_, levels in known_levels.items():
        for level in levels:
            try:
                filter_keys = {"typeOfLevel": type_, "level": int(level)}
                if type_ == "surface":
                    filter_keys["shortName"] = ["t2m", "u10", "v10", "msl", "z", "u", "v", "w", "q", "lsm"]

                ds = xr.open_dataset(
                    input_file,
                    engine="cfgrib",
                    backend_kwargs={"filter_by_keys": filter_keys, "indexpath": ""},
                    decode_times=True
                )

                if ds.longitude.max() > 180:
                    ds = ds.assign_coords(longitude=(ds.longitude + 180) % 360 - 180)
                    ds = ds.sortby("longitude")

                ds_sub = ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
                out_name = f"{type_}_lvl{int(level)}.nc"
                out_path = os.path.join(TEMP_SUBSET_DIR, out_name)
                ds_sub.to_netcdf(out_path)

            except Exception as e:
                print(f"[SUBSET] Failed typeOfLevel='{type_}' level={level}: {e}")

    forecast_times = set()
    with open(input_file, 'rb') as f:
        while True:
            gid = codes_grib_new_from_file(f)
            if gid is None:
                break
            try:
                short_name = codes_get(gid, "shortName")
                if short_name == "tp":
                    forecast_times.add(codes_get(gid, "dataTime"))
            finally:
                codes_release(gid)

    for forecast_time in sorted(forecast_times):
        try:
            ds_tp = xr.open_dataset(
                input_file,
                engine="cfgrib",
                backend_kwargs={
                    "filter_by_keys": {
                        "shortName": "tp",
                        "typeOfLevel": "surface",
                        "level": 0,
                        "dataTime": forecast_time
                    },
                    "indexpath": ""
                },
                decode_times=True
            )

            if ds_tp.longitude.max() > 180:
                ds_tp = ds_tp.assign_coords(longitude=(ds_tp.longitude + 180) % 360 - 180)
                ds_tp = ds_tp.sortby("longitude")

            ds_tp_sub = ds_tp.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
            out_name = f"tp_dataTime{forecast_time:04d}.nc"
            out_path = os.path.join(TEMP_SUBSET_DIR, out_name)
            ds_tp_sub.to_netcdf(out_path)

        except Exception as e:
            print(f"[SUBSET] Failed TP dataTime={forecast_time:04d}: {e}")

    all_nc_files = sorted(glob.glob(os.path.join(TEMP_SUBSET_DIR, "*.nc")))
    datasets = []
    for f in all_nc_files:
        try:
            ds = xr.open_dataset(f)
            datasets.append(ds)
        except Exception as e:
            print(f"[SUBSET] Warning: Failed to load {f}: {e}")

    merged = xr.merge(datasets, compat="override")
    final_path = os.path.join(FINAL_SUBSET_DIR, f"graphcast_{date_str}.nc")
    merged.to_netcdf(final_path)
    print(f"[SUBSET] Saved final subset to {final_path}")
    return os.path.basename(final_path)

# === Forecast-Loop ===
def run_forecasts():
    start_date = datetime(2023, 1, 3)
    end_date = datetime(2023, 2, 3)
    lead_time = 168
    time_str = "1200"
    model = "graphcast"

    prev_date_str = None

    while start_date <= end_date:
        date_str = start_date.strftime("%Y%m%d")
        output_filename = f"graphcast_{date_str}_{time_str}_{lead_time}h_gpu.grib"
        output_path = os.path.join(LOCAL_RESULTS_PATH, output_filename)

        # Lösche Dateien vom vorherigen Forecast
        if prev_date_str:
            try:
                prev_grib = os.path.join(LOCAL_RESULTS_PATH, f"graphcast_{prev_date_str}_{time_str}_{lead_time}h_gpu.grib")
                if os.path.exists(prev_grib):
                    os.remove(prev_grib)

                for f in glob.glob(os.path.join(TEMP_SUBSET_DIR, "*.nc")):
                    os.remove(f)

                prev_final = os.path.join(FINAL_SUBSET_DIR, f"graphcast_{prev_date_str}.nc")
                if os.path.exists(prev_final):
                    os.remove(prev_final)

                print(f"[CLEANUP] Removed previous forecast files for {prev_date_str}")
            except Exception as e:
                print(f"[CLEANUP] Failed to remove previous files: {e}")

        command = [
            "ai-models",
            "--assets", LOCAL_ASSETS_PATH,
            "--path", output_path,
            "--input", "cds",
            "--date", date_str,
            "--time", time_str,
            "--lead-time", str(lead_time),
            model
        ]

        print(f"[FORECAST] Running forecast for {date_str}")
        subprocess.run(command)
        print(f"[FORECAST] Finished forecast for {date_str}")

        subset_filename = subset_grib(output_path, date_str)
        pending_uploads.append(subset_filename)

        prev_date_str = date_str
        start_date += timedelta(days=1)

# === Main ===
if __name__ == "__main__":
    print("[INIT] Downloading assets from Dropbox...")
    download_folder(dbx, DROPBOX_ASSETS_PATH, LOCAL_ASSETS_PATH)
    print("[INIT] Assets downloaded.\n")

    uploader_thread = threading.Thread(target=upload_worker, daemon=True)
    uploader_thread.start()

    run_forecasts()

    print("[DONE] Forecasts finished. Waiting for uploads to complete...")
    uploads_done.set()
    uploader_thread.join()
    print("[DONE] All uploads are done.")
