import os
import time
import dropbox
import subprocess
import threading
import shutil
import zipfile
from datetime import datetime, timedelta
import xarray as xr
from eccodes import codes_grib_new_from_file, codes_get, codes_release

# === Dropbox Setup ===
ACCESS_TOKEN = os.getenv("DROPBOX_TOKEN")
dbx = dropbox.Dropbox(ACCESS_TOKEN)

DROPBOX_ASSETS_PATH = "/run_graphcast/assets"
LOCAL_ASSETS_PATH = "/workspace/assets"

DROPBOX_RESULTS_PATH = "/graphcast_results"
LOCAL_RESULTS_PATH = "/workspace/results"

os.makedirs(LOCAL_RESULTS_PATH, exist_ok=True)
os.makedirs(LOCAL_ASSETS_PATH, exist_ok=True)

# === Download function ===
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

# === CONUS Cropping Config ===
lat_min, lat_max = 15, 50
lon_min, lon_max = -130, -66

known_levels = {
    "surface": [0],
    "heightAboveGround": [2, 10],
    "meanSea": [0],
    "isobaricInhPa": [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
}

# === Queues and Flags ===
processing_queue = []
forecasting_done = threading.Event()

# === Utility Functions ===

def clean_workspace():
    print("[CLEANUP] Cleaning local results folder...")
    for item in os.listdir(LOCAL_RESULTS_PATH):
        item_path = os.path.join(LOCAL_RESULTS_PATH, item)
        try:
            if os.path.isfile(item_path):
                os.remove(item_path)
            else:
                shutil.rmtree(item_path)
            print(f"[CLEANUP] Removed: {item_path}")
        except Exception as e:
            print(f"[CLEANUP] Failed to remove {item_path}: {e}")

def crop_and_prepare(local_grib_path):
    print(f"[PROCESS] Starting subsetting for {local_grib_path}...")
    base_name = os.path.splitext(os.path.basename(local_grib_path))[0]
    output_dir = os.path.join(LOCAL_RESULTS_PATH, base_name)
    os.makedirs(output_dir, exist_ok=True)

    for type_, levels in known_levels.items():
        for level in levels:
            print(f"[PROCESS] Loading typeOfLevel='{type_}', level={level}")
            try:
                filter_keys = {"typeOfLevel": type_, "level": int(level)}
                ds = xr.open_dataset(
                    local_grib_path,
                    engine="cfgrib",
                    backend_kwargs={"filter_by_keys": filter_keys, "indexpath": ""},
                    decode_times=True
                )

                if ds.longitude.max() > 180:
                    ds = ds.assign_coords(longitude=(ds.longitude + 180) % 360 - 180)
                    ds = ds.sortby("longitude")

                ds_sub = ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
                out_name = f"{type_}_lvl{int(level)}.nc"
                out_path = os.path.join(output_dir, out_name)
                ds_sub.to_netcdf(out_path)
                print(f"[PROCESS] Saved subset: {out_path}")

            except Exception as e:
                print(f"[ERROR] Failed to process typeOfLevel='{type_}' level={level}: {e}")

    print("[PROCESS] Processing Total Precipitation (tp)...")
    forecast_times = set()
    with open(local_grib_path, 'rb') as f:
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
                local_grib_path,
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
            out_path = os.path.join(output_dir, out_name)
            ds_tp_sub.to_netcdf(out_path)
            print(f"[PROCESS] Saved TP subset: {out_path}")

        except Exception as e:
            print(f"[ERROR] Failed TP for dataTime={forecast_time:04d}: {e}")

    zip_path = f"{output_dir}.zip"
    print(f"[PROCESS] Zipping folder: {zip_path}")
    shutil.make_archive(output_dir, 'zip', output_dir)
    print(f"[PROCESS] Zipped: {zip_path}")
    return zip_path

def upload_to_dropbox(zip_path):
    file_name = os.path.basename(zip_path)
    dropbox_target = f"{DROPBOX_RESULTS_PATH}/{file_name}"
    print(f"[UPLOAD] Uploading {file_name} to {dropbox_target}")

    with open(zip_path, "rb") as f:
        dbx.files_upload(f.read(), dropbox_target, mode=dropbox.files.WriteMode("overwrite"))

    print(f"[UPLOAD] Upload complete: {file_name}")
    clean_workspace()

def processing_worker():
    print("[PROCESS] Processor started.")
    while not (forecasting_done.is_set() and not processing_queue):
        if processing_queue:
            grib_file = processing_queue.pop(0)
            try:
                zip_file = crop_and_prepare(grib_file)
                upload_to_dropbox(zip_file)
            except Exception as e:
                print(f"[PROCESS] Error processing {grib_file}: {e}")
        else:
            time.sleep(5)
    print("[PROCESS] Processor exiting.")

def run_forecasts():
    start_date = datetime(2023, 1, 3)
    end_date = datetime(2023, 2, 3)
    lead_time = 168
    time_str = "1200"
    model = "graphcast"

    while start_date <= end_date:
        date_str = start_date.strftime("%Y%m%d")
        output_filename = f"graphcast_{date_str}_{time_str}_{lead_time}h_gpu.grib"
        output_path = os.path.join(LOCAL_RESULTS_PATH, output_filename)

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
        print(f"[FORECAST] Forecast complete: {output_filename}")

        processing_queue.append(output_path)

        start_date += timedelta(days=1)

    forecasting_done.set()

if __name__ == "__main__":
    print("[INIT] Downloading assets from Dropbox...")
    download_folder(dbx, DROPBOX_ASSETS_PATH, LOCAL_ASSETS_PATH)
    print("[INIT] Assets downloaded.\n")

    processor_thread = threading.Thread(target=processing_worker, daemon=True)
    processor_thread.start()

    run_forecasts()

    print("[DONE] Forecasting done. Waiting for processing/uploading to finish...")
    processor_thread.join()
    print("[DONE] All processing and uploads completed.")
