import os
import time
import dropbox
import subprocess
import shutil
import zipfile
from datetime import datetime, timedelta
import xarray as xr
from eccodes import codes_grib_new_from_file, codes_get, codes_release
from collections import defaultdict

ACCESS_TOKEN = os.getenv("DROPBOX_TOKEN")
dbx = dropbox.Dropbox(ACCESS_TOKEN)

DROPBOX_ASSETS_PATH = "/run_graphcast/assets"
LOCAL_ASSETS_PATH = "/workspace/assets"

DROPBOX_RESULTS_PATH = "/graphcast_results"
LOCAL_RESULTS_PATH = "/workspace/results"

os.makedirs(LOCAL_RESULTS_PATH, exist_ok=True)
os.makedirs(LOCAL_ASSETS_PATH, exist_ok=True)

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

lat_min, lat_max = 15, 50
lon_min, lon_max = -130, -66

def targeted_cleanup(base_name):
    print(f"[CLEANUP] Cleaning files for: {base_name}")
    targets = [
        f"{base_name}.grib",
        f"{base_name}.zip",
        base_name
    ]
    for target in targets:
        full_path = os.path.join(LOCAL_RESULTS_PATH, target)
        if os.path.exists(full_path):
            try:
                if os.path.isfile(full_path):
                    os.remove(full_path)
                else:
                    shutil.rmtree(full_path)
                print(f"[CLEANUP] Removed: {full_path}")
            except Exception as e:
                print(f"[CLEANUP] Failed to remove {full_path}: {e}")
        else:
            print(f"[CLEANUP] Skipped (not found): {full_path}")

def get_all_field_levels(grib_path):
    field_levels = defaultdict(set)
    with open(grib_path, "rb") as f:
        while True:
            gid = codes_grib_new_from_file(f)
            if gid is None:
                break
            try:
                short_name = codes_get(gid, "shortName")
                level_type = codes_get(gid, "typeOfLevel")
                level = int(codes_get(gid, "level"))
                field_levels[(short_name, level_type)].add(level)
            except:
                pass
            finally:
                codes_release(gid)
    return field_levels

def crop_and_prepare_and_upload(local_grib_path):
    print(f"[PROCESS] Starting subsetting for {local_grib_path}...")
    base_name = os.path.splitext(os.path.basename(local_grib_path))[0]
    output_dir = os.path.join(LOCAL_RESULTS_PATH, base_name)
    os.makedirs(output_dir, exist_ok=True)

    field_levels = get_all_field_levels(local_grib_path)

    for (short_name, level_type), levels in sorted(field_levels.items()):
        for level in sorted(levels):
            try:
                print(f"[PROCESS] Loading {short_name} @ {level_type}={level}")
                ds = xr.open_dataset(
                    local_grib_path,
                    engine="cfgrib",
                    backend_kwargs={
                        "filter_by_keys": {
                            "shortName": short_name,
                            "typeOfLevel": level_type,
                            "level": level
                        },
                        "indexpath": ""
                    },
                    decode_times=True
                )

                if ds.longitude.max() > 180:
                    ds = ds.assign_coords(longitude=(ds.longitude + 180) % 360 - 180)
                    ds = ds.sortby("longitude")

                ds_sub = ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
                out_name = f"{short_name}_{level_type}_lvl{level}.nc"
                out_path = os.path.join(output_dir, out_name)
                ds_sub.to_netcdf(out_path)
                print(f"[SAVED] {out_path}")

            except Exception as e:
                print(f"[SKIPPED] {short_name} @ {level_type}={level}: {e}")

    zip_path = f"{output_dir}.zip"
    print(f"[PROCESS] Zipping folder: {zip_path}")
    shutil.make_archive(output_dir, 'zip', output_dir)
    print(f"[PROCESS] Zipped: {zip_path}")

    file_name = os.path.basename(zip_path)
    dropbox_target = f"{DROPBOX_RESULTS_PATH}/{file_name}"
    print(f"[UPLOAD] Uploading {file_name} to {dropbox_target}")

    with open(zip_path, "rb") as f:
        dbx.files_upload(f.read(), dropbox_target, mode=dropbox.files.WriteMode("overwrite"))

    print(f"[UPLOAD] Upload complete: {file_name}")

def run_forecasts():
    start_date = datetime(2019, 10, 1)
    end_date = datetime(2019, 12, 31)
    lead_time = 168
    time_str = "1200"
    model = "graphcast"

    forecast_queue = []  # (date_str, grib_path)

    def run_forecast(date_str):
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
        return output_path

    def process_forecast(date_str, grib_path, previous_grib_path=None):
        if previous_grib_path:
            prev_base = os.path.splitext(os.path.basename(previous_grib_path))[0]
            print(f"[CLEANUP] Cleaning files from previous forecast: {prev_base}")
            targeted_cleanup(prev_base)

        print(f"[PROCESS] Processing forecast for {date_str}")
        crop_and_prepare_and_upload(grib_path)
        print(f"[PROCESS] Done processing {date_str}")

    # Step 1: Run first forecast only
    current_date = start_date.strftime("%Y%m%d")
    current_grib = run_forecast(current_date)
    forecast_queue.append((current_date, current_grib))
    start_date += timedelta(days=1)

    # Step 2: Run second forecast only
    next_date = start_date.strftime("%Y%m%d")
    next_grib = run_forecast(next_date)
    forecast_queue.append((next_date, next_grib))
    start_date += timedelta(days=1)

    previous_grib_path = None

    # Step 3: Main loop
    while start_date <= end_date:
        # Process oldest forecast (T) and delete T-1
        process_date, process_grib = forecast_queue.pop(0)

        # Start next forecast (T+2)
        current_date = start_date.strftime("%Y%m%d")
        current_grib = run_forecast(current_date)
        forecast_queue.append((current_date, current_grib))

        # Process forecast and clean up previous
        process_forecast(process_date, process_grib, previous_grib_path)

        # Track for next cleanup
        previous_grib_path = process_grib

        start_date += timedelta(days=1)

    # Final processing of remaining forecasts
    while forecast_queue:
        process_date, process_grib = forecast_queue.pop(0)
        process_forecast(process_date, process_grib, previous_grib_path)
        previous_grib_path = process_grib

if __name__ == "__main__":
    print("[INIT] Downloading assets from Dropbox...")
    download_folder(dbx, DROPBOX_ASSETS_PATH, LOCAL_ASSETS_PATH)
    print("[INIT] Assets downloaded.\n")

    run_forecasts()

    print("[DONE] All forecasts processed and uploaded.")
