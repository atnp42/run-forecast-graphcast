import os
import time
import dropbox
import subprocess
from datetime import datetime, timedelta

ACCESS_TOKEN = os.getenv("DROPBOX_TOKEN")
dbx = dropbox.Dropbox(ACCESS_TOKEN)

DROPBOX_ASSETS_PATH = "/graphcast/run_graphcast/assets"
LOCAL_ASSETS_PATH = "/workspace/assets"

DROPBOX_RESULTS_PATH = "/graphcast/graphcast_results"
LOCAL_RESULTS_PATH = "/workspace/results"

os.makedirs(LOCAL_RESULTS_PATH, exist_ok=True)
os.makedirs(LOCAL_ASSETS_PATH, exist_ok=True)


def download_folder(dbx, dropbox_path, local_path):
    entries = dbx.files_list_folder(dropbox_path).entries
    for entry in entries:
        dp = f"{dropbox_path}/{entry.name}"
        lp = f"{local_path}/{entry.name}"

        if isinstance(entry, dropbox.files.FileMetadata):
            print(f"Downloading asset: {dp}")
            with open(lp, "wb") as f:
                _, res = dbx.files_download(dp)
                f.write(res.content)
        elif isinstance(entry, dropbox.files.FolderMetadata):
            os.makedirs(lp, exist_ok=True)
            download_folder(dbx, dp, lp)


def upload_result_to_dropbox(local_file):
    file_name = os.path.basename(local_file)
    dropbox_target_path = f"{DROPBOX_RESULTS_PATH}/{file_name}"

    with open(local_file, "rb") as f:
        dbx.files_upload(f.read(), dropbox_target_path, mode=dropbox.files.WriteMode("overwrite"))
    os.remove(local_file)
    print(f"Uploaded and removed: {file_name}")


def run_forecasts():
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 1, 2)
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

        print(f"Running forecast for {date_str}")
        subprocess.run(command)

        if os.path.exists(output_path):
            upload_result_to_dropbox(output_path)
        else:
            print(f"Warning: result file not found for {date_str}, skipping upload.")

        start_date += timedelta(days=1)


if __name__ == "__main__":
    print("Downloading assets from Dropbox...")
    download_folder(dbx, DROPBOX_ASSETS_PATH, LOCAL_ASSETS_PATH)
    print("Assets downloaded.\n")

    run_forecasts()
