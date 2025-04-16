import os
import time
import dropbox
import subprocess
import threading
from datetime import datetime, timedelta

# === Dropbox Setup ===
ACCESS_TOKEN = os.getenv("DROPBOX_TOKEN")
dbx = dropbox.Dropbox(ACCESS_TOKEN)

DROPBOX_ASSETS_PATH = "/run_graphcast/assets"
LOCAL_ASSETS_PATH = "/workspace/assets"

DROPBOX_RESULTS_PATH = "/graphcast_results"
LOCAL_RESULTS_PATH = "/workspace/results"

os.makedirs(LOCAL_RESULTS_PATH, exist_ok=True)
os.makedirs(LOCAL_ASSETS_PATH, exist_ok=True)

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

def is_file_stable(path, min_size_bytes=4_500_000_000, idle_seconds=30):
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
    CHUNK_SIZE = 100 * 1024 * 1024  # 100 MB

    with open(local_file, "rb") as f:
        if file_size <= CHUNK_SIZE:
            print("[UPLOAD] File is small. Uploading in one request.")
            dbx.files_upload(f.read(), dropbox_target_path, mode=dropbox.files.WriteMode("overwrite"))
            print("[UPLOAD] Upload complete.")
        else:
            total_chunks = (file_size + CHUNK_SIZE - 1) // CHUNK_SIZE
            print(f"[UPLOAD] Large file detected. Uploading in {total_chunks} chunks of {CHUNK_SIZE // (1024 * 1024)} MB each.")

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

# === Forecast-Loop ===

def run_forecasts():
    start_date = datetime(2023, 1, 1)
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

        print(f"[FORECAST] Running forecast for {date_str}")
        subprocess.run(command)
        print(f"[FORECAST] Finished forecast for {date_str}\n")

        # Queue for upload
        pending_uploads.append(output_filename)
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
