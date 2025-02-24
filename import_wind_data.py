import earthaccess
import os

SAVE_DIR_ROOT = "./WindData"

auth = earthaccess.login()

temporal = ("2024-12-01", "2024-12-02")
bounding_box = (-180, -90, 180, 90)

run_str = f"WindData_{temporal[0]}_{temporal[1]}_{bounding_box[0]}_{bounding_box[1]}_{bounding_box[2]}_{bounding_box[3]}"
save_dir = os.path.join(SAVE_DIR_ROOT, run_str)

results = earthaccess.search_data(
    doi="10.5067/7MCPBJ41Y0K6",  # // cspell:disable-line
    temporal=temporal,
    bounding_box=bounding_box,
)
downloaded_files = earthaccess.download(results, local_path=save_dir)
