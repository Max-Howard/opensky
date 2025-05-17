import os
import pandas as pd
import numpy as np
import shutil
import xarray as xr
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

MET_DATA_DIR: str = "./met/wind_monthly_202411.nc4"
MET_DATA = None
RAW_DATA_DIR = "RawFlightData"
OUTPUT_DIR = "ProcessedFlightData"
TIME_SLICE = slice(0, 16) # Data slicing needed to reduce memory usage when multiprocessing
RDP_EPSILON = 10          # RDP tolerance in meters
MAX_TIME_GAP = 30 # Maximum time gap in seconds between points to keep
max_workers = 6

def find_flight_files(directory: str):
    flight_files = []
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            flight_files.append(file)
    return flight_files


def load_met_data(met_data_dir: str = MET_DATA_DIR, slice=None):
    print("Loading MET data... ", end="", flush=True)
    ds = xr.open_dataset(met_data_dir)
    if slice is not None:
        ds = ds.isel(time=slice)

    # Load into RAM
    MET_DATA = ds.load()
    print("done.")
    return MET_DATA


def clean_alts(df: pd.DataFrame) -> pd.DataFrame:
    geo_altitude_diff = df['geoaltitude'].diff()
    baro_altitude_diff = df['baroaltitude'].diff()
    time_gap = df['time'].diff()
    geo_altitude_rate = geo_altitude_diff / time_gap
    baro_altitude_rate = baro_altitude_diff / time_gap
    altitude_change_rate = np.maximum(geo_altitude_rate, baro_altitude_rate)

    indices_to_drop = altitude_change_rate[altitude_change_rate > 25].index
    df.drop(indices_to_drop, inplace=True) # TODO should this be idx + 1?

    threshold = 50
    for col in ['geoaltitude', 'baroaltitude']:
        jumps = df[col].diff().abs() > threshold
        returns = df[col].diff(-1).abs() > threshold
        indices_to_drop = df[jumps & returns].index
        df.drop(indices_to_drop, inplace=True)

    df.reset_index(drop=True, inplace=True)
    return df


def round_values(df: pd.DataFrame) -> pd.DataFrame:
    df['time'] = df['time'].round(2)
    df['lat'] = df['lat'].round(6)
    df['lon'] = df['lon'].round(6)
    df['geoaltitude'] = df['geoaltitude'].round(1)
    df['baroaltitude'] = df['baroaltitude'].round(1)
    df['gs'] = df['gs'].round(1)
    df['heading'] = df['heading'].round(1)
    df['vertrate'] = df['vertrate'].round(1)

    if 'tas' in df.columns:
        df['tas'] = df['tas'].round(1)
    if 'wind_speed' in df.columns:
        df['wind_speed'] = df['wind_speed'].round(1)
    if 'wind_dir' in df.columns:
        df['wind_dir'] = df['wind_dir'].round(1)

    # Rounding can re-introduce duplicates - drop them
    df.drop_duplicates(subset=["time"], keep="first", inplace=True)

    return df


def calc_dist(df: pd.DataFrame) -> pd.DataFrame:
    lat = np.radians(df['lat'].to_numpy())
    lon = np.radians(df['lon'].to_numpy())

    delta_lat = lat[1:] - lat[:-1]
    delta_lon = lon[1:] - lon[:-1]

    a = (np.sin(delta_lat/2)**2 +
         np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(delta_lon/2)**2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    dist = np.concatenate(([0], 6371e3 * c))

    df['dist'] = np.round(dist, 1)
    return df


def calc_tas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the True Airspeed (TAS) using the wind speed and direction from the MET data.
    Currently using barometric altitude to find wind data, as Merra alts are calculated from pressure.
    """

    if MET_DATA is None:
        raise ValueError("MET data has not been loaded.")

    # Compute the indices for each dimension
    # TODO this is calculating based off of edges, not centers. Can be improved.
    df["lat_idx"] = np.digitize(df["lat"], MET_DATA["lat"].values)
    df["lon_idx"] = np.digitize(df["lon"], MET_DATA["lon"].values)
    df["time_idx"] = np.digitize(df["time"], MET_DATA["time"].values.astype(np.int64)/1e9) # TODO this is not selecting correct time
    df["lev_idx"]  = np.digitize(df["geoaltitude"], MET_DATA["h_edge"].values) - 1 # TODO this is a temporary fix

    # Find the unique groups of indices
    unique_groups = df[["lat_idx", "lon_idx", "time_idx", "lev_idx"]].drop_duplicates().reset_index(drop=True)

    # Create xarray DataArrays from unique groups for vectorized indexing
    da_lat = xr.DataArray(unique_groups["lat_idx"].values, dims="points")
    da_lon = xr.DataArray(unique_groups["lon_idx"].values, dims="points")
    da_time = xr.DataArray(unique_groups["time_idx"].values, dims="points")
    da_lev = xr.DataArray(unique_groups["lev_idx"].values, dims="points")

    # Vectorized retrieval of wind speed and wind direction
    unique_groups["wind_speed"] = MET_DATA["WS"].isel(lev=da_lev, lat=da_lat, lon=da_lon, time=da_time).values
    unique_groups["wind_dir"] = MET_DATA["WDIR"].isel(lev=da_lev, lat=da_lat, lon=da_lon, time=da_time).values

    # Merge the unique groups back into the original dataframe
    df = df.merge(unique_groups, on=["lat_idx", "lon_idx", "time_idx", "lev_idx"], how="left")

    # Calculate TAS using cosine rule
    df["tas"] = np.sqrt(df["gs"]**2 + df["wind_speed"]**2 -
        2 * df["gs"] * df["wind_speed"] * np.cos(np.radians(df["wind_dir"] - df["heading"])))

    # Drop intermediate columns
    df.drop(columns=["lat_idx", "lon_idx", "time_idx", "lev_idx"], inplace=True)
    return df


def create_save_dir():
    if os.path.exists(OUTPUT_DIR):
        remove_dir = input(f"The directory {OUTPUT_DIR} already exists. Do you want to remove it contents? (yes/no): ").strip().lower()
        if remove_dir == 'yes':
            shutil.rmtree(OUTPUT_DIR)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        with open(os.path.join(OUTPUT_DIR, ".gitignore"), "w") as gitignore:
            gitignore.write("*\n")

def rdp(points: np.ndarray, epsilon: float) -> list:
    """
    Ramer-Douglas-Peucker polyline simplification.
    points: array of shape (N, 2 or 3)
    epsilon: distance tolerance (same units as points)
    returns: list of indices to keep
    """
    # Base case
    if len(points) < 3:
        return [0, len(points) - 1]

    start, end = points[0], points[-1]
    # Vectorized distance calculation
    line_vec = end - start
    line_len2 = np.dot(line_vec, line_vec)
    # Project each point onto the line segment
    rel = points - start
    t = np.dot(rel, line_vec) / line_len2
    t = np.clip(t, 0.0, 1.0)
    proj = start + np.outer(t, line_vec)
    dists = np.linalg.norm(points - proj, axis=1)

    idx = np.argmax(dists)
    max_dist = dists[idx]

    if max_dist > epsilon:
        left = rdp(points[:idx+1], epsilon)
        right = rdp(points[idx:], epsilon)
        # Combine, adjusting right indices
        return left[:-1] + [i + idx for i in right]
    else:
        return [0, len(points) - 1]

def simplify_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Discared points that are within epsilon meters of the line segment between the first and last point.
    Ensures a point is kept a point every MAX_TIME_GAP seconds, as long as large gap does not already exist.
    """

    # Simple method for converting to meters ok here as distances are small and not used in final simulaton
    coords = np.vstack([
        df['lat'].to_numpy() * 111000,
        df['lon'].to_numpy() * 111000 * np.cos(np.radians(df['lat'].to_numpy())),
        df['baroaltitude'].to_numpy()
    ]).T

    keep_idx = rdp(coords, RDP_EPSILON)
    keep_mask = np.zeros(len(df), dtype=bool)
    keep_mask[keep_idx] = True

    # Ensure at least one point every 60 seconds is kept
    # TODO Ideally this would be vectorised
    last_time = None
    for i, t in enumerate(df['time']):
        if last_time is None:
            last_time = t
            keep_mask[i] = True
        elif keep_mask[i]:
            last_time = t
        elif t - last_time >= MAX_TIME_GAP:
            keep_mask[i-1] = True
            last_time = df['time'][i-1]

    # print(f"Kept {np.sum(keep_mask)}/{len(df)} points of due to RDP simplification.")

    df = df.iloc[keep_mask].reset_index(drop=True)
    return df


def process_file(flight_file_path: str):
    # Load the flight data, drop NaNs and duplicates, sort by time, and rename columns
    df = pd.read_csv(os.path.join(RAW_DATA_DIR, flight_file_path))
    df.rename(columns={"lastposupdate": "time", "velocity": "gs"}, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(subset=["time"], keep="first", inplace=True)
    df.sort_values(by="time", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Discard flights with large gaps in data and flights with less than 100 points
    if len(df) < 100:
        return {"file": flight_file_path, "status": "fail_insufficient_data"}
    elif df["time"].diff().max() >= 100:
        return {"file": flight_file_path, "status": "fail_patchy_data"}

    # Run more intensive cleaning and processing operations
    df = calc_dist(df)
    df = clean_alts(df)
    df = calc_tas(df)
    df = simplify_trajectory(df)
    df = round_values(df)

    df.to_csv(os.path.join(OUTPUT_DIR, flight_file_path), index=False)
    return {"file": flight_file_path, "status": "processed"}


def init_worker(time_slice=None):
    """
    Worker initializer: load MET data (optionally with a time slice) into memory once per process.
    """
    global MET_DATA
    ds = xr.open_dataset(MET_DATA_DIR)
    if time_slice is not None:
        ds = ds.isel(time=time_slice)
    MET_DATA = ds.load()


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()  # Only necessary if creating a standalone executable

    create_save_dir()
    flight_file_paths = find_flight_files(RAW_DATA_DIR)
    MET_DATA = load_met_data(slice=TIME_SLICE)

    results = []
    print("Setting up multiprocessing, progress bar may hang for a moment...")
    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=(TIME_SLICE,)) as executor:
        for result in tqdm(executor.map(process_file, flight_file_paths), total=len(flight_file_paths), desc="Processing flights", unit="flight"):
            results.append(result)

    if results:
        failed_patchy = [r for r in results if r["status"] == "fail_patchy_data"]
        failed_length = [r for r in results if r["status"] == "fail_insufficient_data"]
        successful = [r for r in results if r["status"] == "processed"]
        print(f"Number of files that failed due to patchy data: {len(failed_patchy)}")
        print(f"Number of files that failed due to insufficient data: {len(failed_length)}")
        print(f"Number of files processed successfully: {len(successful)}")