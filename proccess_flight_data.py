import os
import pandas as pd
import numpy as np
import shutil
import xarray as xr
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Non flight data
MET_DATA_DIR: str = "./met/wind_monthly_202411.nc4"
MET_DATA = None
AIRPORT_DATA_DIR: str = "./met/airports.csv"
AIRPORT_DATA = None

# Flight data
RAW_DATA_DIR = "RawFlightData"
OUTPUT_DIR = "ProcessedFlightData"

# Tolerances to determine considered anomalous points
V_MAX = 500         # Max velocity (m/s)
ROCD_MAX = 50        # Max rate of climb/descent (m/s)

# Tolerance for removing points
RDP_EPSILON = 10          # RDP tolerance in (m)

# Tolerance for dropping flights
MAX_TIME_GAP = 60           # Maximum time gap in seconds between points (s)
TOL_DIST_START_END = 10000  # Max distance flight can start/end from apt (m)
TOL_ALT_START_END = 1000    # Max height above airport at start and end of data (m)

TIME_SLICE = slice(0, 16) # Data slicing needed to reduce memory usage when multiprocessing
MAX_WORKERS = 6

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

    indices_to_drop = altitude_change_rate[altitude_change_rate > ROCD_MAX].index
    df.drop(indices_to_drop, inplace=True) # TODO should this be idx + 1?

    threshold = 50
    for col in ['geoaltitude', 'baroaltitude']:
        jumps = df[col].diff().abs() > threshold
        returns = df[col].diff(-1).abs() > threshold
        indices_to_drop = df[jumps & returns].index
        df.drop(indices_to_drop, inplace=True)

    df.reset_index(drop=True, inplace=True)
    return df

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    if "lastposupdate" in df.columns:
        raise ValueError("DataFrame contains 'lastposupdate' column, this should be used as time.")
    df.dropna(inplace=True)
    df.drop_duplicates(subset=["time"], keep="first", inplace=True)
    df.sort_values(by="time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def remove_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows where each of lat, lon, baroaltitude, geoaltitude
    don't cause velocity or rate of climb/descent to exceed the given tolerances.
    """
    if "lastposupdate" in df.columns:
        raise ValueError("DataFrame contains 'lastposupdate' column, this should be used as time.")

    kept_idx = []
    last_vals = {}
    # before = len(df)
    for idx, row in df.iterrows():
        lat, lon = row["lat"], row["lon"]
        baro, geo = row["baroaltitude"], row["geoaltitude"]
        time = row["time"]

        if not last_vals: # Setup initial values
            kept_idx.append(idx)
            last_vals = dict(lat=lat, lon=lon, baro=baro, geo=geo, time=time)
            continue

        d_lat  = abs(lat  - last_vals["lat"])
        d_lon  = abs(lon  - last_vals["lon"])
        d_baro = abs(baro - last_vals["baro"])
        d_geo  = abs(geo  - last_vals["geo"])
        d_time = time - last_vals["time"]

        tol_lat = V_MAX * d_time / 111000
        tol_lon = V_MAX * d_time / (111000 * np.cos(np.radians(last_vals["lat"])))
        tol_alt = ROCD_MAX * d_time

        if (d_lat  <= tol_lat  and
            d_lon  <= tol_lon and
            d_baro <= tol_alt and
            d_geo  <= tol_alt):
            kept_idx.append(idx)
            last_vals = dict(lat=lat, lon=lon, baro=baro, geo=geo, time=time)
    # print(f"Removed {before - len(kept_idx)} points due to anomalies.")
    return df.loc[kept_idx].reset_index(drop=True)


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
    cumulative_dist = np.cumsum(dist)
    df['dist'] = np.round(cumulative_dist, 1)
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

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = np.radians(lon2 - lon1)
    a = np.sin(d_phi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(d_lambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

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

    # TODO Ideally this would be vectorised
    last_time = None
    for i, t in enumerate(df['time']):
        if last_time is None:
            last_time = t
            keep_mask[i] = True
        elif keep_mask[i]:
            last_time = t
        elif t - last_time >= MAX_TIME_GAP/2:
            keep_mask[i-1] = True
            last_time = df['time'][i-1]

    # print(f"Kept {np.sum(keep_mask)}/{len(df)} points of due to RDP simplification.")

    df = df.iloc[keep_mask].reset_index(drop=True)
    return df


def process_file(flight_file_path: str):

    df = pd.read_csv(os.path.join(RAW_DATA_DIR, flight_file_path))
    origin, destination, typecode, icao24, flight_number = flight_file_path.strip(".csv").split("_")

    df.rename(columns={"lastposupdate": "time", "velocity": "gs"}, inplace=True)
    df = clean_dataset(df)

    if len(df) < 1000:
        return {"file": flight_file_path, "status": "fail_insufficient_data", "len": len(df)}

    pre_rdp_len = len(df)
    df = clean_alts(df) # This is done before RDP to avoid avoid creating large gaps later on (RDP does not account for GNSS altitude)
    df = simplify_trajectory(df)
    df = remove_anomalies(df)

    if df["time"].diff().max() > MAX_TIME_GAP:
        return {"file": flight_file_path, "status": "fail_patchy_data", "len": len(df)}

    if df["baroaltitude"].max() < 2000:
        return {"file": flight_file_path, "status": "fail_low_altitude", "len": len(df)}

    origin_airport_data = AIRPORT_DATA.loc[origin]
    destination_airport_data = AIRPORT_DATA.loc[destination]

    # Check if flight starts and ends near airport
    start_dist = haversine(df.iloc[0]['lat'], df.iloc[0]['lon'],origin_airport_data['lat'], origin_airport_data['lon'])
    end_dist = haversine(df.iloc[-1]['lat'], df.iloc[-1]['lon'],destination_airport_data['lat'], destination_airport_data['lon'])
    if start_dist > TOL_DIST_START_END or end_dist > TOL_DIST_START_END:
        return {"file": flight_file_path, "status": "fail_missing_start_end", "len": len(df)}
    start_height_above_apt = df["baroaltitude"].iloc[0] - origin_airport_data["alt"]
    end_height_above_apt = df["baroaltitude"].iloc[0] - destination_airport_data["alt"]
    if max(start_height_above_apt, end_height_above_apt) > TOL_ALT_START_END:
        return {"file": flight_file_path, "status": "fail_no_low_altitude_start_end", "len": len(df)}

    df = calc_dist(df)

    if df["dist"].iloc[-1] < 10000:
        return {"file": flight_file_path, "status": "fail_low_distance"}

    df = calc_tas(df)
    df = round_values(df)

    df.to_csv(os.path.join(OUTPUT_DIR, flight_file_path), index=False)
    return {"file": flight_file_path, "status": "processed", "len": len(df), "rdp_dropped": pre_rdp_len - len(df)}


def init_worker(time_slice=None):
    """
    Worker initializer: load MET data (optionally with a time slice) into memory once per process.
    """
    global MET_DATA
    global AIRPORT_DATA
    AIRPORT_DATA = pd.read_csv("airports.csv").set_index("icao").copy()
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
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=init_worker, initargs=(TIME_SLICE,)) as executor:
        for result in tqdm(executor.map(process_file, flight_file_paths), total=len(flight_file_paths), desc="Processing flights", unit="flight"):
            results.append(result)

    if results:
        failed_patchy = [r for r in results if r["status"] == "fail_patchy_data"]
        failed_length = [r for r in results if r["status"] == "fail_insufficient_data"]
        failed_low_altitude = [r for r in results if r["status"] == "fail_low_altitude"]
        failed_low_distance = [r for r in results if r["status"] == "fail_low_distance"]
        failed_no_low_altitude_start_end = [r for r in results if r["status"] == "fail_no_low_altitude_start_end"]
        failed_missing_start_end = [r for r in results if r["status"] == "fail_missing_start_end"]
        successful = [r for r in results if r["status"] == "processed"]
        num_points_successful = sum(r["len"] for r in successful)
        num_points_rdp_dropped = sum(r["rdp_dropped"] for r in successful)
        print(f"Number of files that failed due to patchy data: {len(failed_patchy)}")
        print(f"Number of files that failed due to insufficient data: {len(failed_length)}")
        print(f"Number of files that failed due to low max altitude: {len(failed_low_altitude)}")
        print(f"Number of files that failed due to low distance: {len(failed_low_distance)}")
        print(f"Number of files that failed due to no low altitude at start/end: {len(failed_no_low_altitude_start_end)}")
        print(f"Number of files that failed due to missing start/end: {len(failed_missing_start_end)}")
        print(f"Number of files processed successfully: {len(successful)}")
        print(f"Number of points processed successfully: {num_points_successful}")
        print(f"Number of points dropped due to RDP: {num_points_rdp_dropped}")