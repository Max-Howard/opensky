import os
import pandas as pd
import numpy as np
import shutil
import xarray as xr
from tqdm import tqdm

MET_DATA_DIR: str = "./met/wind_monthly_202411.nc4"
MET_DATA = None
RAW_DATA_DIR = "RawFlightData"
OUTPUT_DIR = "ProcessedFlightData"

def find_flight_files(directory: str):
    flight_files = []
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            flight_files.append(file)
    return flight_files

def load_met_data(met_data_dir:str = MET_DATA_DIR):
    print("Loading MET data... ", end="", flush=True)
    # MET_DATA = xr.open_dataset(met_data_dir)
    MET_DATA = xr.open_dataset(met_data_dir).load() # Load the data into RAM
    print("done.")
    return MET_DATA

def clean_alts(df: pd.DataFrame) -> pd.DataFrame:
    geo_altitude_diff = df['geoaltitude'].diff()
    baro_altitude_diff = df['baroaltitude'].diff()
    time_gap = df['time'].diff()
    geo_altitude_rate = geo_altitude_diff / time_gap
    baro_altitude_rate = baro_altitude_diff / time_gap
    altitude_change_rate = np.maximum(geo_altitude_rate, baro_altitude_rate)

    # Identify indices where the altitude change rate exceeds the threshold
    indices_to_drop = altitude_change_rate[altitude_change_rate > 25].index
    # indices_to_drop = indices_to_drop.union(indices_to_drop + 1)
    df.drop(indices_to_drop, inplace=True)

    # Additional filtering: Remove rows with sharp jumps that return back to close to the previous value
    threshold = 50  # Define a threshold for what constitutes a sharp jump
    for col in ['geoaltitude', 'baroaltitude']:
        jumps = df[col].diff().abs() > threshold
        returns = df[col].diff(-1).abs() > threshold
        indices_to_drop = df[jumps & returns].index
        df.drop(indices_to_drop, inplace=True)

    df.reset_index(drop=True, inplace=True)
    return df

def round_values(df:pd.DataFrame) -> pd.DataFrame:
    df['time'] = df['time'].round(2)
    df['lat'] = df['lat'].round(6)
    df['lon'] = df['lon'].round(6)
    df['geoaltitude'] = df['geoaltitude'].round(1)
    df['baroaltitude'] = df['baroaltitude'].round(1)
    df['gs'] = df['gs'].round(1)
    df['heading'] = df['heading'].round(1)
    df['vertrate'] = df['vertrate'].round(1)

    # Round MET data if available
    if 'tas' in df.columns:
        df['tas'] = df['tas'].round(1)
    if 'wind_speed' in df.columns:
        df['wind_speed'] = df['wind_speed'].round(1)
    if 'wind_dir' in df.columns:
        df['wind_dir'] = df['wind_dir'].round(1)
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

def process_file(flight_file_path: str):
    # Load the flight data, drop NaNs and duplicates, sort by time, and rename columns
    df = pd.read_csv(os.path.join(RAW_DATA_DIR, flight_file_path))
    df.rename(columns={"lastposupdate": "time", "velocity": "gs"}, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(subset=["time"], keep="first", inplace=True)
    df.sort_values(by="time", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Skip bad files
    if len(df) < 100:
        return flight_file_path
    if max(df["time"].diff()) >= 100:
        return flight_file_path

    # Run more intensive cleaning and processing operations
    df = calc_dist(df)
    df = clean_alts(df)
    df = calc_tas(df)
    df = round_values(df)

    # Remove duplicates again after rounding
    df.drop_duplicates(subset=["time"], keep="first", inplace=True)

    output_filepath = os.path.join(OUTPUT_DIR, flight_file_path)
    df.to_csv(output_filepath, index=False)
    return None


if __name__ == "__main__":
    create_save_dir()
    flight_file_paths = find_flight_files(RAW_DATA_DIR)
    MET_DATA = load_met_data()

    skipped_files = []
    for flight_file_path in tqdm(flight_file_paths, desc="Processing flight files", unit="flights"):
        skipped_file = process_file(flight_file_path)
        if skipped_file:
            skipped_files.append(skipped_file)

    if skipped_files:
        print("\nSkipped files due to insufficient data points or patchy data:")
        for skipped_file in skipped_files:
            print(f"- {skipped_file}")