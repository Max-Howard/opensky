from pyopensky.trino import Trino
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os

trino = Trino()

scanning_stop = datetime.now()
scanning_start = scanning_stop - timedelta(days=100)
REQUESTED_COLUMNS = ["time", "lat", "lon", "velocity", "heading", "vertrate", "onground", "baroaltitude", "geoaltitude", "lastposupdate", "icao24"] # lastcontact
PATH_FLIGHTS_TO_LOAD = "fcounts_sample.csv"
PATH_AIRPORT_ICAO_NAMES = "airports.csv"
FLIGHT_DATA_PATH = "output"

# Max value for filtering out unrealistic data
MAX_VELOCITY = 500  # Maximum velocity in m/s
MAX_VERT_RATE = 100  # Maximum vertical rate in m/s

def load_fcounts(fcount_path=PATH_FLIGHTS_TO_LOAD):
    """
    Load the flight counts from CSV, and add the origin and destination ICAO names
    """
    flight_counts = pd.read_csv(fcount_path)
    airport_names = pd.read_csv(PATH_AIRPORT_ICAO_NAMES)
    flight_counts["origin_name"] = flight_counts["origin"].map(airport_names.set_index("icao")["name"])
    flight_counts["destination_name"] = flight_counts["destination"].map(airport_names.set_index("icao")["name"])
    return flight_counts

def clean_save_dir(backup=True):
    if backup and os.path.exists(FLIGHT_DATA_PATH):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"{FLIGHT_DATA_PATH}_backup_{timestamp}"
        os.rename(FLIGHT_DATA_PATH, backup_dir)
        os.makedirs(FLIGHT_DATA_PATH)
        with open(os.path.join(FLIGHT_DATA_PATH, ".gitignore"), "w") as gitignore_file:
            gitignore_file.write("*")
        print(f"Moved {FLIGHT_DATA_PATH} to {backup_dir}")
    elif os.path.exists(FLIGHT_DATA_PATH):
        input(f"This will delete all files in the {FLIGHT_DATA_PATH} directory. Press enter to continue.")
        for filename in os.listdir(FLIGHT_DATA_PATH):
            file_path = os.path.join(FLIGHT_DATA_PATH, filename)
            if filename != ".gitignore" and os.path.isfile(file_path):
                os.remove(file_path)
        print(f"Deleted all files in {FLIGHT_DATA_PATH}")
    else:
        os.makedirs(FLIGHT_DATA_PATH)
        with open(os.path.join(FLIGHT_DATA_PATH, ".gitignore"), "w") as gitignore_file:
            gitignore_file.write("*")
        print(f"Created {FLIGHT_DATA_PATH}")

def clean_flight_data(flight_path: pd.DataFrame):
    initial_length = len(flight_path)
    flight_path = flight_path.dropna()                                  # Drop rows with missing data
    flight_path = flight_path[flight_path['lastposupdate'].diff() != 0] # Drop rows with without position update
    flight_path = flight_path.drop(columns=['lastposupdate'])           # lastposupdate is no longer needed
    flight_path = flight_path.sort_values(by='time')                    # Sort by time
    flight_path = flight_path.reset_index(drop=True)                    # Reset index
    rows_removed = initial_length - len(flight_path)
    if rows_removed > 0:
        print(f"Removed {rows_removed} of {initial_length} rows due to missing or repeated data")

    # Remove rows where the data makes a jump larger than normal
    # TODO implement a more sophisticated filter including other parameters

    time_gap = flight_path['time'].diff().dt.total_seconds()

    # Calculate the altitude change between consecutive rows
    geo_altitude_diff = flight_path['geoaltitude'].diff()
    baro_altitude_diff = flight_path['baroaltitude'].diff()
    geo_altitude_rate = geo_altitude_diff / time_gap
    baro_altitude_rate = baro_altitude_diff / time_gap
    altitude_change_rate = np.maximum(geo_altitude_rate, baro_altitude_rate)

    # Filter out rows with unrealistic jumps
    initial_length = len(flight_path)
    flight_path = flight_path[altitude_change_rate <= MAX_VERT_RATE]
    rows_removed = initial_length - len(flight_path)
    if rows_removed > 0:
        print(f"Removed {rows_removed} rows due to unrealistic altitude change rate")

    return flight_path

def load_flight_adsb(flight_counts):

    for i in range(len(flight_counts)):
        flight_to_import = flight_counts.iloc[i]

        if isinstance(flight_to_import["typecode"], float): # TODO this can be improved
            if np.isnan(flight_to_import["typecode"]):  # NaN indicates no specified typecode
                print(f"Searching for {flight_to_import['count']} flight(s) ",
                    f"from {flight_to_import['origin_name']} to ",
                    f"{flight_to_import['destination_name']} without typecode")
            else:
                raise NotImplementedError("Typecode is a float, but not NaN")
        else:
            raise NotImplementedError("Cannot sort by typecode")
            print(f"Searching for {flight_to_import['count']} flight(s) ",
                  f"from {flight_to_import['origin_name']} to ",
                  f"{flight_to_import['destination_name']} ",
                  f"with typecode {flight_to_import['typecode']}")

        # Trino seems to handle the case where the typecode is None, so we can just pass it in
        flight_durations = trino.flightlist(
            scanning_start,
            scanning_stop,
            departure_airport=flight_to_import["origin"],
            arrival_airport=flight_to_import["destination"],
            icao24=flight_to_import["typecode"],
            limit=flight_to_import["count"])    # Limit the number of flights to import to the count in the CSV

        if type(flight_durations) is not pd.DataFrame:
            print("Could not find flight.\n")
            continue

        print(f"Found {len(flight_durations)}/{flight_to_import['count']} flights ",
              f"from {flight_to_import['origin_name']} to {flight_to_import['destination_name']}")

        for i in range(len(flight_durations)):
            data_start = flight_durations.iloc[i, flight_durations.columns.get_loc("firstseen")]
            data_stop = flight_durations.iloc[i, flight_durations.columns.get_loc("lastseen")]

            print(f"Loading path for flight number {i+1}, time period: {data_start} to {data_stop}")

            flight_path = trino.history(start=data_start,
                            stop=data_stop,
                            icao24=flight_durations.iloc[i, flight_durations.columns.get_loc("icao24")],
                            selected_columns=REQUESTED_COLUMNS)
            
            flight_path = clean_flight_data(flight_path)

            flight_path["origin"] = flight_to_import["origin"]
            flight_path["destination"] = flight_to_import["destination"]
            # TODO store the typecode in the dataframe as well

            filename = f"""{FLIGHT_DATA_PATH}/{flight_to_import["origin"]}_{flight_to_import["destination"]}_{i+1}.csv"""
            flight_path.to_csv(filename, index=False)
            print(f"""Saved {flight_to_import["origin_name"]} to {flight_to_import["destination_name"]}, flight number {i+1} as {filename}\n""")

clean_save_dir()
flight_counts = load_fcounts()
load_flight_adsb(flight_counts)

