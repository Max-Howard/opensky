from pyopensky.trino import Trino
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os

trino = Trino()

scanning_stop = datetime.now()
scanning_start = scanning_stop - timedelta(days=100)
AIRCRAFT_DB = pd.read_csv("aircraft_db_clean.csv")
AIRPORT_DB = pd.read_csv("airports.csv")
REQUESTED_COLUMNS = ["lastposupdate", "lat", "lon", "velocity", "heading", "vertrate", "baroaltitude", "geoaltitude"] # "time" ,lastcontact,  "onground", "icao24"
PATH_FLIGHTS_TO_LOAD = "fcounts_sample.csv"
FLIGHT_DATA_PATH = "output"

# Max value for filtering out unrealistic data
MAX_VELOCITY = 500  # Maximum velocity in m/s
MAX_VERT_RATE = 100  # Maximum vertical rate in m/s

def find_bada_typecodes(bada_path = "./BADA"):
    typecodes = []
    with open(os.path.join(bada_path, "ReleaseSummary")) as file:
        lines = file.readlines()
        lines = lines[9:]
        for line in lines:
            if line[7:10] == "PTF":
                typecode = line[:4].strip("_")
                typecodes.append(typecode)
    return typecodes

BADA_TYPECODES = find_bada_typecodes()


def is_typecode(typecode):
    try:
        num = float(typecode)
    except (ValueError, TypeError):
        # TODO check if typecode is in the aircraft database here
        return True
    return not np.isnan(num)

def load_fcounts(fcount_path=PATH_FLIGHTS_TO_LOAD):
    """
    Load the flight counts from CSV, and add the origin and destination ICAO names
    """
    flight_counts = pd.read_csv(fcount_path)
    flight_counts["origin_name"] = flight_counts["origin"].map(AIRPORT_DB.set_index("icao")["name"])
    flight_counts["destination_name"] = flight_counts["destination"].map(AIRPORT_DB.set_index("icao")["name"])
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
        print(f"Created {FLIGHT_DATA_PATH} folder for storing flight data")

def clean_flight_data(flight_path: pd.DataFrame):
    initial_length = len(flight_path)
    flight_path = flight_path.dropna()                                  # Drop rows with missing data
    len_after_empty_rm = len(flight_path)
    flight_path = flight_path.sort_values(by='lastposupdate')           # Sort by time
    flight_path = flight_path.drop_duplicates(subset=["lastposupdate"], keep="first") # Drop duplicate rows with without position update
    flight_path = flight_path.reset_index(drop=True)                    # Reset index
    len_after_duplicates_rm = len(flight_path)

    # Remove rows where the data makes a jump larger than normal
    # TODO implement a more sophisticated filter including other parameters

    time_gap = flight_path['lastposupdate'].diff()

    # Calculate the altitude change between consecutive rows
    geo_altitude_diff = flight_path['geoaltitude'].diff()
    baro_altitude_diff = flight_path['baroaltitude'].diff()
    geo_altitude_rate = geo_altitude_diff / time_gap
    baro_altitude_rate = baro_altitude_diff / time_gap
    altitude_change_rate = np.maximum(geo_altitude_rate, baro_altitude_rate)

    # Filter out rows with unrealistic jumps
    flight_path = flight_path[altitude_change_rate <= MAX_VERT_RATE]
    len_after_anomalies_rm = len(flight_path)

    if initial_length - len_after_anomalies_rm > 0:
        rm_empty = initial_length - len_after_empty_rm
        rm_duplicates = len_after_empty_rm - len_after_duplicates_rm
        rm_anomalies = len_after_duplicates_rm - len_after_anomalies_rm
        print(f"Initial length: {initial_length}, final length: {len_after_anomalies_rm}, total removed: {initial_length - len_after_anomalies_rm}")
        print(f"Removed {rm_empty} rows due to missing data, {rm_duplicates} rows due to duplicate data and {rm_anomalies} rows due to anomalies")

    return flight_path

def flight_duration_from_fcount(flight_to_import):

    if not is_typecode(flight_to_import["typecode"]):
        print(f"Searching for {flight_to_import['count']} flight(s) ",
            f"from {flight_to_import['origin_name']} to ",
            f"{flight_to_import['destination_name']} without typecode specified")
        flight_durations = trino.flightlist(
            scanning_start,
            scanning_stop,
            departure_airport=flight_to_import["origin"],
            arrival_airport=flight_to_import["destination"],
            limit=flight_to_import["count"]*2)
        if type(flight_durations) is not pd.DataFrame:
                print("Could not find flight.\n")
                return None
        flight_durations = flight_durations.merge(AIRCRAFT_DB[['icao24', 'typecode']],on='icao24',how='left')
        initial_length = len(flight_durations)
        flight_durations = flight_durations.dropna(subset=['typecode'])
        final_length = len(flight_durations)
        removed_count = initial_length - final_length
        if removed_count > 0:
            print(f"Removed {removed_count} flights due to missing typecode in database.")
        if len(flight_durations) >= flight_to_import["count"]:
            flight_durations = flight_durations.head(flight_to_import["count"])
        print(f"Found {len(flight_durations)}/{flight_to_import['count']} flights, with typecodes: {flight_durations['typecode'].unique()}\n")

    else:
        print(f"Searching for {flight_to_import['count']} flight(s)",
                f"from {flight_to_import['origin_name']} to",
                f"{flight_to_import['destination_name']}",
                f"with typecode {flight_to_import['typecode']}")
        number_to_import = max(flight_to_import["count"]*5, 50) # Need to import extra to find the correct typecode

        flight_durations = trino.flightlist(
            scanning_start,
            scanning_stop,
            departure_airport=flight_to_import["origin"],
            arrival_airport=flight_to_import["destination"],
            limit=number_to_import)

        if flight_durations is None:
            print("Could not find flight(s).\n")
            return None
        else:
            list_start = flight_durations['firstseen'].min()
            list_stop = flight_durations['lastseen'].max()
            flight_durations = flight_durations.merge(
                AIRCRAFT_DB[['icao24', 'typecode']],
                on='icao24',
                how='left'
            )
            num_found = len(flight_durations)
            flight_durations = flight_durations[flight_durations['typecode'] == flight_to_import['typecode']]

            if len(flight_durations) >= flight_to_import["count"]:
                flight_durations = flight_durations.head(flight_to_import["count"])
                print("Found all flights.")
            else:
                print(f"Found {len(flight_durations)}/{flight_to_import['count']} flights.")
                if num_found >= number_to_import:
                    print(f"Limited by max number of flights to import ({number_to_import}). Time period due to limit: {list_start} to {list_stop}.\n")
                else:
                    print(f"Limited by time period. Time period: {list_start} to {list_stop}. Only searched {num_found} flights, out of {number_to_import} limit.\n")

    flight_durations = flight_durations.rename(columns={"departure": "origin", "arrival": "destination"})
    flight_durations["origin_name"] = flight_durations["origin"].map(AIRPORT_DB.set_index("icao")["name"])
    flight_durations["destination_name"] = flight_durations["destination"].map(AIRPORT_DB.set_index("icao")["name"])
    return flight_durations


def random_flights(limit=10):
    flight_durations = trino.flightlist(
        scanning_start,
        scanning_stop,
        limit=limit)
    num_found = len(flight_durations)
    flight_durations = flight_durations.dropna() # subset=['departure', 'arrival']
    after_na = len(flight_durations)
    flight_durations = flight_durations.merge(AIRCRAFT_DB[['icao24', 'typecode']], on='icao24', how='left')
    flight_durations = flight_durations[flight_durations['typecode'].isin(BADA_TYPECODES)]
    after_typecode = len(flight_durations)
    flight_durations = flight_durations.rename(columns={"departure": "origin", "arrival": "destination"})
    flight_durations["origin_name"] = flight_durations["origin"].map(AIRPORT_DB.set_index("icao")["name"])
    flight_durations["destination_name"] = flight_durations["destination"].map(AIRPORT_DB.set_index("icao")["name"])
    print(f"Searched for {limit} random flights, found {num_found} flights, after removing NA: {after_na}, after removing non-BADA typecodes: {after_typecode}")
    return flight_durations


def load_adsb_from_durations(flight_durations):
    for i in range(len(flight_durations)):
        flight_to_import = flight_durations.iloc[i]

        flight_path = trino.history(start=flight_to_import["firstseen"],
                        stop=flight_to_import["lastseen"],
                        icao24=flight_to_import["icao24"],
                        selected_columns=REQUESTED_COLUMNS)

        flight_path = clean_flight_data(flight_path)
        flight_path = flight_path.rename(columns={"lastposupdate": "time"})
        filename = f"""{FLIGHT_DATA_PATH}/{flight_to_import["origin"]}_{flight_to_import["destination"]}_{flight_to_import["typecode"]}_{flight_to_import["icao24"]}_{i+1}.csv"""
        flight_path.to_csv(filename, index=False)
        print(f"""Saved {flight_to_import["origin_name"]} to {flight_to_import["destination_name"]}, flight number {i+1} as {filename}\n""")


clean_save_dir()

flight_counts = load_fcounts()
for i in range(len(flight_counts)):
    flight_to_import = flight_counts.iloc[i]
    flight_durations = flight_duration_from_fcount(flight_to_import)
    load_adsb_from_durations(flight_durations)


# flight_durations = random_flights()
# load_adsb_from_durations(flight_durations)