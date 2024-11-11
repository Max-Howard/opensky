from pyopensky.trino import Trino
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

trino = Trino()

scanning_stop = datetime.now()
scanning_start = scanning_stop - timedelta(days=100)
REQUESTED_COLUMNS = ["time", "lat", "lon", "velocity", "heading", "vertrate", "onground", "baroaltitude", "geoaltitude", "lastposupdate", "lastcontact", "icao24"]
PATH_FLIGHTS_TO_LOAD = "fcounts_sample.csv"

# pairs_to_test = {
#     {"name": "heathrow_denver", "departure":"EGLL","arrival": "KDEN", "typecode": "A140", "count": 1},
#     {"name": "glasgow_amsterdam", "departure" : "EGPF", "arrival" : "EHAM", "typecode": None, "count": 1}}

def test_flight_data_from_list():
    flights_to_import = pd.read_csv(PATH_FLIGHTS_TO_LOAD)
    airport_names = pd.read_csv("airports.csv")
    flights_to_import["origin_name"] = flights_to_import["origin"].map(airport_names.set_index("icao")["name"])
    flights_to_import["destination_name"] = flights_to_import["destination"].map(airport_names.set_index("icao")["name"])
    for i in range(1,len(flights_to_import)):
        flight_to_import = flights_to_import.iloc[i]
        if flight_to_import["typecode"] == np.nan:
            print(f"""Searching for {flight_to_import["count"]} flight(s) from {flight_to_import["origin_name"]} to {flight_to_import["destination_name"]} with typecode {flight_to_import['typecode']}""")
        else:
            print(f"""Searching for {flight_to_import["count"]} flight(s) from {flight_to_import["origin_name"]} to {flight_to_import["destination_name"]} without typecode""")

        # Trino seems to handle the case where the typecode is None, so we can just pass it in
        flight_start_end = trino.flightlist(
            scanning_start,
            scanning_stop,
            departure_airport=flight_to_import["origin"],
            arrival_airport=flight_to_import["destination"],
            icao24=flight_to_import["typecode"],
            limit=flight_to_import["count"])

        if type(flight_start_end) is not pd.DataFrame:
            print("Could not find flight.\n")
            continue

        print(f"Found {len(flight_start_end)} flights from {flight_to_import['origin_name']} to {flight_to_import['destination_name']}")
        print(flight_start_end)

        for i in range(len(flight_start_end)):
            data_start = flight_start_end.iloc[i, flight_start_end.columns.get_loc("firstseen")]
            data_stop = flight_start_end.iloc[i, flight_start_end.columns.get_loc("lastseen")]

            print(f"Loading path for flight number {i+1}, time period: {data_start} to {data_stop}")

            flight_path = trino.history(start=data_start,
                            stop=data_stop,
                            icao24=flight_start_end.iloc[i, flight_start_end.columns.get_loc("icao24")],
                            selected_columns=REQUESTED_COLUMNS)
            initial_length = len(flight_path)
            flight_path = flight_path.dropna().reset_index(drop=True)   # Drop rows with missing data otherwise sorting will fill with NaN
            flight_path = flight_path[flight_path['lastposupdate'].diff() != 0].reset_index(drop=True)  # Drop rows with duplicate lastposupdate
            final_length = len(flight_path)
            rows_removed = initial_length - final_length
            if rows_removed > 0:
                print(f"Removed {rows_removed} rows due to missing or repeated data")
            flight_path = flight_path.sort_values(by='time') # TODO not sure why this is necessary
            filename = f"""output/{flight_to_import["origin"]}_{flight_to_import["destination"]}_{i+1}.csv"""
            flight_path.to_csv(filename, index=False)
            print(f"""Saved {flight_to_import["origin_name"]} to {flight_to_import["destination_name"]}, flight number {i+1} as {filename}\n""")

test_flight_data_from_list()

