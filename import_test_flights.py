from pyopensky.trino import Trino
from datetime import datetime, timedelta

trino = Trino()

scanning_stop = datetime.now()
scanning_start = scanning_stop - timedelta(days=10)
REQUESTED_COLUMNS = ["time", "lat", "lon", "velocity", "heading", "vertrate", "onground", "baroaltitude", "geoaltitude", "lastposupdate", "lastcontact", "icao24"]

pairs_to_test = {
    "heathrow_denver": {"departure":"EGLL","arrival": "KDEN"},
    "glasgow_amsterdam" : {"departure" : "EGPF", "arrival" : "EHAM"}
    }

def test_flight_data_from_list(pairs_to_test):
    for pair_name in pairs_to_test.keys():
        flight = pairs_to_test[pair_name]
        df = trino.flightlist(
            scanning_start,
            scanning_stop,
            departure_airport=flight["departure"],
            arrival_airport=flight["arrival"],
            limit=1)
        data_start = df["firstseen"][0]
        data_stop = df["lastseen"][0]
        icao24 = df["icao24"][0]
        print(f"Found flight for: {pair_name}, loading data from {data_start} to {data_stop}")

        df = trino.history(start=data_start,
                        stop=data_stop,
                        icao24=icao24,
                        selected_columns=REQUESTED_COLUMNS)
        df.to_csv(f'output/{pair_name}.csv', index=False)
        print(f"Saved {pair_name}.csv")

def import_flightlist(pairs_to_test):
    for pair_name in pairs_to_test.keys():
            flight = pairs_to_test[pair_name]
            df = trino.flightlist(
                scanning_start,
                scanning_stop,
                departure_airport=flight["departure"],
                arrival_airport=flight["arrival"],
                limit=10)
            df.to_csv(f'output/lists/{pair_name}.csv', index=False)

test_flight_data_from_list(pairs_to_test)

