from pyopensky.trino import Trino
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
import pyModeS as pms

trino = Trino()

FLIGHT_DATA_PATH = "output"

def load_raw_adsb():
    for filename in os.listdir(FLIGHT_DATA_PATH):
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(FLIGHT_DATA_PATH, filename))
            origin, destination, typecode, icao24, flight_num = filename.rsplit('.', 1)[0].split('_')
            start = df["time"].min()
            end = df["time"].max()
            raw_data = trino.rawdata(start, end, icao24=icao24)
            output_filename = f"{origin}_{destination}_{typecode}_{icao24}_{flight_num}_raw.csv"
            raw_data.to_csv(os.path.join(FLIGHT_DATA_PATH, output_filename), index=False)

            for row in raw_data.itertuples(index=False):

                msg = row.rawmsg
                # try:
                pms.tell(msg)
                print(pms.typecode(msg))
                # print(f"Message: {msg}") 
                # print(pms.adsb.icao(msg))
                # print(pms.adsb.velocity(msg))
                # except RuntimeError as e:
                #     print(f"{msg}: {e}")
                #     pass
            break

load_raw_adsb()