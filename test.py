from pyopensky.trino import Trino
import pandas as pd

trino = Trino()

start = "2022-09-01T00:00:00Z"
stop = "2022-09-02T00:00:00Z"

df = trino.flightlist(start, stop, limit=100000)

df.to_csv('output/flights.csv', index=False)