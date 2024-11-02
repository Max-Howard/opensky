from pyopensky.trino import Trino
import pandas as pd

trino = Trino()
# full description of the whole set of parameters in the documentation
# trino.flightlist(start, stop, *, airport, callsign, icao24)
# trino.history(start, stop, *, callsign, icao24, bounds)
# trino.rawdata(start, stop, *, callsign, icao24, bounds)

start = "2021-09-01T00:00:00Z"
stop = "2021-09-02T00:00:00Z"

flights = trino.history(start, stop)
print("Number of flights:", len(flights))

# Since flights is already a dataframe, you can directly save it to a CSV file
flights.to_csv('flights.csv', index=False)

