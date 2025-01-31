import pandas as pd


def clean_aircraft_db():
    """
    Used to clean the OpenSky aircraft database
    """
    # columns_to_import = ['icao24','timestamp','acars','adsb','built','categoryDescription','country','engines','firstFlightDate','firstSeen','icaoAircraftClass','lineNumber','manufacturerIcao','manufacturerName','model','modes','nextReg','operator','operatorCallsign','operatorIata','operatorIcao','owner','prevReg','regUntil','registered','registration','selCal','serialNumber','status','typecode','vdl']
    columns_to_import = ['icao24','typecode','engines','icaoAircraftClass','model','modes','operator','serialNumber']
    aircraft_db = pd.read_csv("aircraft.csv", usecols=columns_to_import, on_bad_lines='warn', quotechar="'")

    initial_length = len(aircraft_db)
    aircraft_db = aircraft_db.dropna(subset=["typecode"])
    rows_removed = initial_length - len(aircraft_db)
    print(f"Removed {rows_removed} of {initial_length} rows, due to missing typecode")
    aircraft_db = aircraft_db.set_index("icao24")
    aircraft_db.to_csv("aircraft_db_clean.csv")

clean_aircraft_db()