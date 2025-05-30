import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from geographiclib.geodesic import Geodesic
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

geod = Geodesic.WGS84

FLIGHTS_DIR = "./ProcessedFlightData"

FT_TO_M = 0.3048
KTS_TO_MPS = 0.514444
TIME_GAP_CRUISE = 60
ALT_GAP_MAX = 250  # Maximum altitude gap in meters before interpolation
TIME_GAP_ALT_GAP = 10
CRUISE_STEP = 50 * 1852  # 50 nautical miles in meters
LTO_CEILING = 3000 * FT_TO_M
AIRPORT_NAMES = pd.read_csv("airports.csv").set_index("icao")
AC_CRUISE_LEVELS = pd.read_csv("cruise_fl.csv").set_index("typecode")
MET_DATA = xr.open_dataset("./met/wind_monthly_202411.nc4")
BADA_PLOT_TIME_STEP = 1


def fixed_spacing_floats(s, fmt, nan_value=np.nan):
    """
    Return floats contained in given positions of string s

    Parameters
    ----------
    s : str
        String to parse.
    fmt : list of int
        Alternating gaps and lengths, starting with a gap.
    nan_value : ?, optional
        Value to be returned for failed casts to float. The default is np.nan.

    Raises
    ------
    ValueError
        If len(fmt) < 2.

    Returns
    -------
    values : list of float
        Extracted values.

    """
    if len(fmt) < 2:
        raise ValueError("fmt must have length >= 2")

    values = []
    pos = 0
    for i in range(int(len(fmt) / 2)):
        pos += fmt[i * 2]
        length = fmt[i * 2 + 1]
        value = s[pos : pos + length]
        try:
            value = float(value)
        except ValueError:
            value = nan_value
        values.append(value)
        pos += length

    return values


def read_ptf(filepath):
    """
    Read a Performance Table File for a given aircraft

    Parameters
    ----------
    filepath : str
        File to read.

    Returns
    -------
    ptf : dict
        Extracted data.

    """
    ptf = {}
    with open(filepath, "r", encoding="cp1252") as f:
        lines = f.readlines()

        # Header
        line = lines[7]
        ptf["Vcl1"] = float(line[11:14])
        ptf["Vcl2"] = float(line[15:18])
        ptf["Mcl"] = float(line[23:27])
        ptf["mlo"] = float(line[41:])

        line = lines[8]
        ptf["Vcr1"] = float(line[11:14])
        ptf["Vcr2"] = float(line[15:18])
        ptf["Mcr"] = float(line[23:27])
        ptf["mnom"] = float(line[41:54])
        ptf["hMO"] = float(line[71:])

        line = lines[9]
        ptf["Vdes1"] = float(line[11:14])
        ptf["Vdes2"] = float(line[15:18])
        ptf["Mdes"] = float(line[23:27])
        ptf["mhi"] = float(line[41:])

        # Table
        table = []
        irow = 0
        line = lines[16]
        while line[0] != "=":
            values = fixed_spacing_floats(line, [0, 3, 4, 3, 3, 5, 1, 5, 1, 5, 5, 3, 3, 5, 1, 5, 1, 5, 3, 5, 5, 3, 2, 5, 2, 5])
            table.append(values)
            irow += 1
            line = lines[16 + irow * 2]
        ptf["table"] = pd.DataFrame(
            data=table,
            columns=[
                "FL",
                "Vcr",
                "flo_cr",
                "fnom_cr",
                "fhi_cr",
                "Vcl",
                "ROCDlo_cl",
                "ROCDnom_cl",
                "ROCDhi_cl",
                "fnom_cl",
                "Vdes",
                "ROCDnom_des",
                "fnom_des",
            ],
        ).set_index("FL")

    return ptf


class FlightPath:
    def __init__(self, filename, flights_dir=FLIGHTS_DIR):
        filepath = os.path.join(flights_dir, filename)
        if filepath.endswith(".csv"):
            self.df = pd.read_csv(filepath)
        elif filepath.endswith(".pkl"):
            self.df = pd.read_pickle(filepath)
        else:
            raise ValueError("File must be either a .csv or .pkl file.")
        self.filepath = filepath
        origin, destination, typecode, icao24, flight_num = filename.rsplit(".", 1)[0].split("_")
        self.origin_icao = origin
        self.destination_icao = destination
        self.typecode = typecode
        self.icao24 = icao24
        self.flight_num = flight_num
        self.origin_ap_data = AIRPORT_NAMES.loc[self.origin_icao]
        self.destination_ap_data = AIRPORT_NAMES.loc[self.destination_icao]
        self.origin_name = self.origin_ap_data["name"]
        self.destination_name = self.destination_ap_data["name"]
        self.ac_data = read_ptf(f"./BADA/{self.typecode}__.PTF")
        self.bada_df = None
        self.interp_df = None
        self.time_gaps = self.df["time"].diff()
        if max(max(self.df["baroaltitude"]), max(self.df["geoaltitude"])) > self.ac_data["hMO"]:
            input(f"Warning: Flight {self.flight_num} exceeds maximum altitude of {self.ac_data['hMO']}, please acknowledge.")
        print(f"Flight imported from {filename}.")

    def __repr__(self):
        takeoff_time = pd.to_datetime(self.df['time'].iloc[0], unit='s')
        landing_time = pd.to_datetime(self.df['time'].iloc[-1], unit='s')
        duration = (landing_time - takeoff_time).total_seconds()
        duration_hours = int(duration // 3600)
        duration_minutes = int((duration % 3600) // 60)

        return f"\nFlightPath Object\nFrom: {self.origin_name}, To: {self.destination_name}, Typecode: {self.typecode}\nTakeoff: {takeoff_time.strftime('%d-%m-%Y %H:%M')}, Landing: {landing_time.strftime('%d-%m-%Y %H:%M')}, Duration: {duration_hours}:{duration_minutes}\nicao24: {self.icao24}, flight number: {self.flight_num}, containing {len(self.df)} rows."

    def combine_df(self):
        if self.interp_df is None:
            print("No interpolated data found, returning real data.")
            combined_df = self.df.copy()
            combined_df["interpolated"] = False
        else:
            interplated_df = self.interp_df.copy()
            real_df = self.df.copy()
            interplated_df["interpolated"] = True
            real_df["interpolated"] = False
            combined_df = pd.concat([interplated_df, real_df], ignore_index=True)
            combined_df.sort_values(by="time", inplace=True)
            combined_df.reset_index(drop=True, inplace=True)
        return combined_df

    def find_ac_perf(self, altitude, interp=True) -> pd.Series:
        """
        Find the aircraft performance at current altitude.
        """
        current_FL = (altitude / FT_TO_M) / 100
        if interp:
            if not (self.ac_data["table"].index <= current_FL).any():
                input(f"Trying to find BADA performance at altitude {altitude} m, which is outside the range of the BADA table. Please acknowledge.")
                return self.ac_data["table"].iloc[0]
            elif not (self.ac_data["table"].index >= current_FL).any():
                input(f"Trying to find BADA performance at altitude {altitude} m, which is outside the range of the BADA table. Please acknowledge.")
                return self.ac_data["table"].iloc[-1]
            lower_FL = self.ac_data["table"].index[self.ac_data["table"].index <= current_FL].max()
            upper_FL = self.ac_data["table"].index[self.ac_data["table"].index >= current_FL].min()
            lower_row = self.ac_data["table"].loc[lower_FL]
            upper_row = self.ac_data["table"].loc[upper_FL]

            if lower_FL == upper_FL:    # Handle the case where the current altitude is exactly at a FL
                return lower_row
            else:
                interp_row = lower_row + (upper_row - lower_row) * ((current_FL - lower_FL) / (upper_FL - lower_FL))
                return interp_row
        else:
            closest_FL = self.ac_data["table"].index[np.abs(self.ac_data["table"].index - current_FL).argmin()]
            return self.ac_data["table"].loc[closest_FL]

    def calc_bada_path(self):
        """
        Calculate the path of the aircraft using BADA performance data.
        """

        alt_start = self.origin_ap_data["alt"] * FT_TO_M
        alt_stop = self.destination_ap_data["alt"] * FT_TO_M
        if self.typecode in AC_CRUISE_LEVELS.index:
            alt_cruise = AC_CRUISE_LEVELS.loc[self.typecode, "cr_fl"] * 100 * FT_TO_M
        else:
            input(f"{self.typecode} not in cruise FL Database, falling back to max observed alt for BADA cruise alt. Press enter to acnolage:")
            alt_cruise = self.df["baroaltitude"].max()

        lat_takeoff = self.origin_ap_data["lat"]
        lon_takeoff = self.origin_ap_data["lon"]
        lat_landing = self.destination_ap_data["lat"]
        lon_landing = self.destination_ap_data["lon"]

        line = geod.InverseLine(lat_takeoff, lon_takeoff, lat_landing, lon_landing)
        line_inv = geod.InverseLine(lat_landing, lon_landing, lat_takeoff, lon_takeoff)

        total_distance = line.s13

        while True: # Need to iterate in the case that it is not possible to climb to cruise alt within distance
            
            #   First calculate the climb path
            climb_rows = []
            climb_distance_travelled = 0
            current_alt = alt_start

            while current_alt < alt_cruise:
                bada_perf = self.find_ac_perf(current_alt)
                velocity = bada_perf["Vcl"] * KTS_TO_MPS
                vertrate = bada_perf["ROCDnom_cl"] * FT_TO_M / 60
                line_position = line.Position(climb_distance_travelled)
                current_lat = line_position["lat2"]
                current_lon = line_position["lon2"]
                climb_rows.append(
                    {
                        "alt": current_alt,
                        "lat": current_lat,
                        "lon": current_lon,
                        "vertrate": vertrate,
                        "tas": velocity,
                        "dist": climb_distance_travelled,
                    }
                )
                current_alt += BADA_PLOT_TIME_STEP * vertrate
                climb_distance_travelled += BADA_PLOT_TIME_STEP * velocity

            # Decent path
            decent_rows = []
            decent_distance_travelled = 0
            current_alt = alt_stop

            decent_distances = []
            while current_alt < alt_cruise:
                bada_perf = self.find_ac_perf(current_alt)
                velocity = bada_perf["Vdes"] * KTS_TO_MPS
                vertrate = bada_perf["ROCDnom_des"] * FT_TO_M / 60
                line_position = line_inv.Position(decent_distance_travelled)
                current_lat = line_position["lat2"]
                current_lon = line_position["lon2"]
                decent_distances.append(decent_distance_travelled)
                decent_rows.append(
                    {
                        "alt": current_alt,
                        "lat": current_lat,
                        "lon": current_lon,
                        "vertrate": vertrate,
                        "velocity": velocity,
                    }
                )
                current_alt += BADA_PLOT_TIME_STEP * vertrate
                decent_distance_travelled += BADA_PLOT_TIME_STEP * velocity


            decent_rows = decent_rows[::-1]
            decent_distances = decent_distances[::-1]

            if climb_distance_travelled + decent_distance_travelled > total_distance:
                print("Not enough distance climb to and decend from cruise altitude. Reducing target.")
                alt_cruise -= 500

            else:
                print("test123")
                break

        # Cruise path
        cruise_rows = []
        cr_vel = self.find_ac_perf(alt_cruise)["Vcr"] * KTS_TO_MPS
        cruise_distance_needed = total_distance - decent_distance_travelled - climb_distance_travelled
        cruise_distance_travelled = BADA_PLOT_TIME_STEP * cr_vel  # First point is not same position as last climb point

        while cruise_distance_travelled < cruise_distance_needed:
            line_position = line.Position(climb_distance_travelled + cruise_distance_travelled)
            current_lat = line_position["lat2"]
            current_lon = line_position["lon2"]
            cruise_rows.append(
                {
                    "alt": alt_cruise,
                    "lat": current_lat,
                    "lon": current_lon,
                    "vertrate": 0,
                    "tas": cr_vel,
                    "dist": climb_distance_travelled + cruise_distance_travelled,
                }
            )
            cruise_distance_travelled += BADA_PLOT_TIME_STEP * cr_vel


        self.bada_df = pd.DataFrame(climb_rows + cruise_rows)
        self.bada_df["time"] = np.arange(0, len(self.bada_df) * BADA_PLOT_TIME_STEP, BADA_PLOT_TIME_STEP)

        des_df = pd.DataFrame(decent_rows)
        decent_distances = np.array(decent_distances)
        decent_distances = climb_distance_travelled + cruise_distance_travelled + max(decent_distances) - decent_distances
        des_df["distance"] = decent_distances
        dist_cr_to_des_gap = total_distance - cruise_distance_travelled - decent_distance_travelled - climb_distance_travelled
        time_des_start = dist_cr_to_des_gap / cr_vel + self.bada_df["time"].iloc[-1]
        des_df["time"] = np.linspace(time_des_start, time_des_start + len(des_df) * BADA_PLOT_TIME_STEP, num=len(des_df), endpoint=False)

        self.bada_df = pd.concat([self.bada_df, des_df], ignore_index=True)


    def separate_legs(self):
        """
        Separate the flight path into takeoff, climb, cruise, decent, and landing phases.
        """

        MAX_CRUISE_VERT_RATE = 1 # m/s

        # Separate into takeoff, cruise, and landing by altitude
        max_alt = max(self.df["baroaltitude"])
        if max_alt > LTO_CEILING + self.origin_ap_data["alt"]*FT_TO_M:
            non_LTO_start = self.df[self.df["baroaltitude"] > LTO_CEILING + self.origin_ap_data["alt"]*FT_TO_M].index[0]
            self.df.loc[:non_LTO_start-1, "phase"] = "takeoff"
        else:
            non_LTO_start = 0

        if max_alt > LTO_CEILING + self.destination_ap_data["alt"]*FT_TO_M:
            non_LTO_end = self.df[self.df["baroaltitude"] > LTO_CEILING + self.destination_ap_data["alt"]*FT_TO_M].index[-1]
            self.df.loc[non_LTO_end+1:, "phase"] = "landing"
        else:
            non_LTO_end = len(self.df) - 1

        # Seprate cruise into climb, cruise, and decent by vertical rate
        vert_rate_ave = self.df["vertrate"].rolling(window=50, center=True, min_periods=1).median()
        self.df.loc[(vert_rate_ave > MAX_CRUISE_VERT_RATE) & (self.df.index >= non_LTO_start) & (self.df.index <= non_LTO_end), "phase"] = "climb"
        self.df.loc[(vert_rate_ave < -MAX_CRUISE_VERT_RATE) & (self.df.index >= non_LTO_start) & (self.df.index <= non_LTO_end), "phase"] = "decent"
        self.df.loc[(vert_rate_ave >= -MAX_CRUISE_VERT_RATE) & (vert_rate_ave <= MAX_CRUISE_VERT_RATE) & (self.df.index >= non_LTO_start) & (self.df.index <= non_LTO_end), "phase"] = "cruise"

        phase_changes = {phase: 0 for phase in self.df["phase"].unique()}
        change_indexes = self.df.index[self.df["phase"] != self.df["phase"].shift()].tolist()
        for idx in change_indexes:
            phase_changes[self.df["phase"][idx]] += 1
            # print(f"{self.df['phase'][idx]} at {(self.df['time'][idx] - self.df['time'][0]) / 3600:.3f} hours")
        print(f"Separation complete. Number of phases: {phase_changes}")

    def interpolate_alt_gaps(self, alt_source="baroaltitude"):
        # TODO does not climb to cruise altitude, only to next altitude this may be a problem for flight missing lots of data
        # TODO this does not ensure that correct distance is covered, only that the correct altitude is reached
        assert alt_source in ["baroaltitude", "geoaltitude"], "alt_source must be either 'baroaltitude' or 'geoaltitude'."
        idx_alt_gaps = self.df[alt_source].diff().abs() > ALT_GAP_MAX
        alt_gap_indexes = idx_alt_gaps[idx_alt_gaps].index.tolist()
        AC_MAX_ALT = self.ac_data["hMO"] * FT_TO_M

        if not alt_gap_indexes:
            print("No altitude gaps found.")
            return

        new_rows = []

        for gap_num, gap_idx in enumerate(alt_gap_indexes):
            start_row = self.df.iloc[gap_idx - 1]
            end_row = self.df.iloc[gap_idx]

            line = geod.InverseLine(start_row["lat"], start_row["lon"], end_row["lat"], end_row["lon"])
            gap_total_distance = line.s13

            print(f"Interpolating gap {gap_num + 1}/{len(alt_gap_indexes)}")
            time_start = start_row["time"]
            time_end = end_row["time"]
            time_gap = time_end - time_start
            if time_gap < TIME_GAP_ALT_GAP:  # Don't interpolate if gap time is too short
                print("Skipping interpolation due to short time gap.")
                continue
            alt_start = start_row[alt_source]
            alt_end = end_row[alt_source]
            alt_gap = alt_end - alt_start
            last_vert_rate = start_row["vertrate"]
            num_interp_points = max(int(time_gap / TIME_GAP_ALT_GAP), 5)
            interp_time = np.linspace(time_start, time_end, num_interp_points)[
                1:
            ]  # Final point removed later, keep for now to allow full time to be covered
            interp_time_step = interp_time[1] - interp_time[0]

            distance_travelled = 0
            current_alt = alt_start

            print(
                f"Altitude gap of {alt_end - alt_start:.0f} m, starting {(time_start - self.df['time'].iloc[0]) / 3600:.2f}h, duration {time_end - time_start:.0f}s."
            )

            if alt_start < alt_end:  # Need to climb to next altitude
                # First check if the climb rate is sufficient to reach the next altitude using BADA
                if alt_end > AC_MAX_ALT:
                    input("WARN! Altitude exceeds maximum altitude of BADA table. Please acknowledge.")
                    max_bada_alt = AC_MAX_ALT
                else:
                    max_bada_alt = alt_start
                    bada_distance_required_climb = 0
                    for time in interp_time:
                        ac_perf = self.find_ac_perf(max_bada_alt)
                        max_bada_alt += ac_perf["ROCDnom_cl"] * interp_time_step * FT_TO_M / 60
                        if max_bada_alt <= alt_end: # Still need to climb, thus we need to cover the distance
                            vel_at_fl = ac_perf["Vcl"]  # NOTE using TAS not GS, change if possible
                            bada_distance_required_climb += vel_at_fl * interp_time_step * KTS_TO_MPS
                        if max_bada_alt > AC_MAX_ALT: # If altitude exceeds BADA, max altitude reached
                            max_bada_alt = AC_MAX_ALT
                            break

                max_bada_climb = max_bada_alt - alt_start

                max_prev_rate_climb = time_gap * last_vert_rate

                print(
                    f"Max BADA climb: {max_bada_climb:.0f} m, climb if current rate continued until next datapoint: {max_prev_rate_climb:.0f} m, target: {alt_gap:.0f} m."
                )

                if max_bada_climb > alt_gap:  # BADA climb rate is sufficient, interpolate using BADA
                    print("BADA sufficient to fill time gap.")
                    print(
                        f"BADA needs {bada_distance_required_climb / 1000:.0f} km to climb, availible distance: {gap_total_distance / 1000:.0f} km."
                    )
                    if bada_distance_required_climb > gap_total_distance:  # TODO could add some sort of climb pattern
                        input("WARN! Distance required to climb using BADA is greater than availible distance. Please acknowledge.")
                    for time in interp_time:
                        climb_needed = alt_end - current_alt
                        ac_perf = self.find_ac_perf(current_alt)
                        climb_possible = ac_perf["ROCDnom_cl"] * interp_time_step * FT_TO_M / 60
                        climb = min(climb_needed, climb_possible)
                        vert_rate = climb / interp_time_step
                        current_alt += climb
                        ac_vel = ac_perf["Vcl"] * KTS_TO_MPS
                        distance_travelled += ac_vel * interp_time_step
                        line_position = line.Position(distance_travelled)
                        new_rows.append({"time": time, "alt": current_alt, "lat": line_position["lat2"], "lon": line_position["lon2"], "velocity": ac_vel, "distance": distance_travelled+start_row["distance"], "vertrate": vert_rate, "phase": "climb"})
                        if current_alt >= alt_end and time != interp_time[-1]:  # Break if target altitude reached prematurely
                            break

                elif max_prev_rate_climb > alt_gap:  # BADA climb rate is insufficient, use previous vert rate
                    print(f"Using previous vert rate ({last_vert_rate:.0f} m/s) in place of BADA.")
                    for time in interp_time:
                        climb_needed = alt_end - current_alt
                        climb_possible = last_vert_rate * interp_time_step
                        climb = min(climb_needed, climb_possible)
                        current_alt += climb
                        # TODO add pos and vel
                        new_rows.append({"time": time, "alt": current_alt})
                        if current_alt >= alt_end and time != interp_time[-1]:  # Break if target altitude reached prematurely
                            break

                else:  # Both BADA and previous vert rate are insufficient
                    if input("Continue, filling with unrealistic climbrate? (y/n): ").lower() == "y":
                        required_rate = alt_gap / time_gap
                        print(f"Filling with climb at {required_rate:.0f} m/s to fill gap.")
                        for time in interp_time:
                            climb = required_rate * interp_time_step
                            current_alt += climb
                            # TODO add pos and vel
                            new_rows.append({"time": time, "alt": current_alt})
                            if current_alt >= alt_end and time != interp_time[-1]:  # Break if target altitude reached prematurely
                                break
                    else:
                        raise ValueError("Interpolation failed.")
                    
            elif alt_start > alt_end:  # Need to decent to next altitude
                print("Decent not implemented yet.")
                continue  # TODO implement decent interpolation

            if time == interp_time[-1]:  # If the final point was reached, remove it to avoid duplicate with real data
                print("Full time series used, removing final point to avoid duplicate.")
                new_rows.pop()
        if new_rows:
            self.interp_df = pd.DataFrame(new_rows)

    def interpolate_time_gaps(self):
        pass

    def plot_phases(self):

        if self.interp_df is not None:
            for phase in self.interp_df["phase"].unique():
                interp_phase_df = self.interp_df[self.interp_df["phase"] == phase]
                plt.scatter((interp_phase_df["time"] - self.df["time"][0]) / 60**2, interp_phase_df["alt"], label=f"Interpolated {phase}")

        for phase in self.df["phase"].unique():
            phase_df = self.df[self.df["phase"] == phase]
            plt.scatter((phase_df["time"] - self.df["time"][0]) / 60**2, phase_df["baroaltitude"], label=phase)

        plt.legend()
        plt.xlabel("Time (hours)")
        plt.ylabel("Altitude (m)")
        plt.title(f"{self.origin_name} to {self.destination_name} - {self.typecode} - No: {self.flight_num}")
        plt.show()

    def plot_altitude(self, x_axis="time", BADA=False):
        assert x_axis in ["time", "distance"], "x_axis must be either 'time' or 'distance'."
        if x_axis == "time":
            data_start = self.df["time"].iloc[0]
            plt.scatter((self.df["time"] - data_start) / 3600, self.df["geoaltitude"], s=5, alpha=0.5, color="red", label="GNSS Data")
            plt.scatter((self.df["time"] - data_start) / 3600, self.df["baroaltitude"], s=5, alpha=0.5, color="green", label="Barometric Data")
            if self.interp_df is not None:
                plt.scatter((self.interp_df["time"] - data_start) / 3600, self.interp_df["alt"], s=5, alpha=0.5, color="blue", label="Interpolated")
            plt.xlabel("Time (hours)")
        elif x_axis == "distance":
            plt.scatter(self.df["dist"] / 1000, self.df["geoaltitude"], s=5, alpha=0.5, color="red", label="GNSS Data")
            plt.scatter(self.df["dist"] / 1000, self.df["baroaltitude"], s=5, alpha=0.5, color="green", label="Barometric Data")
            if self.interp_df is not None:
                plt.scatter(self.interp_df["dist"] / 1000, self.interp_df["alt"], s=5, alpha=0.5, color="blue", label="Interpolated")
            plt.xlabel("Distance (km)")

        if BADA:
            if self.bada_df is None:
                self.calc_bada_path()
            if x_axis == "time":
                plt.plot(self.bada_df["time"] / 3600, self.bada_df["alt"], label="BADA Path", linestyle="--")
                plt.hlines(self.ac_data["hMO"]*FT_TO_M, 0, max(self.bada_df["time"] / 3600), linestyle="--", color="black", label="BADA Max Altitude")
            elif x_axis == "distance":
                plt.plot(self.bada_df["dist"] / 1000, self.bada_df["alt"], label="BADA Path", linestyle="--")
                plt.hlines(self.ac_data["hMO"]*FT_TO_M, 0, max(self.bada_df["dist"] / 1000), linestyle="--", color="black", label="BADA Max Altitude")

        takeoff_time = pd.to_datetime(self.df["time"].iloc[0], unit="s").round("s")

        plt.legend()
        plt.ylabel("Altitude (m)")
        plt.title(
            f"{self.origin_name} to {self.destination_name} - {self.typecode} - No: {self.flight_num} - Takeoff: {takeoff_time}"
        )
        plt.show()

    def plot_vert_rate(self):
        time_series = (self.df["time"] - self.df["time"][0]) / 60**2
        # vert_rate_ave = self.df["vertrate"].rolling(window=50, center=True, min_periods=1).median()
        # plt.plot(time_series, vert_rate_ave
        #          , alpha=0.5, label="Averaged Data")
        plt.scatter(time_series, self.df["geoaltitude"].diff(), s=5, alpha=0.5, label="GNSS Dif")
        plt.scatter(time_series, self.df["baroaltitude"].diff(), s=5, alpha=0.5, label="Baro Dif")
        plt.scatter(time_series, self.df["vertrate"], s=5, alpha=0.5, label="Vertrate")
        plt.xlabel("Time (hours)")
        plt.ylabel("Vertical Rate (m/s)")
        plt.ylim(-25, 25)
        plt.text(0.5, 0.95, 'Vertical rate axis limited to [-25, 25] m/s', 
             horizontalalignment='center', verticalalignment='center', 
             transform=plt.gca().transAxes, fontsize=10, color='red')
        plt.legend()
        plt.show()

    def vert_rate_vs_altitude(self):
        plt.scatter(self.df['baroaltitude'], self.df["vertrate"], s=8, c="orange", marker='x', label="ADS-B Altitude Rate vs Baro Altitude", alpha=1)
        if self.interp_df is not None:
            plt.scatter(self.interp_df['alt'], self.interp_df["vertrate"], s=8, c="black", marker='x', label="Interpolated Data", alpha=1)

        plt.plot(self.ac_data["table"].index * 100 * FT_TO_M, self.ac_data["table"]["ROCDhi_cl"] * FT_TO_M/ 60, label="BADA High Load Climb Rate", c = "red")
        plt.plot(self.ac_data["table"].index * 100 * FT_TO_M, self.ac_data["table"]["ROCDnom_cl"] * FT_TO_M/ 60, label="BADA Nominal Load Climb Rate", c = "green")
        plt.plot(self.ac_data["table"].index * 100 * FT_TO_M, self.ac_data["table"]["ROCDlo_cl"] * FT_TO_M/ 60, label="BADA Low Load Climb Rate", c = "purple")
        plt.plot(self.ac_data["table"].index * 100 * FT_TO_M, self.ac_data["table"]["ROCDnom_des"] * FT_TO_M/ -60, label="BADA Nominal Load Descent Rate", c = "blue")
        plt.xlabel("Altitude (m)")
        plt.ylabel("Altitude Rate (m/s)")
        plt.title(f"{self.origin_name} to {self.destination_name} Altitude Rate vs Altitude")
        plt.ylim(-25, 25)
        plt.text(0.5, 0.95, 'Vertical rate axis limited to [-25, 25] m/s', 
                        horizontalalignment='center', verticalalignment='center', 
                        transform=plt.gca().transAxes, fontsize=10, color='red')
        plt.legend()
        plt.show()

    def plot_velocity_vs_altitude(self, SepratePhases=False):

        plt.plot(self.ac_data["table"].index * 100 * FT_TO_M, self.ac_data["table"]["Vcl"], label="BADA Climb Velocity", c = "red")
        plt.plot(self.ac_data["table"].index * 100 * FT_TO_M, self.ac_data["table"]["Vcr"], label="BADA Cruise Velocity", c = "green")
        plt.plot(self.ac_data["table"].index * 100 * FT_TO_M, self.ac_data["table"]["Vdes"], label="BADA Descent Velocity", c = "blue")
        if SepratePhases:
            for phase in self.df["phase"].unique():
                phase_df = self.df[self.df["phase"] == phase]
                plt.scatter(phase_df['baroaltitude'], phase_df["gs"]/KTS_TO_MPS, s=8, label=phase + "gs", alpha=1)
                plt.scatter(phase_df['geoaltitude'], phase_df["tas"]/KTS_TO_MPS, s=8, label=phase + "tas", alpha=1)
        else:
            plt.scatter(self.df['baroaltitude'], self.df["gs"]/KTS_TO_MPS, s=8, label="GS", alpha=1)
            plt.scatter(self.df['geoaltitude'], self.df["tas"]/KTS_TO_MPS, s=8, label="TAS", alpha=1)

        plt.xlabel("Altitude (m)")
        plt.ylabel("Velocity (knots)")
        plt.title(f"{self.typecode} - {self.origin_name} to {self.destination_name} Velocity vs Altitude")
        plt.legend()
        plt.show()

    def plot_flight_path(self, color_by="velocity", BADA=True):
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        # ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        # ax.add_feature(cfeature.LAKES, alpha=0.5)
        # ax.add_feature(cfeature.RIVERS)

        min_lon = min(self.df["lon"])
        max_lon = max(self.df["lon"])
        mid_lon = (max_lon + min_lon) / 2
        min_lat = min(self.df["lat"])
        max_lat = max(self.df["lat"])
        mid_lat = (max_lat + min_lat) / 2

        size_lon = max((max_lon - min_lon)*1.2, (max_lat - min_lat)*0.5)
        size_lat = max((max_lon - min_lon)*0.5, (max_lat - min_lat)*1.2)

        min_lon = mid_lon - size_lon/2
        max_lon = mid_lon + size_lon/2
        min_lat = mid_lat - size_lat/2
        max_lat = mid_lat + size_lat/2

        ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

        if BADA:
            if self.bada_df is None:
                self.calc_bada_path()
            plt.plot(self.bada_df["lon"], self.bada_df["lat"], label="BADA Path", linestyle="--")

        if color_by == "velocity":
            cbar_label = "Velocity (m/s)"
            cbar_source = "velocity"
            max_velocity = max(self.df["velocity"])
            min_velocity = min(self.df["velocity"])
            norm = Normalize(vmin=min_velocity, vmax=max_velocity)
        elif color_by.endswith("altitude"):
            cbar_source_interp = "alt"
            if color_by == "geoaltitude":
                cbar_label = "Geo Altitude (m)"
                cbar_source = "geoaltitude"
            elif color_by == "baroaltitude":
                cbar_label = "Baro Altitude (m)"
                cbar_source = "baroaltitude"
            max_altitude = max(self.df[cbar_source])
            min_altitude = min(self.df[cbar_source])
            norm = Normalize(vmin=min_altitude, vmax=max_altitude)
        elif color_by == "tailwind":
            self.df["tailwind"] = self.df["gs"] - self.df["tas"]
            cbar_label = "Tailwind (m/s)"
            cbar_source = "tailwind"
            max_tailwind = max(self.df["tailwind"])
            min_tailwind = min(self.df["tailwind"])
            norm = Normalize(vmin=min_tailwind, vmax=max_tailwind)

            lat_idx= np.digitize(self.df["lat"], MET_DATA["lat"].values)
            lon_idx = np.digitize(self.df["lon"], MET_DATA["lon"].values)
            time_idx = np.digitize(self.df["time"], MET_DATA["time"].values.astype(np.int64)/1e9)
            cruiselev = max(np.digitize(self.df["geoaltitude"], MET_DATA["h_edge"].values) - 1)
            met = MET_DATA.sel(lev=cruiselev, method='nearest')
            # removed some arrows here
            met = met.isel(lat=slice(min(lat_idx), max(lat_idx)+1, 5), lon=slice(min(lon_idx), max(lon_idx)+1, 5), time=time_idx[len(time_idx)//2])
            lat = met["lat"].values
            lon = met["lon"].values
            WS = met["WS"].values
            WDIR = met["WDIR"].values
            u = -WS * np.sin(np.radians(WDIR))
            v = -WS * np.cos(np.radians(WDIR))
            quiver = ax.quiver(lon, lat, u, v, transform=ccrs.PlateCarree(), scale=500)

            # Add a quiver key to explain arrow length (optional)
            plt.quiverkey(quiver, 0.9, -0.1, 10, "10 m/s", labelpos='E')


        points = plt.scatter(
            self.df["lon"],
            self.df["lat"],
            c=self.df[cbar_source],
            cmap="viridis",
            norm=norm,
            s=5,
            alpha=0.5,
            marker="o",
            label="ADS-B Data",
        )

        if self.interp_df is not None:
            plt.scatter(
                self.interp_df["lon"],
                self.interp_df["lat"],
                c=self.interp_df[cbar_source_interp],
                cmap="viridis",
                norm=norm,
                s=5,
                alpha=0.5,
                label="Interpolated Data",
                marker="^",
            )

        cbar = plt.colorbar(points, orientation='horizontal', pad=0.05, aspect=50, label=cbar_label)
        cbar.ax.set_position([0.215, -0.05, 0.6, 0.3])
        plt.title(f"{self.origin_name} to {self.destination_name} - {self.typecode} - No: {self.flight_num}")
        plt.legend()
        plt.show()

    def return_decent_alt(self, x_axis="time"):
        """
        Return the altitude data for the decent phase of the flight.
        Used to compare the decent path of different flights.
        """
        assert x_axis in ["time", "distance"], "x_axis must be either 'time' or 'distance'."

        one_hour_before_final = self.df["time"].iloc[-1] - 3600
        decent_start_idx = self.df[(self.df["phase"] == "decent") & (self.df["time"] >= one_hour_before_final)].index[0]
        decent_df = self.df.loc[decent_start_idx:]

        if x_axis == "time":
            data_start = decent_df["time"].iloc[0]
            x: pd.Series = (decent_df["time"] - data_start) / 3600
            xlabel:str = "Time (hours)"

        elif x_axis == "distance":
            x: pd.Series = decent_df["dist"] / 1000
            xlabel:str = "Distance (km)"


        y = decent_df["baroaltitude"]
        ylabel:str = "Barometric Altitude (m)"
        data_label:str = f"Landing at {self.destination_name}"

        return x, y, xlabel, ylabel, data_label


def load_flight_paths_obj() -> list[FlightPath]:
    flight_paths = []
    for filename in os.listdir("./output"):
        if filename.endswith(".csv") or filename.endswith(".pkl"):
            flight = FlightPath(filename)
            flight_paths.append(flight)
    return flight_paths


# flight_paths = load_flight_paths_obj()
# for flight in flight_paths:
#     print(flight)
#     flight.clean_alt_data()
#     flight.interpolate_alt_gaps()
#     flight.plot_altitude(BADA=True, x_axis="distance")
#     flight.plot_flight_path(color_by="geoaltitude")
#     flight.plot_vert_rate()
#     flight.separate_legs()
#     flight.plot_phases()
#     flight.vert_rate_vs_altitude()

descents = []
for filename in os.listdir(FLIGHTS_DIR):
    if filename.endswith(".csv") or filename.endswith(".pkl"):
        flight = FlightPath(filename)
        # print(flight)
        # flight.clean_alt_data()
        flight.calc_bada_path()

        flight.separate_legs()
        # flight.interpolate_alt_gaps(alt_source="baroaltitude")
        # flight.GS_to_TAS()
        flight.plot_altitude(BADA=True, x_axis="distance")
        # flight.plot_flight_path(color_by="geoaltitude")
        flight.plot_flight_path(color_by="tailwind")
        flight.plot_velocity_vs_altitude()
        # flight.plot_vert_rate()
        # flight.plot_phases()
        # flight.vert_rate_vs_altitude()
        # x,y,xlabel,ylabel,datalabel = flight.return_decent_alt(x_axis="time")
        # descents.append((x,y,xlabel,ylabel,datalabel))

# for descent in descents:
#     plt.scatter(descent[0], descent[1], label=descent[4])
# plt.xlabel(descents[0][2])
# plt.ylabel(descents[0][3])
# plt.legend()
# plt.show()