import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from geographiclib.geodesic import Geodesic
import cartopy.crs as ccrs
import cartopy.feature as cfeature

geod = Geodesic.WGS84

FT_TO_M = 0.3048
KTS_TO_MPS = 0.514444
TIME_GAP_CRUISE = 60
ALT_GAP_MAX = 250  # Maximum altitude gap in meters before interpolation
TIME_GAP_ALT_GAP = 10
CRUISE_STEP = 50 * 1852  # 50 nautical miles in meters
LTO_CEILING = 3000 * FT_TO_M
AIRPORT_NAMES = pd.read_csv("airports.csv").set_index("icao")
AC_CRUISE_LEVELS = pd.read_csv("cruise_fl.csv").set_index("typecode")
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
    def __init__(self, filename, flights_dir="./output"):
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
        self.ac_data = read_ptf(f"./BADA/{self.typecode}__.ptf")
        self.bada_df = None
        self.interp_df = None
        self.time_gaps = self.df["time"].diff()
        if max(max(self.df["baroaltitude"]), max(self.df["geoaltitude"])) > self.ac_data["hMO"]:
            input(f"Warning: Flight {self.flight_num} exceeds maximum altitude of {self.ac_data['hMO']}, please acknowledge.")
        self._find_distance_travelled()
        print(f"Flight imported from {filename}.")

    def __repr__(self):
        return f"\nFlightPath Object\nFrom: {self.origin_name}, To: {self.destination_name}, Typecode: {self.typecode}\nTakeoff: {self.df['time'].iloc[0]}, Landing: {self.df['time'].iloc[-1]}, Duration: {self.df['time'].iloc[-1] - self.df['time'].iloc[0]} seconds\nicao24: {self.icao24}, flight number: {self.flight_num}, containing {len(self.df)} rows."

    def combine_df(self):
        interplated_df = self.interp_df.copy()
        real_df = self.df.copy()
        interplated_df["interpolated"] = True
        real_df["interpolated"] = False
        combined_df = pd.concat([interplated_df, real_df], ignore_index=True)
        combined_df.sort_values(by="time", inplace=True)
        combined_df.reset_index(drop=True, inplace=True)
        return combined_df

    def _find_distance_travelled(self):
        distances = [0]
        for i in range(1, len(self.df)):
            prev_point = self.df.iloc[i - 1]
            curr_point = self.df.iloc[i]
            distance = geod.Inverse(prev_point["lat"], prev_point["lon"], curr_point["lat"], curr_point["lon"])["s12"]
            distances.append(distances[-1] + distance)
        self.df["distance"] = distances

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

    def clean_alt_data(self):
        # TODO better way to clean altitude data
        initial_length = len(self.df)
        mask = abs(self.df["baroaltitude"].diff().fillna(0) / self.time_gaps - self.df["vertrate"]) <= 50
        self.df = self.df.loc[mask]
        mid_length = len(self.df)
        mask = abs(self.df["geoaltitude"].diff().fillna(0) / self.time_gaps - self.df["vertrate"]) <= 50
        self.df = self.df.loc[mask]
        final_length = len(self.df)
        if initial_length - final_length > 0:
            # print(f"{self.origin_name} to {self.destination_name} - {self.typecode} - {self.flight_num}")
            print(f"Removed {initial_length - mid_length} rows due to baroaltitude anomalies.")
            print(f"Removed {mid_length - final_length} rows due to geoaltitude anomalies.\n")
        self.df = self.df.reset_index(drop=True)

    def calc_bada_path(self):
        """
        Calculate the path of the aircraft using BADA performance data.
        """
        # alt_start = self.df["geoaltitude"].iloc[0]
        # alt_stop = self.df["geoaltitude"].iloc[-1]
        # alt_cruise = max(self.df["geoaltitude"])

        alt_start = self.origin_ap_data["alt"]
        alt_stop = self.destination_ap_data["alt"]
        alt_cruise = AC_CRUISE_LEVELS.loc[self.typecode, "cr_fl"] * 100 * FT_TO_M

        lat_takeoff = self.origin_ap_data["lat"]
        lon_takeoff = self.origin_ap_data["lon"]
        lat_landing = self.destination_ap_data["lat"]
        lon_landing = self.destination_ap_data["lon"]

        line = geod.InverseLine(lat_takeoff, lon_takeoff, lat_landing, lon_landing)
        line_inv = geod.InverseLine(lat_landing, lon_landing, lat_takeoff, lon_takeoff)

        total_distance = line.s13

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
                    "velocity": velocity,
                    "distance": climb_distance_travelled,
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
            raise ValueError(
                "Not enough distance climb to and decend from cruise altitude.\nImplement a better way to calculate the path."
            )

        else:
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
                        "velocity": cr_vel,
                        "distance": climb_distance_travelled + cruise_distance_travelled,
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
        des_df["time"] = np.arange(time_des_start, time_des_start + len(des_df) * BADA_PLOT_TIME_STEP, BADA_PLOT_TIME_STEP)

        self.bada_df = pd.concat([self.bada_df, des_df], ignore_index=True)


    def separate_legs(self):
        """
        Separate the flight path into takeoff, climb, cruise, decent, and landing phases.
        """
        # Introduce a phase column, default to cruise
        self.df["phase"] = "cruise"

        # Separate into takeoff, cruise, and landing by altitude
        non_lto_idxs = self.df[self.df["geoaltitude"] > LTO_CEILING]
        non_LTO_start = non_lto_idxs.index[0]
        non_LTO_end = non_lto_idxs.index[-1]
        self.df.loc[:non_LTO_start, "phase"] = "takeoff"
        self.df.loc[non_LTO_end:, "phase"] = "landing"

        # Seprate cruise into climb, cruise, and decent by vertical rate
        vert_rate_ave = self.df["vertrate"].rolling(window=50, center=True, min_periods=1).median()
        self.df.loc[(vert_rate_ave > 0.5) & (self.df["phase"] == "cruise"), "phase"] = "climb"
        self.df.loc[(vert_rate_ave < -0.5) & (self.df["phase"] == "cruise"), "phase"] = "decent"

        phase_changes = {phase: 0 for phase in self.df["phase"].unique()}
        change_indexes = self.df.index[self.df["phase"] != self.df["phase"].shift()].tolist()
        for idx in change_indexes:
            phase_changes[self.df["phase"][idx]] += 1
            print(f"{self.df['phase'][idx]} at {(self.df['time'][idx] - self.df['time'][0]) / 3600:.3f} hours")
        print(f"Separation complete. Number of phases: {phase_changes}")

    def interpolate_alt_gaps(self):
        # TODO does not climb to cruise altitude, only to next altitude this may be a problem for flight missing lots of data
        # TODO this does not ensure that correct distance is covered, only that the correct altitude is reached
        idx_alt_gaps = self.df["baroaltitude"].diff().abs() > ALT_GAP_MAX
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
            alt_start = start_row["baroaltitude"]
            alt_end = end_row["baroaltitude"]
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
                    input("WARN. Altitude exceeds maximum altitude of BADA table. Please acknowledge.")
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
                        f"BADA needs {bada_distance_required_climb / 1000:.0f} km to climb, actual distance covered: {gap_total_distance / 1000:.0f} km."
                    )
                    if bada_distance_required_climb > gap_total_distance:  # TODO could add some sort of climb pattern
                        input("WARN. Distance required to climb using BADA is greater than total distance. Please acknowledge.")
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
                        new_rows.append({"time": time, "alt": current_alt, "lat": line_position["lat2"], "lon": line_position["lon2"], "velocity": ac_vel, "distance": distance_travelled+start_row["distance"], "vertrate": vert_rate})
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

            if time == interp_time[-1]:  # If the final point was reached, remove it to avoid duplicate with real data
                print("Full time series used, removing final point to avoid duplicate.")
                new_rows.pop()

        self.interp_df = pd.DataFrame(new_rows)

    def interpolate_time_gaps(self):
        pass

    def plot_phases(self):
        phases = self.df["phase"].unique()
        for phase in phases:
            phase_df = self.df[self.df["phase"] == phase]
            plt.scatter((phase_df["time"] - self.df["time"][0]) / 60**2, phase_df["geoaltitude"], label=phase)
        plt.legend()
        plt.xlabel("Time (hours)")
        plt.ylabel("Altitude (m)")
        plt.show()

    def plot_altitude(self, x_axis="time", BADA=False):
        assert x_axis in ["time", "distance"], "x_axis must be either 'time' or 'distance'."
        if x_axis == "time":
            data_start = self.df["time"].iloc[0]
            plt.scatter((self.df["time"] - data_start) / 3600, self.df["baroaltitude"], s=5, alpha=0.5, color="red", label="Data")
            if self.interp_df is not None:
                plt.scatter((self.interp_df["time"] - data_start) / 3600, self.interp_df["alt"], s=5, alpha=0.5, color="blue", label="Interpolated")
            plt.xlabel("Time (hours)")
        elif x_axis == "distance":
            plt.scatter(self.df["distance"] / 1000, self.df["baroaltitude"], s=5, alpha=0.5, color="red", label="Data")
            if self.interp_df is not None:
                plt.scatter(self.interp_df["distance"] / 1000, self.interp_df["alt"], s=5, alpha=0.5, color="blue", label="Interpolated")
            plt.xlabel("Distance (km)")

        if BADA:
            if self.bada_df is None:
                self.calc_bada_path()
            if x_axis == "time":
                plt.plot(self.bada_df["time"] / 3600, self.bada_df["alt"], label="BADA Path", linestyle="--")
            elif x_axis == "distance":
                plt.plot(self.bada_df["distance"] / 1000, self.bada_df["alt"], label="BADA Path", linestyle="--")

        takeoff_time = pd.to_datetime(self.df["time"].iloc[0], unit="s").round("s")

        plt.legend()
        plt.ylabel("Altitude (m)")
        plt.title(
            f"{self.origin_name} to {self.destination_name} - {self.typecode} - No: {self.flight_num} - Takeoff: {takeoff_time}"
        )
        plt.show()

    def plot_vert_rate(self):
        time_series = self.df["time"] - self.df["time"][0]  # / 60**2
        vert_rate_ave = self.df["vertrate"].rolling(window=50, center=True, min_periods=1).median()
        plt.scatter(time_series, vert_rate_ave, s=5, alpha=0.5, label="Averaged Data")
        plt.scatter(time_series, self.df["vertrate"], s=5, alpha=0.5, label="Raw Data")
        plt.xlabel("Time (hours)")
        plt.ylabel("Vertical Rate (m/s)")
        plt.legend()
        plt.show()

    def vert_rate_vs_altitude(self):
        time_series = (self.df['time'] - self.df["time"][0]) / 60**2

        plt.scatter(self.df['geoaltitude'], self.df["vertrate"], s=8, c=time_series, cmap='plasma', marker='x', label="ADS-B Altitude Rate vs GNSS Altitude", alpha=1)
        if self.interp_df is not None:
            plt.scatter(self.interp_df['alt'], self.interp_df["vertrate"], s=8, c="r", marker='x', label="Interpolated Data", alpha=1)

        plt.plot(self.ac_data["table"].index * 100 * FT_TO_M, self.ac_data["table"]["ROCDhi_cl"] * FT_TO_M/ 60, label="BADA High Load Climb Rate", c = "red")
        plt.plot(self.ac_data["table"].index * 100 * FT_TO_M, self.ac_data["table"]["ROCDnom_cl"] * FT_TO_M/ 60, label="BADA Nominal Load Climb Rate", c = "green")
        plt.plot(self.ac_data["table"].index * 100 * FT_TO_M, self.ac_data["table"]["ROCDlo_cl"] * FT_TO_M/ 60, label="BADA Low Climb Load Rate", c = "purple")
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
        min_lat = min(self.df["lat"])
        max_lat = max(self.df["lat"])

        lon_margin = (max_lon - min_lon) * 0.1
        lat_margin = (max_lat - min_lat) * 0.1
        min_lon -= lon_margin
        max_lon += lon_margin
        min_lat -= lat_margin
        max_lat += lat_margin

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

for filename in os.listdir("./output"):
    if filename.endswith(".csv") or filename.endswith(".pkl"):
        flight = FlightPath(filename)
        print(flight)
        flight.clean_alt_data()
        flight.interpolate_alt_gaps()
        flight.plot_altitude(BADA=True, x_axis="time")
        flight.vert_rate_vs_altitude()