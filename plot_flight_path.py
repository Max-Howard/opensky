import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os
from geopy.distance import geodesic
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import contextily as ctx
import geopandas as gpd
from geographiclib.geodesic import Geodesic


TIME_GAP = 60
CRUISE_STEP = 50 * 1852       # 50 nautical miles in meters
DELTA_ALT_CRUISE = 3000 * 0.3048    # 3000 ft in meters
DELTA_ALT_CRUISE = 16000 * 0.3048    # 16000 ft in meters
AIRPORT_NAMES = pd.read_csv("airports.csv")

class FlightPath:
      def __init__(self, flightname:str, df:pd.DataFrame):
            self.flightname = flightname
            df["time"] = pd.to_datetime(df["time"])
            self.icao24 = df["icao24"].iloc[0]
            self.origin_icao = df["origin"].iloc[0]
            self.destination_icao = df["destination"].iloc[0]
            df.drop(columns=["icao24"], inplace=True)
            df.drop(columns=["origin"], inplace=True)
            df.drop(columns=["destination"], inplace=True)
            self.origin_name = AIRPORT_NAMES[AIRPORT_NAMES["icao"] == self.origin_icao]["name"].iloc[0]
            self.destination_name = AIRPORT_NAMES[AIRPORT_NAMES["icao"] == self.destination_icao]["name"].iloc[0]
            self.df = df

      def __repr__(self):
            return f"FlightPath: {self.flightname}, with {len(self.df)} rows."
      
      def separate_legs(self):
            """
            Separate the flight path into takeoff, cruise, and landing phases.
            """
            # Find the start and end of the cruise phase
            cutoff_alt = DELTA_ALT_CRUISE + self.df["geoaltitude"].iloc[0]
            cruise_start_idx = self.df["geoaltitude"].gt(cutoff_alt).idxmax()
            cruise_end_idx = self.df["geoaltitude"].gt(cutoff_alt)[::-1].idxmax()
            self.takeoff = self.df.iloc[:cruise_start_idx]
            self.cruise = self.df.iloc[cruise_start_idx:cruise_end_idx]
            self.landing = self.df.iloc[cruise_end_idx:]

      def interpolate_great_circle(self):
            pass

def load_flight_paths_obj() -> list[FlightPath]:
      flight_paths = []
      for filename in os.listdir('./output'):
            if filename.endswith('.csv'):
                  file_path = os.path.join('./output', filename)
                  df = pd.read_csv(file_path)
                  flight_paths.append(df)
      return flight_paths

def load_flight_paths() -> dict[pd.DataFrame]:
      """
      Load all flight paths from the output folder.
      Add a gap column to the dataframes to indicate points with a time gap greater than TIME_GAP.
      Return a dictionary of flight paths.
      """
      flight_paths = {}
      for file in os.listdir('./output'):
            if file.endswith('.csv'):
                  file_path = os.path.join('./output', file)
                  df = pd.read_csv(file_path)
                  df["time"] = pd.to_datetime(df["time"])
                  print(f"Loaded {file} with {len(df)} rows.")
                  flight_paths[file.replace('.csv', '')] = df
      return flight_paths


def interpolate_great_circle(flight_paths: dict[pd.DataFrame]):
      for flight_name in flight_paths.keys():
            df:pd.DataFrame = flight_paths[flight_name]

            # Find the start and end of the cruise phase
            cutoff_alt = DELTA_ALT_CRUISE + df["geoaltitude"].iloc[0] # Cruise altitude starts ALT_CRUISE above the first altitude
            cruise_start_idx = df["geoaltitude"].gt(cutoff_alt).idxmax()
            cruise_end_idx = df["geoaltitude"].gt(cutoff_alt)[::-1].idxmax()

            print(f"Found cruise start at index {cruise_start_idx} and end at index {cruise_end_idx}.")

            # df = df.drop(df.index[cruise_start_idx:cruise_end_idx]).reset_index(drop=True)


            # Find time gaps greater than TIME_GAP
            time_gaps = df["time"].diff().dt.total_seconds() > TIME_GAP
            gap_indexes = time_gaps[time_gaps].index.tolist()

            df["interpolated"] = False

            added_rows = 0
            for gap_num,idx in enumerate(gap_indexes):
                  idx_corr = idx + added_rows # Corrected index to account for added rows
                  start_row = df.iloc[idx_corr - 1]
                  end_row = df.iloc[idx_corr]
                  start_time = pd.to_datetime(start_row["time"])
                  end_time = pd.to_datetime(end_row["time"])
                  gap_total_seconds = (end_time - start_time).total_seconds()

                  # Initialize the geodesic line
                  geod = Geodesic.WGS84
                  line = geod.InverseLine(start_row["lat"], start_row["lon"], end_row["lat"], end_row["lon"])
                  gap_total_distance = line.s13
                  num_points = int(np.ceil(gap_total_distance / CRUISE_STEP)) # Number of points to add to fill the gap

                  print(f"Adding {num_points} points to fill gap {gap_num + 1} of {len(gap_indexes)}, in {flight_name}, at index {idx}.")

                  time_intervals = pd.date_range(start_time, end_time, periods=num_points + 2)[1:-1]


                  step = gap_total_distance / (num_points + 1)
                  lat_intervals = []
                  lon_intervals = []
                  for i in range(1, num_points+1): # Skip the first and last points as they are already in the dataframe
                        point = line.Position(i * step)
                        lat_intervals.append(point['lat2'])
                        lon_intervals.append(point['lon2'])

                  velocity_intervals = np.linspace(start_row["velocity"], end_row["velocity"], num_points + 2)[1:-1]
                  gap_mean_velocity = gap_total_distance / gap_total_seconds
                  velocity_correction = gap_mean_velocity - np.mean(velocity_intervals)
                  velocity_intervals += velocity_correction # TODO this may introduce errors in the velocity data
                  print(f"Added: {velocity_correction}, to interval velocities, to ensure continuity with distance travelled.")
                  interpolated_rows = pd.DataFrame({
                              "time": time_intervals,
                              "lat": lat_intervals,
                              "lon": lon_intervals,
                              "velocity": velocity_intervals
                  })
                  interpolated_rows["interpolated"] = True
                  added_rows += len(interpolated_rows)
                  df = pd.concat([df.iloc[:idx_corr], interpolated_rows, df.iloc[idx_corr:]]).reset_index(drop=True)
            flight_paths[flight_name] = df
      return flight_paths


def analyse_data(flight_paths, time_gap_thresh=60, vel_mismatch_thresh=10, vel_sample_size=100):
      vel_mismatch_lat = []
      vel_mismatch_lon = []
      for flight_path_name, flight_path in flight_paths.items():
            times = pd.to_datetime(flight_path['time'])
            lats = flight_path['lat']
            lons = flight_path['lon']

            for i in range(1, len(times)):
                  time_gap = (times[i] - times[i - 1]).total_seconds()
                  distance_travelled = np.round(geodesic((lats[i - 1], lons[i - 1]), (lats[i], lons[i])).meters)
                  if time_gap < 0:
                        print(f"""Negative time jump in flight path {flight_path_name}, between indexes {i-1} and {i}.\n""")
                  elif time_gap > time_gap_thresh:
                        print(f"""Large time gap in flight path {flight_path_name}, between indexes {i-1} and {i}.\n"""
                              f"""Time gap: {time_gap} seconds, distance gap: {distance_travelled/1000} km\n""")
                  elif abs((distance_travelled / time_gap) - flight_path['velocity'][i]) > 10:   # Velocity check over 1 index
                        print(f"Velocity mismatch at index {i} in {flight_path_name}.\n"
                              f"Calculated velocity: {distance_travelled / time_gap} m/s, reported velocity: {flight_path['velocity'][i]} m/s\n")
                  elif i % vel_sample_size == 0:                                          # Every vel_sample_size indexes, check velocity over sample
                        time_gap = (times[i] - times[i - vel_sample_size]).total_seconds()
                        distance_travelled = 0
                        for j in range(i - vel_sample_size, i):
                                distance_travelled += geodesic((lats[j], lons[j]), (lats[j + 1], lons[j + 1])).meters
                        distance_travelled = np.round(distance_travelled)
                        displacement = np.round(geodesic((lats[i - vel_sample_size], lons[i - vel_sample_size]), (lats[i], lons[i])).meters)
                        speed_step = np.round((displacement / time_gap))
                        velocity_step = np.round((distance_travelled / time_gap))
                        velocity_reported = np.round(np.mean(flight_path['velocity'][i-vel_sample_size:i]))
                        if abs(velocity_step - velocity_reported) > vel_mismatch_thresh:
                              vel_mismatch_lat.append(lats[i-vel_sample_size:i])
                              vel_mismatch_lon.append(lons[i-vel_sample_size:i])
                              print(f"""Reported speed mismatched with distance travelled in {flight_path_name}, between indexes {i-vel_sample_size} and {i}.\n"""
                                    f"""Reported velocity: {velocity_reported} m/s, required velocity: {velocity_step} m/s, required speed: {speed_step} m/s\n"""
                                    f"""Time gap: {time_gap} seconds, distance travelled: {distance_travelled/1000} km, displacement: {displacement/1000} km\n""")
      return vel_mismatch_lat, vel_mismatch_lon


def plot_cartopy(flight_paths):
      plt.figure(figsize=(10, 5))
      ax = plt.axes(projection=ccrs.PlateCarree())
      ax.add_feature(cfeature.LAND)
      ax.add_feature(cfeature.OCEAN)
      # ax.add_feature(cfeature.COASTLINE)
      ax.add_feature(cfeature.BORDERS, linestyle=':')
      # ax.add_feature(cfeature.LAKES, alpha=0.5)
      # ax.add_feature(cfeature.RIVERS)

      min_lon = min(min(flight_path['lon']) for flight_path in flight_paths.values())
      max_lon = max(max(flight_path['lon']) for flight_path in flight_paths.values())
      min_lat = min(min(flight_path['lat']) for flight_path in flight_paths.values())
      max_lat = max(max(flight_path['lat']) for flight_path in flight_paths.values())
      lon_margin = (max_lon - min_lon) * 0.1
      lat_margin = (max_lat - min_lat) * 0.1
      min_lon -= lon_margin
      max_lon += lon_margin
      min_lat -= lat_margin
      max_lat += lat_margin

      ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

      alt_measured = "geoaltitude"
      alt_measured = "baroaltitude"

      max_velocity = max(max(flight_path['velocity']) for flight_path in flight_paths.values())
      min_velocity = min(min(flight_path['velocity']) for flight_path in flight_paths.values())
      print(f"Max velocity: {max_velocity}, Min velocity: {min_velocity}")

      # Filter out unrealistic altitude changes
      for flight_path in flight_paths.values():
            flight_path[alt_measured] = flight_path[alt_measured].mask(flight_path[alt_measured].diff().abs() > 100)

      max_altitude = max(max(flight_path[alt_measured].dropna()) for flight_path in flight_paths.values())
      min_altitude = min(min(flight_path[alt_measured].dropna()) for flight_path in flight_paths.values())
      print(f"Max altitude: {max_altitude}, Min altitude: {min_altitude}")

      # Find the flight path and index where the max altitude occurs
      for flight_path_name, flight_path in flight_paths.items():
            if max_altitude in flight_path[alt_measured].values:
                  max_altitude_index = flight_path[alt_measured].idxmax()
                  max_altitude_location = (flight_path['lat'].iloc[max_altitude_index], flight_path['lon'].iloc[max_altitude_index])
                  print(f"Max altitude occurs in flight path {flight_path_name} at index {max_altitude_index}, location: {max_altitude_location}")
                  break

      cbar_var = "Velocity"
      cbar_var = "Altitude"

      if cbar_var == "Velocity":
            cbar_label = "Velocity (m/s)"
            norm = Normalize(vmin=min_velocity, vmax=max_velocity)
      elif cbar_var == "Altitude":
            cbar_label = "Altitude (m)"
            norm = Normalize(vmin=min_altitude, vmax=max_altitude)

      # Plot the flight paths
      for flight_path_name, flight_path in flight_paths.items():
            # Plot the path
            plt.plot(flight_path['lon'], flight_path['lat'], c='red', linestyle='-', linewidth=0.5, alpha=0.7) #, label=f'{flight_path_name} Path'

            # Remove the interpolated points before plotting the scatter points
            flight_path = flight_path[~flight_path['interpolated']]
            if cbar_var == "Altitude":
                  c = flight_path[alt_measured]
            elif cbar_var == "Velocity":
                  c = flight_path['velocity']
            points = plt.scatter(flight_path['lon'], flight_path['lat'], c=c, cmap='viridis', norm=norm, s=4, alpha=1.0)

      # Plot the interpolated points
      # interpolated_points = pd.concat([flight_path[flight_path['interpolated']] for flight_path in flight_paths.values()])
      # plt.scatter(interpolated_points['lon'], interpolated_points['lat'], c='red', s=10, alpha=0.6, label='Interpolated Points')

      cbar = plt.colorbar(points, orientation='horizontal', pad=0.05, aspect=50, label=cbar_label)
      cbar.ax.set_position([0.215, -0.05, 0.6, 0.3])  # Adjust the position and size of the colorbar
      # plt.scatter(vel_mismatch_lon, vel_mismatch_lat, c='red', s=15, marker='x', label='Velocity Mismatched with Distance')

      plt.legend()
      plt.show()

def detail_plot(flight_paths):

      # Define the area around Heathrow
      cen_lat, cen_lon = 52.3105, 4.7683
      # cen_lat, cen_lon = 51.4700, -0.4543
      size_lat = 0.1
      size_lon = 0.15

      # Filter points within the plot area
      lat = []
      lon = []
      velocities = []
      for flight_path_name, flight_path in flight_paths.items():
            mask = (
                    (flight_path['lat'] >= cen_lat - size_lat) &
                    (flight_path['lat'] <= cen_lat + size_lat) &
                    (flight_path['lon'] >= cen_lon - size_lon) &
                    (flight_path['lon'] <= cen_lon + size_lon)
            )
            lat.extend(flight_path['lat'][mask])
            lon.extend(flight_path['lon'][mask])
            velocities.extend(flight_path['velocity'][mask])

      # Create a GeoDataFrame for the scatter points
      gdf_points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat), crs="EPSG:4326")

      # Convert to Web Mercator for compatibility with contextily
      gdf_points = gdf_points.to_crs(epsg=3857)

      # Plot the scatter points
      fig, ax = plt.subplots(figsize=(10, 10))
      gdf_points.plot(ax=ax, color='blue', marker='.', markersize=5, label='Locations', alpha=0.7)

      # Add a basemap
      ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=14)

      # Add title and legend
      ax.set_axis_off()
      plt.title('Flight Paths around Heathrow')
      plt.legend()
      plt.show()




      


flight_paths = load_flight_paths()
flight_paths = interpolate_great_circle(flight_paths)
vel_mismatch_lat, vel_mismatch_lon = analyse_data(flight_paths)
# plot_cartopy(flight_paths)

# detail_plot(flight_paths)