import pandas as pd
import matplotlib.pyplot as plt
import os
from geopy.distance import geodesic
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def load_flight_paths():
      flight_paths = {}
      for file in os.listdir('./output'):
            if file.endswith('.csv'):
                  file_path = os.path.join('./output', file)
                  df = pd.read_csv(file_path)
                  initial_length = len(df)
                  df = df.dropna().reset_index(drop=True)   # Drop rows with missing data otherwise sorting will fill with NaN
                  final_length = len(df)
                  rows_removed = initial_length - final_length
                  print(f"Removed {rows_removed} rows with missing data in flight path {file}\n")
                  df = df.sort_values(by='time') # TODO not sure why this is necessary
                  flight_paths[file.replace('.csv', '')] = df

      return flight_paths


def analyse_data(flight_paths, time_gap_thresh=60, vel_mismatch_thresh=10, vel_sample_size=200, replace_vel_with_speed=False, position_update_thresh=30):
      anomalies = {"neg_time": {"lats": [], "lons": []},
                   "large_time_gap": {"lats": [], "lons": []},
                   "vel_mismatch": {"lats": [], "lons": []}}
      for flight_path_name, flight_path in flight_paths.items():

            initial_length = len(flight_path)
            flight_path = flight_path.dropna().reset_index(drop=True)
            final_length = len(flight_path)
            rows_removed = initial_length - final_length
            print(f"Removed {rows_removed} rows with missing data in flight path {flight_path_name}\n")

            times = pd.to_datetime(flight_path['time'])
            last_pos_update = pd.to_datetime(flight_path['lastposupdate'], unit='s', origin='unix', utc=True)
            lats = flight_path['lat']
            lons = flight_path['lon']
            
            for i in range(1, len(times)):
                  time_gap = (times[i] - times[i - 1]).total_seconds()
                  distance_travelled = np.round(geodesic((lats[i - 1], lons[i - 1]), (lats[i], lons[i])).kilometers)
                  if time_gap < 0:
                        anomalies["neg_time"]["lats"].append(lats[i])
                        anomalies["neg_time"]["lons"].append(lons[i])
                        print(f"""Negative time jump in flight path {flight_path_name}, between indexes {i-1} and {i}.\n"""
                              f"""Initial time: {times[i-1]}, final time: {times[i]}\n""")
                  elif time_gap > time_gap_thresh:
                        anomalies["large_time_gap"]["lats"].append(lats[i])
                        anomalies["large_time_gap"]["lons"].append(lons[i])
                        print(f"""Large time gap in flight path {flight_path_name}, between indexes {i-1} and {i}.\n"""
                              f"""Time gap: {time_gap} seconds, distance gap: {distance_travelled} km\n""")
                  if i % vel_sample_size == 0:
                        # continue
                        time_gap = (times[i] - times[i - vel_sample_size]).total_seconds()
                        distance_travelled = 0
                        for j in range(i - vel_sample_size, i):
                                distance_travelled += geodesic((lats[j], lons[j]), (lats[j + 1], lons[j + 1])).kilometers
                        distance_travelled = np.round(distance_travelled)
                        displacement = np.round(geodesic((lats[i - vel_sample_size], lons[i - vel_sample_size]), (lats[i], lons[i])).kilometers)
                        speed_step = np.round((displacement * 1000 / time_gap))
                        velocity_step = np.round((distance_travelled * 1000 / time_gap))
                        velocity_reported = np.round(np.mean(flight_path['velocity'][i-vel_sample_size:i]))
                        if time_gap > time_gap_thresh * vel_sample_size or time_gap < 0:
                              continue # Skip velocity check if time gap is too large or negative time gap
                        elif (times[i] - last_pos_update[i]).total_seconds() > position_update_thresh: # Skip if lastposupdate is not available
                              print(f"Skipping velocity mismatch check for {flight_path_name} at index {i} due to out of date position data. ({(times[i] - last_pos_update[i]).total_seconds()}s out of date.)\n")
                              continue
                        elif (times[i-vel_sample_size] - last_pos_update[i-vel_sample_size]).total_seconds() > position_update_thresh:
                              continue # No need to print as the previous check will have already printed the same message
                        if not replace_vel_with_speed:
                              if abs(velocity_step - velocity_reported) > vel_mismatch_thresh:
                                    anomalies["vel_mismatch"]["lats"].append(lats[i-vel_sample_size:i])
                                    anomalies["vel_mismatch"]["lons"].append(lons[i-vel_sample_size:i])
                                    print(f"""Reported speed mismatched with distance travelled in {flight_path_name}, between indexes {i-vel_sample_size} and {i}.\n"""
                                          f"""Reported velocity: {velocity_reported} m/s, required velocity: {velocity_step} m/s, required speed: {speed_step} m/s\n"""
                                          f"""Time gap: {time_gap} seconds, distance travelled: {distance_travelled} km, displacement: {displacement} km\n""")
                        else:
                              if abs(speed_step - velocity_reported) > vel_mismatch_thresh:
                                    anomalies["vel_mismatch"]["lats"].append(lats[i-vel_sample_size:i])
                                    anomalies["vel_mismatch"]["lons"].append(lons[i-vel_sample_size:i])
                                    print(f"""Reported speed mismatched with displacement in {flight_path_name}, between indexes {i-vel_sample_size} and {i}.\n"""
                                          f"""Reported velocity: {velocity_reported} m/s, required speed: {speed_step} m/s\n"""
                                          f"""Time gap: {time_gap} seconds, distance travelled: {distance_travelled} km, displacement: {displacement} km\n""")
      return anomalies

def plot_cartopy(flight_paths):
      fig = plt.figure(figsize=(10, 5))
      ax = plt.axes(projection=ccrs.PlateCarree())
      ax.add_feature(cfeature.LAND)
      ax.add_feature(cfeature.OCEAN)
      ax.add_feature(cfeature.COASTLINE)
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

      for flight_path_name, flight_path in flight_paths.items():
            points = plt.scatter(flight_path['lon'], flight_path['lat'], c=range(len(flight_path['lat'])), cmap='viridis', s=5, label=f'{flight_path_name} Points', alpha=1.0)
            cbar = plt.colorbar(points, orientation='horizontal', pad=0.05, aspect=50, label=f'Index for {flight_path_name}')

      plt.scatter(anomalies["neg_time"]["lons"], anomalies["neg_time"]["lats"], c='red', s=10, marker='x', label='Negative Time Anomalies')
      plt.scatter(anomalies["large_time_gap"]["lons"], anomalies["large_time_gap"]["lats"], c='orange', s=10, marker='x', label='Large Time Gap Anomalies')
      plt.scatter(anomalies["vel_mismatch"]["lons"], anomalies["vel_mismatch"]["lats"], c='purple', s=10, marker='x', label='Velocity Mismatch Anomalies')

      plt.legend()
      plt.show()


flight_paths = load_flight_paths()
anomalies = analyse_data(flight_paths, replace_vel_with_speed=False)
plot_cartopy(flight_paths)