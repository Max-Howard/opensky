import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
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
                  flight_paths[file.replace('.csv', '')] = df
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
                  distance_travelled = np.round(geodesic((lats[i - 1], lons[i - 1]), (lats[i], lons[i])).kilometers)
                  if time_gap < 0:
                        print(f"""Negative time jump in flight path {flight_path_name}, between indexes {i-1} and {i}.\n""")
                  elif time_gap > time_gap_thresh:
                        print(f"""Large time gap in flight path {flight_path_name}, between indexes {i-1} and {i}.\n"""
                              f"""Time gap: {time_gap} seconds, distance gap: {distance_travelled} km\n""")
                  elif distance_travelled / time_gap - flight_path['velocity'][i] > 10:   # Velocity check over 1 index
                        print(f"Velocity mismatch at index {i} in {flight_path_name}.\n"
                              f"Calculated velocity: {distance_travelled / time_gap} m/s, reported velocity: {flight_path['velocity'][i]} m/s\n")
                  elif i % vel_sample_size == 0:                                          # Every vel_sample_size indexes, check velocity over sample
                        time_gap = (times[i] - times[i - vel_sample_size]).total_seconds()
                        distance_travelled = 0
                        for j in range(i - vel_sample_size, i):
                                distance_travelled += geodesic((lats[j], lons[j]), (lats[j + 1], lons[j + 1])).kilometers
                        distance_travelled = np.round(distance_travelled)
                        displacement = np.round(geodesic((lats[i - vel_sample_size], lons[i - vel_sample_size]), (lats[i], lons[i])).kilometers)
                        speed_step = np.round((displacement * 1000 / time_gap))
                        velocity_step = np.round((distance_travelled * 1000 / time_gap))
                        velocity_reported = np.round(np.mean(flight_path['velocity'][i-vel_sample_size:i]))
                        if abs(velocity_step - velocity_reported) > vel_mismatch_thresh:
                              vel_mismatch_lat.append(lats[i-vel_sample_size:i])
                              vel_mismatch_lon.append(lons[i-vel_sample_size:i])
                              print(f"""Reported speed mismatched with distance travelled in {flight_path_name}, between indexes {i-vel_sample_size} and {i}.\n"""
                                    f"""Reported velocity: {velocity_reported} m/s, required velocity: {velocity_step} m/s, required speed: {speed_step} m/s\n"""
                                    f"""Time gap: {time_gap} seconds, distance travelled: {distance_travelled} km, displacement: {displacement} km\n""")
      return vel_mismatch_lat, vel_mismatch_lon


def plot_cartopy(flight_paths):
      fig = plt.figure(figsize=(10, 5))
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

      max_velocity = max(max(flight_path['velocity']) for flight_path in flight_paths.values())
      min_velocity = min(min(flight_path['velocity']) for flight_path in flight_paths.values())
      norm = Normalize(vmin=min_velocity, vmax=max_velocity)

      for flight_path_name, flight_path in flight_paths.items():
            points = plt.scatter(flight_path['lon'], flight_path['lat'], c=flight_path['velocity'], cmap='viridis', norm=norm, s=5, label=f'{flight_path_name} Points', alpha=1.0)
            plt.plot(flight_path['lon'], flight_path['lat'], linestyle='-', linewidth=0.5, label=f'{flight_path_name} Path', alpha=0.7)

      cbar = plt.colorbar(points, orientation='horizontal', pad=0.05, aspect=50, label='Velocity')
      # plt.scatter(vel_mismatch_lon, vel_mismatch_lat, c='red', s=15, marker='x', label='Velocity Mismatch Anomalies')

      # plt.legend()
      plt.show()


flight_paths = load_flight_paths()
vel_mismatch_lat, vel_mismatch_lon = analyse_data(flight_paths)
plot_cartopy(flight_paths)