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
from shapely.geometry import Point


TIME_GAP = 60
ALT_GAP_MAX = 100 # Maximum altitude gap in meters before interpolation
CRUISE_STEP = 50 * 1852       # 50 nautical miles in meters
DELTA_ALT_CRUISE = 3000 * 0.3048    # 3000 ft in meters
DELTA_ALT_CRUISE = 16000 * 0.3048    # 16000 ft in meters
FT_TO_M = 0.3048
AIRPORT_NAMES = pd.read_csv("airports.csv").set_index("icao")

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
        raise ValueError('fmt must have length >= 2')
    
    values = []
    pos = 0
    for i in range(int(len(fmt)/2)):
        pos += fmt[i*2]
        length = fmt[i*2+1]
        value = s[pos:pos+length]
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
    with open(filepath, 'r', encoding='cp1252') as f:
        lines = f.readlines()
        
        # Header
        line = lines[7]
        ptf['Vcl1'] = float(line[11:14])
        ptf['Vcl2'] = float(line[15:18])
        ptf['Mcl'] = float(line[23:27])
        ptf['mlo'] = float(line[41:])
        
        line = lines[8]
        ptf['Vcr1'] = float(line[11:14])
        ptf['Vcr2'] = float(line[15:18])
        ptf['Mcr'] = float(line[23:27])
        ptf['mnom'] = float(line[41:54])
        ptf['hMO'] = float(line[71:])
        
        line = lines[9]
        ptf['Vdes1'] = float(line[11:14])
        ptf['Vdes2'] = float(line[15:18])
        ptf['Mdes'] = float(line[23:27])
        ptf['mhi'] = float(line[41:])
        
        # Table
        table = []
        irow = 0
        line = lines[16]
        while line[0] != '=':
            values = fixed_spacing_floats(line, [0, 3, 4, 3, 3, 5, 1, 5, 1, 5,
                                                 5, 3, 3, 5, 1, 5, 1, 5, 3, 5,
                                                 5, 3, 2, 5, 2, 5])
            table.append(values)
            irow += 1
            line = lines[16+irow*2]
        ptf['table'] = pd.DataFrame(
            data=table,
            columns=['FL', 'Vcr', 'flo_cr', 'fnom_cr', 'fhi_cr',
                     'Vcl', 'ROCDlo_cl', 'ROCDnom_cl', 'ROCDhi_cl', 'fnom_cl',
                     'Vdes', 'ROCDnom_des', 'fnom_des']
        ).set_index('FL')
    
    return ptf

def load_flight_paths(path="./output") -> dict[pd.DataFrame]:
      """
      Load all flight paths from the output folder.
      Add a gap column to the dataframes to indicate points with a time gap greater than TIME_GAP.
      Return a dictionary of flight paths.
      """
      flight_paths = {}
      for file in os.listdir(path):
            if file.endswith('.csv'):
                  file_path = os.path.join(path, file)
                  df = pd.read_csv(file_path)
                  print(f"Loaded {file} with {len(df)} rows.")
                  flight_paths[file.replace('.csv', '')] = df
      return flight_paths

def find_ptf_data_from_alt(alt, ac_type):
            ptf_filepath = f"BADA/{ac_type}__.ptf"
            ptf = read_ptf(ptf_filepath)
            ac_data = ptf['table']
            closest_FL = ac_data.index[np.abs(ac_data.index - alt / 100 / FT_TO_M).argmin()]
            return ac_data.loc[closest_FL]


def clean_alt_data(flight_paths: dict[pd.DataFrame]):
      for name, flight_path in flight_paths.items():
            initial_length = len(flight_path)
            flight_path = flight_path[flight_path["baroaltitude"].diff().abs() <= 100]
            mid_length = len(flight_path)
            flight_path = flight_path[flight_path["geoaltitude"].diff().abs() <= 100]
            final_length = len(flight_path)
            if initial_length - final_length > 0:
                  print(f"Removed {initial_length - mid_length} rows in {name} due to baroaltitude anomalies.")
                  print(f"Removed {mid_length - final_length} rows in {name} due to geoaltitude anomalies.")
            flight_paths[name] = flight_path.reset_index(drop=True)
      return flight_paths


# def interpolate_great_circle(flight_paths: dict[pd.DataFrame]):
#       ALT_GAP_MAX = 100
#       CLIMB_THRESHOLD = 50 # m Min climb required to separate climb from cruise
#       for flight_name in flight_paths.keys():
#             df:pd.DataFrame = flight_paths[flight_name]

#             origin, destination, typecode, icao24, flight_num = flight_name.split('_')
#             origin_alt = AIRPORT_NAMES[AIRPORT_NAMES["icao"] == origin]["alt"][0]
#             dest_alt = AIRPORT_NAMES[AIRPORT_NAMES["icao"] == destination]["alt"][0]
#             # Find the start and end of the cruise phase
#             non_LTO_idx_start = DELTA_ALT_CRUISE + origin_alt
#             non_LTO_idx_end = DELTA_ALT_CRUISE + dest_alt
#             df["interpolated"] = False

#             # Find gaps in the altitude data
#             alt_gaps = df["geoaltitude"].diff().abs() > ALT_GAP_MAX
#             alt_gap_indexes = alt_gaps[alt_gaps].index.tolist()

#             # Find time gaps greater than TIME_GAP
#             time_gaps = df["time"].diff() > TIME_GAP
#             time_gap_indexes = time_gaps[time_gaps].index.tolist()

#             combined_gap_indexes = sorted(list(set(alt_gap_indexes + time_gap_indexes)))
#             added_rows = 0
#             for gap_num,idx in enumerate(combined_gap_indexes):

#                   idx_corr = idx + added_rows # Corrected index to account for added rows
#                   start_row = df.iloc[idx_corr - 1]
#                   end_row = df.iloc[idx_corr]
#                   start_time = start_row["time"]
#                   end_time = end_row["time"]
#                   gap_total_seconds = (end_time - start_time)
#                   start_baro_alt = start_row["baroaltitude"]
#                   end_baro_alt = end_row["baroaltitude"]
#                   start_geo_alt = start_row["geoaltitude"]
#                   end_geo_alt = end_row["geoaltitude"]
#                   cl_rate_required = end_geo_alt - start_geo_alt / gap_total_seconds
#                   ave_alt = (start_geo_alt + end_geo_alt) / 2

#                   # Initialize the geodesic line
#                   geod = Geodesic.WGS84
#                   line = geod.InverseLine(start_row["lat"], start_row["lon"], end_row["lat"], end_row["lon"])
#                   gap_total_distance = line.s13

#                   data_at_fl = find_ptf_data_from_alt(ave_alt, typecode)
#                   if start_geo_alt < end_geo_alt - CLIMB_THRESHOLD:
#                         cl_rate = data_at_fl["ROCDnom_cl"]
#                         if cl_rate < cl_rate_required:
#                               climb_time = (end_geo_alt - start_geo_alt) / cl_rate
#                               climb_velocity = data_at_fl["Vcl"]
#                         else:
#                               raise ValueError("Climb rate required is greater than the aircraft's climb rate.")

#                   num_points = int(np.ceil(gap_total_distance / CRUISE_STEP)) # Number of points to add to fill the gap

#                   print(f"Adding {num_points} points to fill gap {gap_num + 1} of {len(combined_gap_indexes)}, in {flight_name}, at index {idx}.")

#                   time_intervals = np.linspace(start_time, end_time, num_points + 2)[1:-1]
#                   step = gap_total_distance / (num_points + 1)
#                   lat_intervals = []
#                   lon_intervals = []
#                   for i in range(1, num_points+1): # Skip the first and last points as they are already in the dataframe
#                         point = line.Position(i * step)
#                         lat_intervals.append(point['lat2'])
#                         lon_intervals.append(point['lon2'])

#                   velocity_intervals = np.linspace(start_row["velocity"], end_row["velocity"], num_points + 2)[1:-1]
#                   gap_mean_velocity = gap_total_distance / gap_total_seconds
#                   velocity_correction = gap_mean_velocity - np.mean(velocity_intervals)
#                   velocity_intervals += velocity_correction # TODO this may introduce errors in the velocity data
#                   print(f"Added: {velocity_correction}, to interval velocities, to ensure continuity with distance travelled.")
#                   interpolated_rows = pd.DataFrame({
#                               "time": time_intervals,
#                               "lat": lat_intervals,
#                               "lon": lon_intervals,
#                               "velocity": velocity_intervals
#                   })
#                   interpolated_rows["interpolated"] = True
#                   added_rows += len(interpolated_rows)
#                   df = pd.concat([df.iloc[:idx_corr], interpolated_rows, df.iloc[idx_corr:]]).reset_index(drop=True)
#             flight_paths[flight_name] = df
#       return flight_paths

def find_time_diff(flight_paths):
      for flight_name, df in flight_paths.items():
            df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
            df["lastposupdate"] = pd.to_datetime(df["lastposupdate"], unit='s').dt.tz_localize(None)
            df["time_diff"] = (df["time"] - df["lastposupdate"]).abs()
            max_time_diff = df["time_diff"].max()
            print(f"Max time difference in {flight_name}: {max_time_diff}")

def analyse_data(flight_paths, time_gap_thresh=60, vel_mismatch_thresh=10, vel_sample_size=100):
      vel_mismatch_lat = []
      vel_mismatch_lon = []
      for flight_path_name, flight_path in flight_paths.items():
            times = flight_path['time']
            lats = flight_path['lat']
            lons = flight_path['lon']
            num_vel_mismatch = 0

            for i in range(1, len(times)):
                  time_gap = (times[i] - times[i - 1])
                  distance_travelled = np.round(geodesic((lats[i - 1], lons[i - 1]), (lats[i], lons[i])).meters)
                  if time_gap <= 0:
                        print(f"""Negative time jump ({time_gap}) in flight path {flight_path_name}, between indexes {i-1} and {i}.\n""")
                  elif time_gap > time_gap_thresh:
                        pass
                        # print(f"""Large time gap in flight path {flight_path_name}, between indexes {i-1} and {i}.\n"""
                        #       f"""Time gap: {time_gap} seconds, distance gap: {distance_travelled/1000} km\n""")
                  elif abs((distance_travelled / time_gap) - flight_path['velocity'][i]) > 10:   # Velocity check over 1 index
                        num_vel_mismatch += 1
                        # print(f"Velocity mismatch at index {i} in {flight_path_name}.\n"
                        #       f"Calculated velocity: {distance_travelled / time_gap} m/s, reported velocity: {flight_path['velocity'][i]} m/s\n")
                  elif i % vel_sample_size == 0:                                          # Every vel_sample_size indexes, check velocity over sample
                        time_gap = (times[i] - times[i - vel_sample_size])
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
            print(f"Found {num_vel_mismatch} velocity mismatches in {flight_path_name}.\n")
      return vel_mismatch_lat, vel_mismatch_lon


def plot_cartopy(flight_paths, color_by="velocity"):
      """
      
      """
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

      if color_by == "velocity":
            cbar_label = "Velocity (m/s)"
            cbar_source = "velocity"
            max_velocity = max(max(flight_path['velocity']) for flight_path in flight_paths.values())
            min_velocity = min(min(flight_path['velocity']) for flight_path in flight_paths.values())
            norm = Normalize(vmin=min_velocity, vmax=max_velocity)
      elif color_by.endswith("altitude"):
            if color_by == "geoaltitude":
                  cbar_label = "Geo Altitude (m)"
                  cbar_source = "geoaltitude"
            elif color_by == "baroaltitude":
                  cbar_label = "Baro Altitude (m)"
                  cbar_source = "baroaltitude"
            max_altitude = max(max(flight_path[cbar_source]) for flight_path in flight_paths.values())
            min_altitude = min(min(flight_path[cbar_source]) for flight_path in flight_paths.values())
            norm = Normalize(vmin=min_altitude, vmax=max_altitude)

      # Plot the flight paths
      for flight_path_name, flight_path in flight_paths.items():

            if flight_path_name != "OMDB_EGLL_B788_4242f4_1":
                  continue

            # Remove the interpolated points before plotting the scatter points
            if 'interpolated' in flight_path.columns:
                  plt.plot(flight_path['lon'], flight_path['lat'], c='red', linestyle='-', linewidth=0.5, alpha=0.7) #, label=f'{flight_path_name} Path'
                  flight_path = flight_path[~flight_path['interpolated']]
 
            c = flight_path[cbar_source]
            points = plt.scatter(flight_path['lon'], flight_path['lat'], c=c, cmap='viridis', norm=norm, s=4, alpha=1.0)

      # Plot the interpolated points
      # interpolated_points = pd.concat([flight_path[flight_path['interpolated']] for flight_path in flight_paths.values()])
      # plt.scatter(interpolated_points['lon'], interpolated_points['lat'], c='red', s=10, alpha=0.6, label='Interpolated Points')

      cbar = plt.colorbar(points, orientation='horizontal', pad=0.05, aspect=50, label=cbar_label)
      cbar.ax.set_position([0.215, -0.05, 0.6, 0.3])  # Adjust the position and size of the colorbar

      plt.legend()
      plt.show()

def detail_plot(flight_paths):
      plt.figure(figsize=(10, 5))
      # Define the center as the final lat and lon value of the first flight path
      first_flight_path = next(iter(flight_paths.values()))
      cen_lat = first_flight_path['lat'].iloc[-1]
      cen_lon = first_flight_path['lon'].iloc[-1]

      # Define the size of the plot area
      plot_size_nm = 65
      lat_step = plot_size_nm * 1.852 / 110.574
      lon_step = plot_size_nm * 1.852 / (111.320 * np.cos(np.radians(cen_lat)))

      # Filter points within the plot area
      lat = []
      lon = []
      velocities = []
      altitudes = []
      for flight_path_name, flight_path in flight_paths.items():
            mask = (
                  (flight_path['lat'] >= cen_lat - lat_step) &
                  (flight_path['lat'] <= cen_lat + lat_step) &
                  (flight_path['lon'] >= cen_lon - lon_step) &
                  (flight_path['lon'] <= cen_lon + lon_step)
            )
            lat.extend(flight_path['lat'][mask])
            lon.extend(flight_path['lon'][mask])
            velocities.extend(flight_path['velocity'][mask])
            altitudes.extend(flight_path['geoaltitude'][mask])

      # Create a GeoDataFrame for the scatter points
      gdf_flight_points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat), crs="EPSG:4326")
      gdf_flight_points = gdf_flight_points.to_crs(epsg=3857)

      # Create a circle of points around the center point
      radius = 50 * 1852  # 50 nautical miles in meters
      num_points = 100
      angles = np.linspace(0, 2 * np.pi, num_points)
      circle_points = []
      for angle in angles:
            destination = geodesic(meters=radius).destination((cen_lat, cen_lon), np.degrees(angle))
            circle_points.append((destination.longitude, destination.latitude))

      # Create a GeoDataFrame for the circle points
      gdf_circle = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in circle_points], crs="EPSG:4326")
      gdf_circle = gdf_circle.to_crs(epsg=3857)

      # Plot the scatter points
      fig, ax = plt.subplots(figsize=(6, 6))
      scatter = ax.scatter(gdf_flight_points.geometry.x, gdf_flight_points.geometry.y, c=altitudes, cmap='viridis', marker='.', s=5, alpha=0.7, vmin=0, vmax=12000)
      cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal',fraction=0.046, pad=0.04) #, pad=0.01, aspect=50
      cbar.set_label('Altitude (m)')
      # Plot the circle points
      ax.plot(gdf_circle.geometry.x, gdf_circle.geometry.y, color='red', linestyle='-', linewidth=1.5, alpha=0.7)

      # Add a basemap
      ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=8)

      # Plot settings
      minx, miny, maxx, maxy = gdf_flight_points.total_bounds
      ax.set_xlim(minx, maxx)
      ax.set_ylim(miny, maxy)
      ax.set_aspect('equal', adjustable='box')
      ax.set_axis_off()
      plt.legend(['Flight Paths', '50 Nautical Mile Radius'], loc='upper right')
      # plt.title('Flight Paths around Heathrow')
      plt.savefig("test.png",bbox_inches='tight', dpi = 300)
      plt.show()

def baro_vs_geo_altitude_plot(flight_paths):
      plt.figure(figsize=(10, 5))
      for flight_path_name, flight_path in flight_paths.items():
            time_series = (flight_path['time'] - flight_path["time"][0]) / 60**2
            plt.plot(time_series, flight_path['geoaltitude'], color="orange", label=f"{flight_path_name} Geo Altitude")
            plt.scatter(time_series, flight_path['geoaltitude'], color="orange", s=5)
            plt.plot(time_series, flight_path['baroaltitude'], color="blue", label=f"{flight_path_name} Baro Altitude")
            plt.scatter(time_series, flight_path['baroaltitude'], color="blue", s=5)
      plt.xlabel("Time (hours)")
      plt.ylabel("Altitude (m)")
      plt.show()


def altitude_plot(flight_paths):
      plt.figure(figsize=(10, 5))
      colors = plt.cm.get_cmap('tab10', len(flight_paths))
      for i, (flight_path_name, flight_path) in enumerate(flight_paths.items()):
            bada_timestep = 1
            bada_times = np.arange(0, flight_path["time"].iloc[-1]-flight_path["time"][0], bada_timestep) / 60**2
            ac_type = flight_path_name.split("_")[2]
            ptf_filepath = f"BADA/{ac_type}__.ptf"
            ptf = read_ptf(ptf_filepath)
            ac_data = ptf['table']
            alt_start = flight_path["geoaltitude"].iloc[0]
            alt_stop = flight_path["geoaltitude"].iloc[-1]
            alt_cruise = max(flight_path["geoaltitude"])

            bada_climb_alts = [alt_start]
            current_alt = bada_climb_alts[0]
            while current_alt < alt_cruise:
                  current_FL = current_alt / 100 * FT_TO_M
                  closest_FL = ac_data.index[np.abs(ac_data.index - current_FL).argmin()]
                  bada_vert_rate = ac_data.loc[closest_FL, "ROCDnom_cl"]
                  bada_climb_alts.append(current_alt)
                  current_alt += bada_timestep * bada_vert_rate * FT_TO_M / 60

            bada_decent_alts = [alt_stop]
            current_alt = bada_decent_alts[0]
            while current_alt < alt_cruise:
                  current_FL = current_alt / 100 * FT_TO_M
                  closest_FL = ac_data.index[np.abs(ac_data.index - current_FL).argmin()]
                  bada_vert_rate = ac_data.loc[closest_FL,"ROCDnom_des"]
                  bada_decent_alts.append(current_alt)
                  current_alt += bada_timestep * bada_vert_rate * FT_TO_M / 60
            
            num_points_cruise = len(bada_times) - (len(bada_climb_alts) + len(bada_decent_alts))
            bada_cruise_alts = np.linspace(alt_cruise, alt_cruise, num_points_cruise)
            bada_decent_alts = bada_decent_alts[::-1]
            bada_alts = np.concatenate([bada_climb_alts, bada_cruise_alts, bada_decent_alts])
            print(f"Length of BADA altitudes. Cruise: {len(bada_cruise_alts)}, Climb: {len(bada_climb_alts)}, Descent: {len(bada_decent_alts)}")

            time_series = (flight_path['time'] - flight_path["time"][0]) / 60**2
            # plt.plot(time_series, flight_path['baroaltitude'], label=f"Interolated {flight_path_name}", color=colors(i))
            plt.plot(bada_times, bada_alts, label=f"{ac_type} BADA {flight_path_name}", color=colors(i), linestyle='--')
            plt.scatter(time_series, flight_path['baroaltitude'], s=5, marker='x', color=colors(i), alpha=0.5)
            # Calculate altitude by integrating vertical rate
            # integrated_altitude = (np.cumsum(flight_path["vertrate"].fillna(0) * flight_path["time"].diff().fillna(0))) + flight_path["baroaltitude"].iloc[0]
            # plt.plot(time_series, integrated_altitude, label=f"Integrated Altitude {flight_path_name}", linestyle='-.', color=colors(i))
      plt.legend()
      plt.xlabel("Time (hours)")
      plt.ylabel("Baro Altitude (m)")
      plt.show()

def vert_rate_plot(flight_paths):
      for flight_path_name, flight_path in flight_paths.items():
            plt.figure(figsize=(10, 5))
            avarage_points_count = 60
            time_series = (flight_path['time'] - flight_path["time"][0]) / 60**2
            # plt.scatter(time_series, flight_path['geoaltitude'].diff() / flight_path["time"].diff(), color="orange", s=5, label="Calculated from Geo Altitude")
            # plt.scatter(time_series, flight_path['baroaltitude'].diff() / flight_path["time"].diff(), color="blue", s=5, label="Calculated from Baro Altitude")
            # plt.scatter(time_series, flight_path["vertrate"], color="green", s=5, label="Reported Vertical Rate")
            
            # Calculate and plot the mean value
            # geo_alt_mean = flight_path['geoaltitude'].diff().rolling(window=avarage_points_count).mean() / flight_path["time"].diff().rolling(window=10).mean()
            # baro_alt_mean = flight_path['baroaltitude'].diff().rolling(window=avarage_points_count).mean() / flight_path["time"].diff().rolling(window=10).mean()
            # vert_rate_mean = flight_path["vertrate"].rolling(window=avarage_points_count).mean()
            # plt.plot(time_series, geo_alt_mean, color="orange", linestyle='--', label="Mean Geo Altitude Rate")
            # plt.plot(time_series, baro_alt_mean, color="blue", linestyle='--', label="Mean Baro Altitude Rate")
            # plt.plot(time_series, vert_rate_mean, color="green", linestyle='--', label="Mean Reported Vertical Rate)")

            # Calculate and plot the median value
            geo_alt_median = flight_path['geoaltitude'].diff().rolling(window=avarage_points_count).median() / flight_path["time"].diff().rolling(window=10).median()
            baro_alt_median = flight_path['baroaltitude'].diff().rolling(window=avarage_points_count).median() / flight_path["time"].diff().rolling(window=10).median()
            vert_rate_median = flight_path["vertrate"].rolling(window=avarage_points_count).median()
            plt.plot(time_series, geo_alt_median, color="orange", linestyle='-', label="Median Geo Altitude Rate")
            plt.plot(time_series, baro_alt_median, color="blue", linestyle='-', label="Median Baro Altitude Rate")
            plt.plot(time_series, vert_rate_median, color="green", linestyle='-', label="Median Reported Vertical Rate")

            plt.legend()
            plt.title("Altitude Rate vs Time for " + flight_path_name)
            plt.xlabel("Time (hours)")
            plt.ylabel("Altitude Rate (m/s)")
            plt.ylim(-25, 25)
            plt.text(0.5, 0.95, 'Vertical rate axis limited to [-25, 25] m/s', 
                         horizontalalignment='center', verticalalignment='center', 
                         transform=plt.gca().transAxes, fontsize=10, color='red')
            plt.text(0.5, 0.9, f'Median values calculated over {avarage_points_count} points', 
                         horizontalalignment='center', verticalalignment='center', 
                         transform=plt.gca().transAxes, fontsize=10, color='red')
            plt.show()

def vert_rate_vs_altitude(flight_paths):
      avarage_points_count = 5
      for flight_path_name, flight_path in flight_paths.items():
            plt.figure(figsize=(10, 5))
            ac_type = flight_path_name.split("_")[2]
            ptf_filepath = f"BADA/{ac_type}__.ptf"
            ptf = read_ptf(ptf_filepath)
            ac_data = ptf['table']
            time_series = (flight_path['time'] - flight_path["time"][0]) / 60**2
            geo_rate_alt_median = flight_path['geoaltitude'].diff().rolling(window=avarage_points_count).median() / flight_path["time"].diff().rolling(window=10).median()
            baro_rate_alt_median = flight_path['baroaltitude'].diff().rolling(window=avarage_points_count).median() / flight_path["time"].diff().rolling(window=10).median()
            sc1 = plt.scatter(flight_path['geoaltitude'], flight_path["vertrate"], s=8, c=time_series, cmap='plasma', marker='x', label="ADS-B Altitude Rate vs GPS Altitude", alpha=1)
            sc2 = plt.scatter(flight_path['geoaltitude'], geo_rate_alt_median, s=8, c=time_series, cmap='plasma', marker='o', label=f"GPS Altitude Rate vs GPS Altitude (Median over {avarage_points_count} values)", alpha=1)
            cbar = plt.colorbar(sc1, orientation='vertical', pad=0.01)
            cbar.set_label('Time (h) for geoaltitude points')

            sc3 = plt.scatter(flight_path['baroaltitude'], flight_path["vertrate"], s=8, c=time_series, cmap='viridis', marker='x', label="ADS-B Altitude Rate vs Baro Altitude", alpha=1)
            sc4 = plt.scatter(flight_path['baroaltitude'], baro_rate_alt_median, s=8, c=time_series, cmap='viridis', marker='o', label=f"Baro Altitude Rate vs Baro Altitude (Median over {avarage_points_count} values)", alpha=1)
            cbar = plt.colorbar(sc3, orientation='vertical', pad=0.01)
            cbar.set_label('Time (h) for baroaltitude points')

            plt.plot(ac_data.index * 100 * FT_TO_M, ac_data["ROCDhi_cl"] * FT_TO_M/ 60, label="BADA High Load Climb Rate", c = "red")
            plt.plot(ac_data.index * 100 * FT_TO_M, ac_data["ROCDnom_cl"] * FT_TO_M/ 60, label="BADA Nominal Load Climb Rate", c = "green")
            plt.plot(ac_data.index * 100 * FT_TO_M, ac_data["ROCDlo_cl"] * FT_TO_M/ 60, label="BADA Low Climb Load Rate", c = "purple")
            plt.plot(ac_data.index * 100 * FT_TO_M, ac_data["ROCDnom_des"] * FT_TO_M/ -60, label="BADA Nominal Load Descent Rate", c = "blue")
            plt.xlabel("Altitude (m)")
            plt.ylabel("Altitude Rate (m/s)")
            plt.title(f"{flight_path_name} Altitude Rate vs Altitude (geo alt for measured rate)")
            plt.ylim(-25, 25)
            plt.text(0.5, 0.95, 'Vertical rate axis limited to [-25, 25] m/s', 
                         horizontalalignment='center', verticalalignment='center', 
                         transform=plt.gca().transAxes, fontsize=10, color='red')
            plt.legend()
            plt.show()

flight_paths = load_flight_paths("output")



flight_paths = interpolate_great_circle(flight_paths)
# vel_mismatch_lat, vel_mismatch_lon = analyse_data(flight_paths)
# detail_plot(flight_paths)

flight_paths = clean_alt_data(flight_paths)
print(flight_paths.keys())
plot_cartopy(flight_paths, color_by="geoaltitude")
# altitude_plot(flight_paths)
# vert_rate_plot(flight_paths)
# vert_rate_vs_altitude(flight_paths)