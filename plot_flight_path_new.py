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
from matplotlib.lines import Line2D


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

def load_flight_paths(path="./output", include=None, limit=None) -> dict[pd.DataFrame]:
      """
      Load all flight paths from the output folder.
      Add a gap column to the dataframes to indicate points with a time gap greater than TIME_GAP.
      Return a dictionary of flight paths.
      """
      flight_paths = {}
      for file in os.listdir(path):
            if not file.endswith('.csv'):
                  continue
            origin, destination, typecode, icao24, flight_number = file.strip(".csv").split("_")
            if include and include not in file:
                  continue
            if limit and len(flight_paths) >= limit:
                  break
            # if destination != "EGLL":
            #       continue

            file_path = os.path.join(path, file)
            df = pd.read_csv(file_path)
            # print(f"Loaded {file} with {len(df)} rows.")
            flight_paths[file.replace('.csv', '')] = df
      print(f"Loaded {len(flight_paths)} flight paths.")
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
      Plot flight paths on a map with optional color coding.
      color_by: "velocity", "geoaltitude", "baroaltitude", or "index"
      """
      plt.figure(figsize=(12, 6))
      ax = plt.axes(projection=ccrs.PlateCarree())
      ax.add_feature(cfeature.LAND)
      ax.add_feature(cfeature.OCEAN)
      ax.add_feature(cfeature.COASTLINE)
      ax.add_feature(cfeature.BORDERS, linestyle=':')
      ax.add_feature(cfeature.LAKES, alpha=0.5)
      ax.add_feature(cfeature.RIVERS)

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

      # Set plot limits centered around Heathrow Airport (EGLL)
      heathrow_lat = 51.4700
      heathrow_lon = -0.4543
      min_lon = heathrow_lon - 0.5 * 1.60934 / (111.320 * np.cos(np.radians(heathrow_lat)))
      max_lon = heathrow_lon + 21 * 1.60934 / (111.320 * np.cos(np.radians(heathrow_lat)))
      min_lat = heathrow_lat - 2 * 1.60934 / 110.574
      max_lat = heathrow_lat + 15 * 1.60934 / 110.574

      ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

      if color_by == "velocity":
            cbar_label = "Velocity (m/s)"
            cbar_source = "velocity"
            max_val = max(max(flight_path['velocity']) for flight_path in flight_paths.values())
            min_val = min(min(flight_path['velocity']) for flight_path in flight_paths.values())
            norm = Normalize(vmin=min_val, vmax=max_val)
      elif color_by.endswith("altitude"):
            if color_by == "geoaltitude":
                  cbar_label = "GNSS Altitude (m)"
                  cbar_source = "geoaltitude"
            elif color_by == "baroaltitude":
                  cbar_label = "Barometric Altitude (m)"
                  cbar_source = "baroaltitude"
            max_val = min(5000, max(max(flight_path[cbar_source]) for flight_path in flight_paths.values()))
            min_val = min(min(flight_path[cbar_source]) for flight_path in flight_paths.values())
            norm = Normalize(vmin=min_val, vmax=max_val)
      elif color_by == "index":
            cbar_label = "Index"
            cbar_source = None
            max_val = max(len(flight_path) for flight_path in flight_paths.values())
            min_val = 0
            norm = Normalize(vmin=min_val, vmax=max_val)
      else:
            raise ValueError(f"Unsupported color_by: {color_by}")

      # Plot the flight paths
      for flight_path_name, flight_path in flight_paths.items():
            # Remove the interpolated points before plotting the scatter points
            if 'interpolated' in flight_path.columns:
                  plt.plot(flight_path['lon'], flight_path['lat'], c='red', linestyle='-', linewidth=0.5, alpha=0.7)
                  flight_path = flight_path[~flight_path['interpolated']]

            if color_by == "index":
                  c = np.arange(len(flight_path))
            else:
                  c = flight_path[cbar_source]
            points = plt.scatter(flight_path['lon'], flight_path['lat'], c=c, cmap='viridis_r', marker=".", norm=norm, s=8, alpha=1.0)

      cbar = plt.colorbar(points, orientation='horizontal', pad=0.05, aspect=50, label=cbar_label)
      cbar.ax.set_position([0.215, -0.05, 0.6, 0.3])  # Adjust the position and size of the colorbar
      plt.gca().xaxis.set_major_locator(plt.NullLocator())
      plt.gca().yaxis.set_major_locator(plt.NullLocator())
      plt.legend()
      plt.show()
      # plt.savefig("test.png",bbox_inches='tight')

def detail_plot(flight_paths):
      plt.figure(figsize=(10, 5))
      # Define the center as the final lat and lon value of the first flight path
      first_flight_path = next(iter(flight_paths.values()))
      cen_lat = first_flight_path['lat'].iloc[-1]
      cen_lon = first_flight_path['lon'].iloc[-1]
      # cen_lat = 52.3169
      # cen_lon = 4.7459
      cen_lat = 51.478
      cen_lon = -0.4642979

      # Define the size of the plot area
      plot_size_nm = 2.5
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
                  (flight_path['lon'] <= cen_lon + lon_step) &
                  (flight_path['geoaltitude'] <= 750)
            )
            lat.extend(flight_path['lat'][mask])
            lon.extend(flight_path['lon'][mask])
            velocities.extend(flight_path['tas'][mask])
            altitudes.extend(flight_path['geoaltitude'][mask])

      # Create a GeoDataFrame for the scatter points
      gdf_flight_points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat), crs="EPSG:4326")
      gdf_flight_points = gdf_flight_points.to_crs(epsg=3857)

      # Create a circle of points around the center point
      # radius = 50 * 1852  # 50 nautical miles in meters
      # num_points = 100
      # angles = np.linspace(0, 2 * np.pi, num_points)
      # circle_points = []
      # for angle in angles:
      #       destination = geodesic(meters=radius).destination((cen_lat, cen_lon), np.degrees(angle))
      #       circle_points.append((destination.longitude, destination.latitude))

      # Create a GeoDataFrame for the circle points
      # gdf_circle = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in circle_points], crs="EPSG:4326")
      # gdf_circle = gdf_circle.to_crs(epsg=3857)

      # Plot the scatter points
      fig, ax = plt.subplots(figsize=(6, 6))
      scatter = ax.scatter(gdf_flight_points.geometry.x, gdf_flight_points.geometry.y, c=altitudes, cmap='plasma', marker='.', s=5, alpha=0.7, vmin=0, vmax=750)
      cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal',fraction=0.046, pad=0.04) #, pad=0.01, aspect=50
      cbar.set_label('Altitude (m)')
      # Plot the circle points
      # ax.plot(gdf_circle.geometry.x, gdf_circle.geometry.y, color='red', linestyle='-', linewidth=1.5, alpha=0.7)

      # Add a basemap
      ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=14)

      # Plot settings
      minx, miny, maxx, maxy = gdf_flight_points.total_bounds
      # print(f"Bounds: {minx}, {miny}, {maxx}, {maxy}")
      # ax.set_xlim(minx, maxx)
      # ax.set_ylim(miny, maxy+10000)
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
      """
      Plot vertical rate (vertrate) vs barometric altitude for all flights on the same plot, without colorscale.
      """
      plt.figure(figsize=(10, 5))
      plotted_bada = set()
      all_baroaltitude = []
      all_vertrate = []
      for flight_path_name, flight_path in flight_paths.items():
            ac_type = flight_path_name.split("_")[2]
            ptf_filepath = f"BADA/{ac_type}__.ptf"
            ptf = read_ptf(ptf_filepath)
            ac_data = ptf['table']

            # Only keep one point every 15 seconds for plotting
            time_seconds = flight_path["time"] - flight_path["time"].iloc[0]
            mask = (time_seconds // 15).diff().fillna(1).astype(bool)
            sampled_baroaltitude = flight_path['baroaltitude'][mask]
            sampled_vertrate = flight_path["vertrate"][mask]
            all_baroaltitude.extend(sampled_baroaltitude)
            all_vertrate.extend(sampled_vertrate)

            # all_baroaltitude.extend(flight_path['baroaltitude'])
            # all_vertrate.extend(flight_path["vertrate"])

            # Only plot BADA curves once per aircraft type
            if ac_type not in plotted_bada:
                  plt.plot(ac_data.index * 100 * FT_TO_M, ac_data["ROCDhi_cl"] * FT_TO_M / 60, label="BADA High Load Climb Rate", c="red")
                  plt.plot(ac_data.index * 100 * FT_TO_M, ac_data["ROCDnom_cl"] * FT_TO_M / 60, label="BADA Nominal Load Climb Rate", c="limegreen")
                  plt.plot(ac_data.index * 100 * FT_TO_M, ac_data["ROCDlo_cl"] * FT_TO_M / 60, label="BADA Low Climb Load Rate", c="purple")
                  plt.plot(ac_data.index * 100 * FT_TO_M, ac_data["ROCDnom_des"] * FT_TO_M / -60, label="BADA Nominal Load Descent Rate", c="blue")
                  # Add idle descent rate curve (glide slope 17:1)
                  # Convert Vdes from knots to m/s before dividing by glide ratio
                  idle_descent_rate = (ac_data["Vdes"] * 0.514444) / 17  # Vdes in knots, convert to m/s, 17:1 glide slope
                  plt.plot(ac_data.index * 100 * FT_TO_M, -idle_descent_rate, label="17:1 Glide Slope", c="orange", linestyle="--")
                  plotted_bada.add(ac_type)

      # Plot the observed ROCD points with alpha=0.1, but add a proxy artist for the legend with alpha=1
      scatter = plt.scatter(
            all_baroaltitude,
            all_vertrate,
            s=8,
            marker='.',
            label=None,
            alpha=0.05
      )
      # Add a proxy artist for the legend with alpha=1
      proxy = Line2D([-10000], [-10000], marker='.', color='w', markerfacecolor='C0', markersize=8, alpha=1, linestyle='None', label="Observed ROCD")
      plt.gca().add_artist(proxy)

      # Add horizontal lines at 1 and -1 m/s for cruise cutoff
      plt.axhline(1, color='black', linestyle='--', linewidth=1.5, label='Cruise Cutoff')
      plt.axhline(-1, color='black', linestyle='--', linewidth=1.5)

      plt.xlabel("Barometric Altitude (m)")
      plt.ylabel("Vertical Rate (m/s)")
      # plt.title("Vertical Rate vs Barometric Altitude (All Flights)")
      plt.ylim(-30, 30)
      plt.xlim(0, 12500)
      plt.legend(fontsize=8)
      plt.savefig("vert_rate_vs_altitude.png", bbox_inches='tight', pad_inches=0.2, dpi=300)
      plt.show()

def plot_vert_rate_occurrences(flight_paths):
      plt.figure(figsize=(10, 5))
      for flight_path_name, flight_path in flight_paths.items():
            # Filter for altitudes between 5000 and 6000 meters (baroaltitude)
            mask = (flight_path["baroaltitude"] >= 5000) & (flight_path["baroaltitude"] <= 6000)
            filtered_vert_rate = flight_path.loc[mask, "vertrate"]
            if not filtered_vert_rate.empty:
                    plt.hist(filtered_vert_rate, bins=100, alpha=0.5, label=flight_path_name)
      plt.xlabel("Vertical Rate (m/s)")
      plt.ylabel("Count")
      plt.legend()
      plt.title("Vertical Rate Distribution (5000-6000m Baro Altitude)")
      plt.show()

def plot_lto_cutoff(flight_paths):
      plt.figure(figsize=(12, 6))
      ax = plt.axes(projection=ccrs.PlateCarree())

      # Remove cartopy features, use contextily basemap instead
      # ax.add_feature(cfeature.LAND)
      # ax.add_feature(cfeature.OCEAN)
      # ax.add_feature(cfeature.COASTLINE)
      # ax.add_feature(cfeature.BORDERS, linestyle=':')
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

      # Set plot limits centered around Heathrow Airport (EGLL)
      heathrow_lat = 51.4700
      heathrow_lon = -0.4543
      min_lon = heathrow_lon - 25 * 1.60934 / (111.320 * np.cos(np.radians(heathrow_lat)))
      max_lon = heathrow_lon + 25 * 1.60934 / (111.320 * np.cos(np.radians(heathrow_lat)))
      min_lat = heathrow_lat - 25 * 1.60934 / 110.574
      max_lat = heathrow_lat + 25 * 1.60934 / 110.574

      ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

      # Remove all points over 50 NM from Heathrow
      heathrow_lat = 51.4700
      heathrow_lon = -0.4543
      max_distance_nm = 50
      max_distance_m = max_distance_nm * 1852

      for flight_path_name in list(flight_paths.keys()):
            df = flight_paths[flight_path_name]
            distances = df.apply(
                  lambda row: geodesic((row['lat'], row['lon']), (heathrow_lat, heathrow_lon)).meters, axis=1
            )
            mask = distances <= max_distance_m
            filtered_df = df[mask].reset_index(drop=True)
            if filtered_df.empty:
                  del flight_paths[flight_path_name]
            else:
                  flight_paths[flight_path_name] = filtered_df

      # Plot the flight paths
      lto = []
      non_lto = []
      for flight_path_name, flight_path in flight_paths.items():
            heathrow_alt = AIRPORT_NAMES.loc["EGLL", "alt"]
            lto.append(flight_path[flight_path['geoaltitude'] < 3000*FT_TO_M + heathrow_alt])
            non_lto.append(flight_path[flight_path['geoaltitude'] >= 3000*FT_TO_M + heathrow_alt])

      # if non_lto:
      #       non_lto_df = pd.concat(non_lto, ignore_index=True)
      #       ax.plot(non_lto_df['lon'], non_lto_df['lat'], c='blue', alpha=0.5, label="Non-LTO")
      # if lto:
      #       lto_df = pd.concat(lto, ignore_index=True)
      #       ax.plot(lto_df['lon'], lto_df['lat'], c='red', alpha=0.5, label="LTO")

      if non_lto:
            non_lto_df = pd.concat(non_lto, ignore_index=True)
            ax.scatter(non_lto_df['lon'], non_lto_df['lat'], c='blue', marker=".", s=8, alpha=0.5, label="Non-LTO")
      if lto:
            lto_df = pd.concat(lto, ignore_index=True)
            ax.scatter(lto_df['lon'], lto_df['lat'], c='red', marker=".", s=8, alpha=0.5, label="LTO")

      # Add contextily basemap
      try:
            # Convert to Web Mercator for contextily

            # Create a bounding box polygon for the extent
            bbox = gpd.GeoDataFrame(
                  geometry=[Point(min_lon, min_lat).buffer(distance=0.01)],
                  crs="EPSG:4326"
            ).to_crs(epsg=3857).total_bounds

            # Set the extent in Web Mercator for contextily
            ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=10, crs=ccrs.PlateCarree())
      except Exception as e:
            print(f"Contextily basemap could not be added: {e}")

      plt.legend()
      plt.savefig("lto_cutoff.png", bbox_inches='tight', dpi=300, pad_inches=0)
      plt.show()

def plot_vertical_rate_time_histogram(flight_paths, bins=100):
      """
      Plot a histogram where the x-axis is vertical rate (m/s) and the y-axis is total time (seconds)
      spent operating at that vertical rate, summed across all flights.
      """
      all_vert_rates = []
      all_times = []
      for flight_path in flight_paths.values():
            # Calculate time deltas between points
            time_deltas = flight_path["time"].diff().fillna(0)
            vert_rates = flight_path["vertrate"]
            all_vert_rates.extend(vert_rates)
            all_times.extend(time_deltas)

      # Bin the vertical rates and sum the time spent in each bin
      hist, bin_edges = np.histogram(all_vert_rates, bins=bins, weights=all_times)
      bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

      plt.figure(figsize=(10, 5))
      plt.bar(bin_centers, hist, width=(bin_edges[1] - bin_edges[0]), align='center', alpha=0.7)
      plt.xlabel("Vertical Rate (m/s)")
      plt.ylabel("Total Time at Vertical Rate (s)")
      plt.yscale("log")
      plt.title("Time Spent at Each Vertical Rate")
      plt.show()



# flight_paths = load_flight_paths("ProcessedFlightData", include="A320", limit=1)

# plot_cartopy(flight_paths, color_by="index")

# flight_paths = load_flight_paths("Processed_test", include="A320", limit=1)

# plot_cartopy(flight_paths, color_by="index")

# plot_vert_rate_occurrences(flight_paths)

# flight_paths = load_flight_paths("ProcessedFlightData", limit=100)

# plot_lto_cutoff(flight_paths)

flight_paths = load_flight_paths("ProcessedFlightData", include="A320", limit=1000)
# flight_paths = load_flight_paths("ProcessedNoRDP", include="A320", limit=1000)

# plot_vertical_rate_time_histogram(flight_paths, bins=100)

vert_rate_vs_altitude(flight_paths)

# flight_paths = interpolate_great_circle(flight_paths)
# vel_mismatch_lat, vel_mismatch_lon = analyse_data(flight_paths)
# detail_plot(flight_paths)

# flight_paths = clean_alt_data(flight_paths)
# print(flight_paths.keys())
# plot_cartopy(flight_paths, color_by="geoaltitude")
# altitude_plot(flight_paths)
# vert_rate_plot(flight_paths)
# vert_rate_vs_altitude(flight_paths)
