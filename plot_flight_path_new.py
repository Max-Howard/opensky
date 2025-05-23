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

def plt_lateral_inefficiency(flight_paths):
      """
      Calculate lateral inefficiency for each flight path and plot histogram weighted by total great circle distance.
      """
      # Calculate lateral inefficiency and collect flight distances
      lateral_inefficiencies_adsb = []
      lateral_inefficiencies_openavem = []
      great_circle_distances = []
      openavem_distances = []
      adsb_distances = []
      for flight_path_name, flight_path in flight_paths.items():
            origin, destination, typecode, icao24, flight_number = flight_path_name.strip(".csv").split("_")

            if origin == destination:
                  continue

            origin_lat = AIRPORT_NAMES.loc[origin, "lat"]
            origin_lon = AIRPORT_NAMES.loc[origin, "lon"]

            destination_lat = AIRPORT_NAMES.loc[destination, "lat"]
            destination_lon = AIRPORT_NAMES.loc[destination, "lon"]

            great_circle_distance = geodesic((origin_lat, origin_lon), (destination_lat, destination_lon)).meters
            adsb_distance = flight_path['dist'].max()

            # Calculate distance from origin airport to first datapoint
            start_point = (flight_path['lat'].iloc[0], flight_path['lon'].iloc[0])
            origin_point = (origin_lat, origin_lon)
            start_gap = geodesic(origin_point, start_point).meters

            # Calculate distance from last datapoint to destination airport
            end_point = (flight_path['lat'].iloc[-1], flight_path['lon'].iloc[-1])
            destination_point = (destination_lat, destination_lon)
            end_gap = geodesic(end_point, destination_point).meters

            adsb_distance += start_gap + end_gap
            openavem_distance = great_circle_distance*1.0387 + 40.5 * 1.852 * 1000

            inefficiency = (adsb_distance - great_circle_distance) / great_circle_distance
            lateral_inefficiencies_adsb.append(inefficiency * 100)
            lateral_inefficiencies_openavem.append((openavem_distance - great_circle_distance) / openavem_distance * 100)
            great_circle_distances.append(great_circle_distance / 1000)  # Convert to kilometers
            openavem_distances.append(openavem_distance/1000)
            adsb_distances.append(adsb_distance / 1000)  # Convert to kilometers

      print(sum(great_circle_distances))
      print(sum(openavem_distances))
      print(sum(adsb_distances))

      # Convert to numpy arrays for easier masking
      lateral_inefficiencies_adsb = np.array(lateral_inefficiencies_adsb)
      lateral_inefficiencies_openavem = np.array(lateral_inefficiencies_openavem)
      great_circle_distances = np.array(great_circle_distances)

      # Mask out flights with less than 1000 km great circle distance
      mask = great_circle_distances >= 1
      ineff_adsb = lateral_inefficiencies_adsb[mask]
      ineff_openavem = lateral_inefficiencies_openavem[mask]
      distances = great_circle_distances[mask]

      # Determine common x-axis range
      min_x = min(ineff_adsb.min(), ineff_openavem.min())
      max_x = max(ineff_adsb.max(), ineff_openavem.max())

      # min_x = 0
      max_x = 50

      # Plot histograms of inefficiency, weighted by great circle distance, side by side
      fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)

      axes[0].hist(
            ineff_openavem,
            bins=100,
            weights=distances,
            color='salmon',
            edgecolor='black',
            range=(min_x, max_x)
      )
      axes[0].set_xlabel("Lateral Inefficiency (%)")
      axes[0].set_ylabel("Total Great Circle Distance (km)")
      axes[0].set_yscale("log")
      axes[0].set_title("Stock openAVEM Assumptions")
      axes[0].set_xlim(min_x, max_x)

      axes[1].hist(
            ineff_adsb,
            bins=100,
            weights=distances,
            color='skyblue',
            edgecolor='black',
            range=(min_x, max_x)
      )
      axes[1].set_xlabel("Lateral Inefficiency (%)")
      axes[1].set_title("From ADS-B Tracks")
      axes[1].set_xlim(min_x, max_x)

      plt.tight_layout()
      plt.show()


            


# flight_paths = load_flight_paths("ProcessedFlightData", include="A320", limit=1)

# plot_cartopy(flight_paths, color_by="index")

# flight_paths = load_flight_paths("Processed_test", include="A320", limit=1)

# plot_cartopy(flight_paths, color_by="index")

# plot_vert_rate_occurrences(flight_paths)

# flight_paths = load_flight_paths("ProcessedFlightData", limit=100)

# plot_lto_cutoff(flight_paths)

# flight_paths = load_flight_paths("ProcessedFlightData", include="A320", limit=1000)
# # flight_paths = load_flight_paths("ProcessedNoRDP", include="A320", limit=1000)

# # plot_vertical_rate_time_histogram(flight_paths, bins=100)

# vert_rate_vs_altitude(flight_paths)

# flight_paths = interpolate_great_circle(flight_paths)
# vel_mismatch_lat, vel_mismatch_lon = analyse_data(flight_paths)
# detail_plot(flight_paths)

# flight_paths = clean_alt_data(flight_paths)
# print(flight_paths.keys())
# plot_cartopy(flight_paths, color_by="geoaltitude")
# altitude_plot(flight_paths)
# vert_rate_plot(flight_paths)
# vert_rate_vs_altitude(flight_paths)


flight_paths = load_flight_paths("ProcessedFlightData", include="A320")
plt_lateral_inefficiency(flight_paths)