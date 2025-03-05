"""
Script to prepare wind data for openAVEM from MERRA-2 data
"""

import os
import xarray as xr
import numpy as np
import glob
from datetime import datetime, timezone
import shutil
import requests
from bs4 import BeautifulSoup
import urllib.parse


BASEDIR = r'RawWindData/'


def process_files(files_in, file_out, nlayers=32):
    """
    Read files_in, concatenate, and save file_out with wind speed and direction

    Parameters
    ----------
    files_in : list of str
        Paths to the netCDF files containing MERRA-2 wind data.
    file_out : str
        Path to output file.
    nlayers : int or None, optional
        Number of vertical layers to keep. If None, all layers are kept.
        The default is 32.

    Returns
    -------
    ds : xr.Dataset
        Dataset containing wind speed [m/s] and wind direction [degrees
        clockwise from north].

    """
    H_EDGES = np.array(
        [-6, 123, 254, 387, 521, 657, 795, 934, 1075, 1218, 1363, 1510, 1659,
         1860, 2118, 2382, 2654, 2932, 3219, 3665, 4132, 4623, 5142, 5692,
         6277, 6905, 7582, 8320, 9409, 10504, 11578, 12633]
    )
    dslist = []
    for fpath in files_in:
        # print(fpath)
        ds = xr.open_dataset(fpath, drop_variables=['DTRAIN', 'OMEGA', 'RH'])
        with xr.set_options(keep_attrs=True):
            ds = ds.sel(lev=slice(None, nlayers))
        dslist.append(ds)
        ds.close()

    print('Concatenating...', end='')
    ds = xr.concat(dslist, 'time')
    print(' done')
    print('Processing...', end='')
    old_attrs = ds.attrs
    start_date = min([d.attrs['Start_Date'] for d in dslist])
    end_date = max([d.attrs['End_Date'] for d in dslist])
    with xr.set_options(keep_attrs=True):
        ds = ds.sel(lev=slice(None, nlayers))
        ds = ds.assign(WS=np.sqrt(ds.U**2 + ds.V**2))
        ds.WS.attrs = {'long_name': 'wind_speed',
                       'standard_name': 'wind_speed',
                       'units': 'm s-1',}
        ds = ds.assign(WDIR=np.arctan2(ds.U, ds.V)*180/np.pi)
        ds.WDIR.attrs = {'long_name': 'wind_direction',
                         'standard_name': 'wind_direction',
                         'units': 'degrees clockwise from north',}
        ds = ds.drop_vars(['U', 'V'])

    # Add altitude coordinate
    ds.coords['h_edge'] = ('lev', H_EDGES,
                           {'long_name': ('altitude over sea level'
                                          + ' at lower edge of grid box'),
                            'units': 'm'})

    now = datetime.now(timezone.utc).strftime('%Y/%m/%d %H:%M:%S %Z')
    ds.attrs = {
        'Title': 'MERRA2 concatenated wind data, processed from GEOS-Chem input'
                 + ' files for use in openAVEM',
         'Contact': 'Maximilian Howard (mdeh2@cam.ac.uk)',
         'References': '',
         'Filename': os.path.basename(file_out),
         'History': 'File generated on: ' + now,
         'ProductionDateTime': 'File generated on: ' + now,
         'ModificationDateTime': 'File generated on: ' + now,
         'Format': 'NetCDF-4',
         'SpatialCoverage': 'global',
         'Version': 'MERRA2',
         'VersionID': old_attrs['VersionID'],
         'Nlayers': nlayers,
         'Start_Date': start_date,
         'Start_Time': '00:00:00.0',
         'End_Date': end_date,
         'End_Time': '23:59:59.99999',
         'Delta_Lon': old_attrs['Delta_Lon'],
         'Delta_Lat': old_attrs['Delta_Lat']
    }
    print(' done')

    print(f'Saving "{file_out}"...', end='')
    encoding = dict(zlib=True, shuffle=True, complevel=1,
                    chunksizes=[ds.time.size, 1, ds.lat.size, ds.lon.size])
    enc = {v:encoding for v in list(ds.data_vars.keys())}
    ds.to_netcdf(file_out, format='NETCDF4', engine='netcdf4',
                 encoding=enc)
    ds.close()
    print(' done')
    print(ds)

    return ds


def download_files(year, month):
    """
    Download MERRA-2 wind data files for a given year and month.
    """

    base_url = f"http://geoschemdata.wustl.edu/ExtData/GEOS_0.5x0.625/MERRA2/{year}/{month}/"
    save_dir = os.path.join(BASEDIR, year, month)

    # Check download directory before starting
    if os.path.exists(save_dir):
        user_input = input(f"The directory {save_dir} already exists, rename/remove it? (yes/no/new_name): ")
        if user_input.lower() == 'yes':
            shutil.rmtree(save_dir)
            os.makedirs(save_dir)
        elif user_input.lower() == 'no':
            print("Cannot continue without removing the existing directory, skipping download.")
            return
        else:
            renamed = os.path.join(BASEDIR, user_input)
            os.rename(save_dir, renamed)
            print(f"Renamed {save_dir} to {renamed}")
    else:
        os.makedirs(save_dir)

    # Create a .gitignore file if it doesn't exist
    gitignore_path = os.path.join(BASEDIR, '.gitignore')
    if not os.path.exists(gitignore_path):
        with open(gitignore_path, 'w') as gitignore_file:
            gitignore_file.write('*\n')

    # Send a GET request to fetch the directory listing
    response = requests.get(base_url)
    if response.status_code != 200:
        print("Failed to retrieve the webpage. Status code:", response.status_code)
        exit(1)

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the links to the files
    all_links = soup.find_all('a')
    file_urls = []
    for link in all_links:
        href = link.get('href')
        if href and "A3dyn" in href and href.endswith(".nc4"):
            file_url = urllib.parse.urljoin(base_url, href)
            file_urls.append(file_url)

    print(f"{datetime.now().strftime('%H:%M:%S')}: Downloading {len(file_urls)} MERRA-2 wind data files for {year}/{month}")

    # Download the files
    for idx,file_url in enumerate(file_urls):
        file_response = requests.get(file_url, stream=True)
        file_num_string: str = f"{idx+1} of {len(file_urls)}"
        if file_response.status_code == 200:
            start_time = datetime.now()
            file_path = os.path.join(save_dir, href)
            total_size = int(file_response.headers.get('content-length', 0))
            block_size = 8192  # 8 Kibibytes
            wrote = 0
            with open(file_path, 'wb') as f:
                for chunk in file_response.iter_content(chunk_size=block_size):
                    if chunk:  # filter out keep-alive chunks
                        f.write(chunk)
                        wrote += len(chunk)
                        done = int(50 * wrote / total_size)
                        print(f"\rDownloading file {file_num_string}. File size: {total_size / (1024 * 1024):.2f} MB [{'=' * done}{' ' * (50-done)}] {wrote / total_size:.2%}", end='')
            seconds_taken = int(round((datetime.now()-start_time).total_seconds(),0))
            print("\r\033[K", end='')
            print(f"File {file_num_string}. Downloaded {total_size / (1024 * 1024):.2f} MB in {seconds_taken}s. Saved to: {file_path}")
        else:
            raise Exception(f"Failed to download {href}. Status code: {file_response.status_code}")
    return save_dir


if __name__ == '__main__':
    year = '2024'

    # Download files
    for month in [f'{m:02}' for m in range(1, 12 + 1)]:
        save_path = download_files(year, month)

    # Process files
    for month in [f'{m:02}' for m in range(1, 12 + 1)]:
        output_fpath = r'./met/wind_monthly_' + year + month + '.nc4'
        search = f'{BASEDIR}{year}/{month}/' + r'*.A3dyn.*.nc4'
        fpaths = glob.glob(search)
        if len(fpaths) > 0:
            print(f'Processing {len(fpaths)} files for "{year}/{month}".')
            ds = process_files(fpaths, output_fpath)
        else:
            print(f'Warning: no files found for "{search}"')