"""
File to download wind data directly from NASA's data servers
"""
import earthaccess
import os
from datetime import datetime, timezone
import xarray as xr
import numpy as np
import glob


SAVE_DIR_ROOT = "./WindData"

def import_from_NASA():
    auth = earthaccess.login(persist=True)  # noqa: F841

    temporal = ("2024-12-25", "2025-01-01")
    bounding_box = (-180, -90, 180, 90)

    run_str = f"WindData_{datetime.strftime('%Y%m%d_%H%M%S')}"
    save_dir = os.path.join(SAVE_DIR_ROOT, run_str)

    results = earthaccess.search_data(
        doi="10.5067/SUOQESM06LPK",  # // cspell:disable-line
        temporal=temporal,
        bounding_box=bounding_box,
    )
    downloaded_files = earthaccess.download(results, local_path=save_dir)

    return downloaded_files

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
    dslist = []
    for fpath in files_in:
        print(fpath)
        ds = xr.open_dataset(fpath, drop_variables=["CLOUD", "DELP", "EPV", "O3", "OMEGA", "PHIS", "PS", "QI", "QL", "QV", "RH", "SLP"]) # Could add more here
        with xr.set_options(keep_attrs=True):
            ds = ds.isel(lev=slice(-nlayers, None))
        dslist.append(ds)
        ds.close()
    
    print('Concatenating...', end='')
    ds = xr.concat(dslist, 'time')
    print(' done')
    print('Processing...', end='')
    old_attrs = ds.attrs
    start_date = min([d.attrs['RangeBeginningDate'] for d in dslist])
    end_date = max([d.attrs['RangeEndingDate'] for d in dslist])
    with xr.set_options(keep_attrs=True):
        ds = ds.isel(lev=slice(-nlayers, None))
        # ds = ds.mean('time')
        ds = ds.assign(WS=np.sqrt(ds.U**2 + ds.V**2))
        ds.WS.attrs = {'long_name': 'wind_speed',
                       'standard_name': 'wind_speed',
                       'units': 'm s-1',}
        ds = ds.assign(WDIR=np.arctan2(ds.U, ds.V)*180/np.pi)
        ds.WDIR.attrs = {'long_name': 'wind_direction',
                         'standard_name': 'wind_direction',
                         'units': 'degrees clockwise from north',}
        ds = ds.drop(['U', 'V'])

    now = datetime.now(timezone.utc).strftime('%Y/%m/%d %H:%M:%S %Z')
    ds.attrs = {
        'Title': 'MERRA2 processed wind data,'
                 + ' files for use in openAVEM',
         'Contact': 'Flávio Quadros (f.quadros@tudelft.nl)',
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
         'Delta_Lon': old_attrs['LongitudeResolution'],
         'Delta_Lat': old_attrs['LatitudeResolution']
    }
    print(' done')
    
    print(f'Saving "{file_out}"...', end='')
    encoding = dict(zlib=True, shuffle=True, complevel=1,
                    chunksizes=[ds.time.size, 1, ds.lat.size, ds.lon.size])
    
    for var in ds.data_vars:
        print(f"Variable {var}: dims={ds[var].dims}, shape={ds[var].shape}")


    print([1, ds.lat.size, ds.lon.size])

    enc = {v:encoding for v in list(ds.data_vars.keys())}
    ds.to_netcdf(file_out, format='NETCDF4', engine='netcdf4',
                 encoding=enc)
    ds.close()
    print(' done')
    print(ds)
    
    return ds

if __name__ == '__main__':
    import_from_NASA()  
    basedir = os.path.join(os.path.dirname(__file__), 'WindData', 'Raw')
    year = '2024'
    for month in [f'{m:02}' for m in range(1, 12 + 1)]:
        output_fpath = os.path.join(os.path.dirname(__file__), 'met', f'wind_monthly_{year}{month}.nc4')
        search = os.path.join(basedir, f"MERRA2_400.tavg3_3d_asm_Nv.{year}{month}*.nc4")
        fpaths = glob.glob(search)
        if len(fpaths) > 0:
            ds = process_files(fpaths, output_fpath)
        else:
            print(f'Warning: no files found for "{search}"')