import os
import paths
import xarray as xr
import pandas as pd
from apollo import era5 as er
from apollo import era5land as erland
from apollo import hydropoint as hp
from train_model import load_data


### Specify meteorological variables and spatiotemporal ranges
area = ['60.00/-8.00/48.00/4.00']
yyyy = [str(y) for y in range(1979,2022,1)]
mm = [str(m) for m in range(1,13,1)]
dd = [str(d) for d in range(1,32,1)]
hh = [str(t).zfill(2) + ':00' for t in range(0, 24, 1)]
met = ['total_precipitation','snowmelt', 'temperature','u_component_of_wind',
       'v_component_of_wind','relative_humidity',
       'volumetric_soil_water_layer_1','volumetric_soil_water_layer_2',
       'volumetric_soil_water_layer_3','volumetric_soil_water_layer_4']

## Download Rainfall and snowmelt from ERA5-land (higher resolution)
for yy in yyyy:

    filename = paths.WEATHER_UK + '/Rainfall_0.1/Rainfall_' + str(yy)
    rain_query = er.query(filename, 'reanalysis-era5-land', met[:2], area, yy, mm, dd, hh)

    for m in mm:
        filename_month = paths.WEATHER_UK + '/Rainfall_0.1/Rainfall_' + str(yy) + '_' + str(m)
        rain_query_month = er.query(filename_month, 'reanalysis-era5-land', met[:2],
                                    area, yy, m, dd, hh)

        if not os.path.exists(filename_month + '.nc'):
            print('downloading ', filename_month)
            rain_data = er.era5(rain_query_month).download()

    if not os.path.exists(filename + '.nc'):
        rain_data = xr.open_mfdataset(paths.WEATHER_UK + f"/Rainfall_0.1/Rainfall_{str(yy)}*.nc",
                                      concat_dim='time', combine='nested')
        rain_data = rain_data.sortby('time')
        rain_data.to_netcdf(path=(filename + '.nc'))

    # USE THE ERA5-LAND method to aggregate the mean!!
    if not os.path.exists(str(rain_query['file_stem']) + '_aggregated.nc'):
        erland.aggregate_mean(str(rain_query['file_stem']) + '.nc',
                          str(rain_query['file_stem']) + '_aggregated.nc')

    # USE THE ERA5-LAND method to aggregate the mean from 9 to 9
    if not os.path.exists(filename + '_aggregated_9to9.nc'):
        print('shifting again', filename)
        erland.aggregate_mean_shift(str(rain_query['file_stem']) + '.nc',
                                str(rain_query['file_stem']) + '_aggregated_9to9.nc',
                                year=int(yy),
                                shift=9)

# Combine the daily midnight-midnight precipitation (High Res)
if not os.path.exists(paths.RAINFALL_UK_HR + '__'):
    full_rain_data = xr.open_mfdataset(paths.WEATHER_UK + '/Rainfall_0.1/Rainfall_*_aggregated.nc', concat_dim='time',
                                       combine='nested')
    full_rain_data = full_rain_data.sortby('latitude', ascending=False)
    full_rain_data.to_netcdf(path=paths.RAINFALL_UK_HR)

# Combine the daily 9 to 9 precipiptation (High Res)
if not os.path.exists(paths.RAINFALL_UK_SHIFTED_HR + '__'):
    full_shifted_rain_data = xr.open_mfdataset(paths.WEATHER_UK + '/Rainfall_0.1/Rainfall_*_aggregated_9to9.nc',
                                               concat_dim='time', combine='nested')
    full_shifted_rain_data = full_shifted_rain_data.sortby('latitude', ascending=False)
    full_shifted_rain_data.to_netcdf(path=paths.RAINFALL_UK_SHIFTED_HR)

'''
# Chunking is important for the HR dataset (initial chunking + rechunking)
domain_weather_HR = xr.open_mfdataset([paths.RAINFALL_UK_SHIFTED_HR, paths.PRESSURE_UK],
                                      chunks={'time': 10, 'latitude': 50, 'longitude': 50})
domain_weather_HR = domain_weather_HR.chunk({'time': -1, 'latitude': -1, 'longitude': -1})

# Interpolate the missing HR (NaN) values for snow and precipitation
interpolated_weather_HR = domain_weather_HR.astype('float32')
for var in ['tp', 'smlt']:
    for dim in ['time', 'latitude', 'longitude']:
        interpolated_weather_HR[var] = interpolated_weather_HR[var].interpolate_na(dim=dim, method='linear')
'''

domain_weather_HR = xr.open_dataset(paths.RAINFALL_UK_SHIFTED_HR)

db = pd.read_csv(paths.DATA + '/Catchments_Database.csv')

EXT = '_9to9'
OUT_FP = paths.CATCHMENT_BASINS

for i in range(len(db)):
    print('start with ', i)

    db_path = paths.CATCHMENT_BASINS + '/' + str(db.iloc[i,0])

    test = hp.hydrobase(db.iloc[i,0],
                        db_path + '/' + db.iloc[i,3],
                        db_path + '/' + db.iloc[i,4])

    # Linear interpolation high resolution data
    '''
    cache_HR = test.output_era5land_file(interpolated_weather_HR, surface_data, 28,
                                  out_fp=OUT_FP,
                                  ext=EXT + '_HR',
                                  interpolation_method='linear',
                                  reload=False)'''

    # Use era5 files as reference to alterate the precipitation with both NRFA and surface interpolated values
    rain_columns = (['Rain'] + ['Rain-' + f'{d + 1}' for d in range(27)] +
                    ['Rain_28_Mu', 'Rain_90_Mu', 'Rain_180_Mu'])
    ref_path = (paths.CATCHMENT_BASINS + '/' + str(db.iloc[i,0]) + '/' +
               str(db.iloc[i,0]) + f"_lumped{EXT}_linear.csv")
    rf_ref = load_data.load_data(ref_path, verbose=False).drop(columns=rain_columns)


    # Surface interpolation high resolution data
    surf_interp_cache_HR = test.output_surface_interpolated_file(domain_rain=domain_weather_HR,
                                                              rf_ref=rf_ref,
                                                              out_fp=OUT_FP,
                                                              ext=EXT + '_HR',
                                                              resolution=0.1,
                                                              multiplicator=1000*24,
                                                              reload=True)
