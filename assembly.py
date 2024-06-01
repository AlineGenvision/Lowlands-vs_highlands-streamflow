"""
Compiles the hydrological database required for modelling streamflow based on
meteorological variables, using era5 data retrieval script and hydrological
data handling functions from the apollo library.

@author: robertrouse
"""

# Import cdsapi and create a Client instance
import os
import paths
import xarray as xr
import pandas as pd
from apollo import era5 as er
from apollo import hydropoint as hp


### Specify meteorological variables and spatiotemporal ranges
area = ['60.00/-8.00/48.00/4.00']
yyyy = [str(y) for y in range(1979,2022,1)]
mm = [str(m) for m in range(1,13,1)]
dd = [str(d) for d in range(1,32,1)]
hh = [str(t).zfill(2) + ':00' for t in range(0, 24, 1)]
met = ['total_precipitation','temperature','u_component_of_wind',
       'v_component_of_wind','relative_humidity',
       'volumetric_soil_water_layer_1','volumetric_soil_water_layer_2',
       'volumetric_soil_water_layer_3','volumetric_soil_water_layer_4', 'snowmelt']


### Download meteorological variable sets from Copernicus Data Store
for yy in yyyy:
    filename = paths.WEATHER_UK + '/Rainfall/Rainfall_' + str(yy)
    rain_query = er.query(filename, 'reanalysis-era5-single-levels', met[0],
                          area, yy, mm, dd, hh)

    # Save the hourly precipitation data and aggregate daily (from midnight to midnight)
    if not os.path.exists(filename + '.nc'):
        print('downloading ', filename)
        rain_data = er.era5(rain_query).download()

    er.aggregate_mean(str(rain_query['file_stem']) + '.nc',
                      str(rain_query['file_stem']) + '_aggregated.nc')

    # Shift the daily aggregation (from 9am to 9am)
    if not os.path.exists(filename + '_9to9.nc'):
        print('shifting', filename)
        er.shift_time_9am_to_9am(str(rain_query['file_stem']) + '.nc',
                                 int(yy),
                                 str(rain_query['file_stem']) + '_9to9.nc')
        er.aggregate_mean(str(rain_query['file_stem']) + '_9to9.nc',
                          str(rain_query['file_stem']) + '_aggregated_9to9.nc',
                          shift=9)

if not os.path.exists(paths.RAINFALL_UK):
    full_rain_data = xr.open_mfdataset(paths.WEATHER_UK + '/Rainfall/Rainfall_*_aggregated.nc', concat_dim='time',
                                       combine='nested')
    full_rain_data.to_netcdf(path=paths.RAINFALL_UK)

if not os.path.exists(paths.RAINFALL_UK_SHIFTED):
    full_shifted_rain_data = xr.open_mfdataset(paths.WEATHER_UK + '/Rainfall/Rainfall_*_aggregated_9to9.nc',
                                               concat_dim='time', combine='nested')
    full_shifted_rain_data.to_netcdf(path=paths.RAINFALL_UK_SHIFTED)

for yy in yyyy:
    filename = paths.WEATHER_UK + '/Pressure/Pressure_' + str(yy)

    if not os.path.exists(filename + '1000hPa.nc'):
        print("downloading ", filename)
        pressure_query = er.query(filename, 'reanalysis-era5-pressure-levels', met[1:5],
                          area, yy, mm, dd, '12:00', ['1000'])
        pressure_data = er.era5(pressure_query).download()

full_pressure_data = xr.open_mfdataset(paths.WEATHER_UK + '/Pressure/Pressure_*.nc', concat_dim='time', combine='nested')
if not os.path.exists(paths.PRESSURE_UK):
    full_pressure_data.to_netcdf(path=paths.PRESSURE_UK)


for yy in yyyy:
    filename = paths.SURFACE_UK + '/Soil_Moisture_' + str(yy)

    if not os.path.exists(filename + '.nc'):
        print("downloading ", filename)
        soil_moisture_query = er.query(filename, 'reanalysis-era5-land', met[5:-1], area, yy, mm, dd, '12:00')
        soil_moisture_data = er.era5(soil_moisture_query).download()

full_soil_moisture_data = xr.open_mfdataset(paths.SURFACE_UK + '/Soil_Moisture_*.nc', concat_dim='time', combine='nested')
if not os.path.exists(paths.SOIL_MOISTURE_UK):
    full_soil_moisture_data.to_netcdf(path=paths.SOIL_MOISTURE_UK)


for yy in yyyy:
    filename = paths.WEATHER_UK + '/Snowmelt/Snowmelt_' + str(yy)

    if not os.path.exists(filename + '.nc'):
        print("downloading ", filename)
        snowmelt_query = er.query(filename, 'reanalysis-era5-land', met[-1:], area, yy, mm, dd, '12:00')
        snowmelt_data = er.era5(snowmelt_query).download()

full_snowmelt_data = xr.open_mfdataset(paths.WEATHER_UK + '/Snowmelt/Snowmelt_*.nc', concat_dim='time', combine='nested')
if not os.path.exists(paths.SNOWMELT_UK):
    full_snowmelt_data.to_netcdf(path=paths.SNOWMELT_UK)


### Produce lumped regression files per catchment
domain_weather = xr.open_mfdataset([paths.RAINFALL_UK,
                                    paths.PRESSURE_UK])
surface_data = xr.open_dataset(paths.SOIL_MOISTURE_UK)
snow_data = xr.open_dataset(paths.SNOWMELT_UK)
domain_rain = xr.open_mfdataset([paths.RAINFALL_HOURLY_UK])
db = pd.read_csv(paths.DATA + '/Catchments_Fens.csv')
for i in range(len(db)):
    print('start with ', i)

    db_path = paths.CATCHMENT_BASINS + '/' + str(db.loc[i][0])

    test = hp.hydrobase(db.loc[i][0],
                        db_path + '/' + db.loc[i][3],
                        db_path + '/' + db.loc[i][4])
    cache = test.output_file(domain_weather, surface_data, snow_data,28, out_fp=paths.CATCHMENT_BASINS)
    #more_cache = test.output_hourly_rain_file(domain_rain, 24, out_fp=paths.CATCHMENT_BASINS)
