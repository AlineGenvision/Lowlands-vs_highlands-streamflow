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
       'volumetric_soil_water_layer_3','volumetric_soil_water_layer_4',]


### Download meteorological variable sets from Copernicus Data Store
for yy in yyyy:
    filename = paths.WEATHER_UK + '/Rainfall_' + str(yy)

    if not os.path.exists(filename + '.nc'):
        print('downloading ', filename)
        rain_query = er.query(filename, 'reanalysis-era5-single-levels', met[0],
                          area, yy, mm, dd, hh)
        rain_data = er.era5(rain_query).download()
        er.aggregate_mean(str(rain_query['file_stem']) + '.nc',
                        str(rain_query['file_stem']) + '_aggregated.nc')

full_rain_data = xr.open_mfdataset(paths.WEATHER_UK + '/Rainfall_*_aggregated.nc', concat_dim='time', combine='nested')
if not os.path.exists(paths.RAINFALL_UK):
    full_rain_data.to_netcdf(path=paths.RAINFALL_UK)

for yy in yyyy:
    filename = paths.WEATHER_UK + '/Pressure_' + str(yy)

    if not os.path.exists(filename + '1000hPa.nc'):
        print("downloading ", filename)
        pressure_query = er.query(filename, 'reanalysis-era5-pressure-levels', met[1:5],
                          area, yy, mm, dd, '12:00', ['1000'])
        pressure_data = er.era5(pressure_query).download()

full_pressure_data = xr.open_mfdataset(paths.WEATHER_UK + '/Pressure_*.nc', concat_dim='time', combine='nested')
if not os.path.exists(paths.PRESSURE_UK):
    full_pressure_data.to_netcdf(path=paths.PRESSURE_UK)

for yy in yyyy:
    filename = paths.SURFACE_UK + '/Soil_Moisture_' + str(yy)

    if not os.path.exists(filename + '.nc'):
        print("downloading ", filename)
        soil_moisture_query = er.query(filename, 'reanalysis-era5-land', met[5:], area, yy, mm, dd, '12:00')
        soil_moisture_data = er.era5(soil_moisture_query).download()

full_soil_moisture_data = xr.open_mfdataset(paths.SURFACE_UK + '/Soil_Moisture_*.nc', concat_dim='time', combine='nested')
if not os.path.exists(paths.SOIL_MOISTURE_UK):
    full_soil_moisture_data.to_netcdf(path=paths.SOIL_MOISTURE_UK)

#pressure_query = er.query('Pressure','reanalysis-era5-pressure-levels',
#                          met[1:5], area, yyyy, mm, dd, '12:00', ['1000'])
#pressure_data = er.era5(pressure_query).download()

#soil_query = er.query('Soil_Moisture','reanalysis-era5-land', met[5:],
#                      area, yyyy, mm, dd, '12:00')
#soil_data = er.era5(soil_query).download()


### Produce lumped regression files per catchment
domain_weather = xr.open_mfdataset([paths.RAINFALL_UK,
                                    paths.PRESSURE_UK])
surface_data = xr.open_dataset(paths.SOIL_MOISTURE_UK)
db = pd.read_csv(paths.DATA + '/Catchment_Database.csv')
for i in range(len(db)):
    db_path = paths.CATCHMENT_BASINS + '/' + str(db.loc[i][0])
    test = hp.hydrobase(db.loc[i][0],
                        db_path + '/' + db.loc[i][3],
                        db_path + '/' + db.loc[i][4])
    cache = test.output_file(domain_weather, surface_data, 28, out_fp=paths.CATCHMENT_BASINS)
