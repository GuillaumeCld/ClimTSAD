import xarray as xr
import numpy as np

pressure_path = "/home/guillaume/Devel/Spatiotemporal-anomalies/Data/era5_msl_daily_eu.nc"
preci_path = "/home/guillaume/Devel/burst-research/Data/ERA5_land/tp_daily_sum_1950_2023.nc"
temperature_path = "/home/guillaume/Devel/burst-research/Data/ERA5_land/t2m_daily_avg_1950_2023.nc"

my_lat = 44.83
my_lon = -0.56

# extract a single time series and save it to a npy file
pressure_ds = xr.open_dataset(pressure_path)
# select the time series for a specific location (e.g., latitude=50, longitude=10)
pressure_time_series = pressure_ds['msl'].sel(lat=my_lat, lon=my_lon, method='nearest')
pressure_time_series_values = pressure_time_series.values
print(f"Pressure time series shape: {pressure_time_series_values.shape}, first 5 values: {pressure_time_series_values[:5]}")
print(f"Pressure start date: {pressure_time_series['time'].values[0]}, end date: {pressure_time_series['time'].values[-1]}")
np.save('data/pressure_time_series.npy', pressure_time_series_values)    

preci_ds = xr.open_dataset(preci_path)
# select the time series for a specific location (e.g., latitude=50, longitude=10)
preci_time_series = preci_ds['tp'].sel(latitude=my_lat, longitude=my_lon, method='nearest')
preci_time_series_values = preci_time_series.values
print(f"Precipitation time series shape: {preci_time_series_values.shape}, first 5 values: {preci_time_series_values[:5]}")
print(f"Precipitation start date: {preci_time_series['time'].values[0]}, end date: {preci_time_series['time'].values[-1]}")
np.save('data/preci_time_series.npy', preci_time_series_values)


temperature_ds = xr.open_dataset(temperature_path)
# select the time series for a specific location (e.g., latitude=50, longitude=10)
temperature_time_series = temperature_ds['t2m'].sel(latitude=my_lat, longitude=my_lon, method='nearest')
temperature_time_series_values = temperature_time_series.values
print(f"Temperature time series shape: {temperature_time_series_values.shape}, first 5 values: {temperature_time_series_values[:5]}")
print(f"Temperature start date: {temperature_time_series['time'].values[0]}, end date: {temperature_time_series['time'].values[-1]}")
np.save('data/temperature_time_series.npy', temperature_time_series_values) 
