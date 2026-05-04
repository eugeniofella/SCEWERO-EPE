import xarray as xr
import numpy as np
import pandas as pd

def timeseries_processing(ts_loc_seas, dry_threshold, accumulation_time, months):

    # set dry events as nan and set the rolling window
    ts_loc_seas_acc = ts_loc_seas.where(ts_loc_seas >= dry_threshold).rolling(time=accumulation_time, center=True).sum()

    # remove first and last days of the season accordingly to the accumulation_time
    for i in range(accumulation_time//2):

        # remove days
        ts_loc_seas_acc = ts_loc_seas_acc.sel(time=~((ts_loc_seas_acc["time"].dt.month == months[0]) & (ts_loc_seas_acc["time"].dt.day == 1+i)))
        ts_loc_seas_acc = ts_loc_seas_acc.sel(time=~((ts_loc_seas_acc["time"].dt.month == months[-1]) & (ts_loc_seas_acc["time"].dt.day == ts_loc_seas_acc.time.isel(time=-1).dt.day.values - i)))

    return ts_loc_seas_acc


def overlapping_season(n_over_season, era5_TP_loc, q_era5_TP_loc_seas_acc, NQp_era5_TP_loc_seas_acc, dry_threshold, accumulation_time, months):

    # define the shift we are interested in
    day_shift_list = np.linspace(n_over_season//2*-7, n_over_season//2*7, n_over_season, dtype = int)
    
    NQp_era5_TP_loc_seas_acc_list = []
    
    
    for day_shift in day_shift_list:
    
        # shift the timeseries
        era5_TP_loc_shifted = era5_TP_loc.shift(time=day_shift)# note: a positive offset indicates a backward shift in time
    
        # select the season of interest
        era5_TP_loc_seas_shifted = era5_TP_loc_shifted.where(era5_TP_loc_shifted['time'].dt.month.isin(months), drop=True)
    
        # call the function
        era5_TP_loc_seas_acc_shifted = timeseries_processing(era5_TP_loc_seas_shifted, dry_threshold, accumulation_time, months)
    
        # define the NQp (i.e. the number of days in season, above the Qth percentile threshold)
        NQp_era5_TP_loc_seas_acc_shifted = (era5_TP_loc_seas_acc_shifted > q_era5_TP_loc_seas_acc).groupby("time.year").sum()
    
        # shift the predictors date accordingly
        new_dates = (pd.to_datetime(NQp_era5_TP_loc_seas_acc.predictors_date.values) + pd.Timedelta(days=(-day_shift))).strftime('%Y-%m-%d').values.astype('U10')
        
        # assign the corresponding predictors date
        NQp_era5_TP_loc_seas_acc_shifted = NQp_era5_TP_loc_seas_acc_shifted.assign_coords(predictors_date=('year', new_dates))
    
        # add to the list of shifted NQp dataarrays
        NQp_era5_TP_loc_seas_acc_list.append(NQp_era5_TP_loc_seas_acc_shifted.set_index(year='predictors_date').rename({'year': 'predictors_date'}))
    
    # merge all the shifted variables
    NQp_era5_TP_loc_seas_acc_over = xr.merge(NQp_era5_TP_loc_seas_acc_list)

    return NQp_era5_TP_loc_seas_acc_over
    
        

