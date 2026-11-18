import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import datetime as dt
import zarr
from datetime import datetime,timedelta,date
from pathlib import Path
from dateutil.relativedelta import relativedelta
#import urllib.request, urllib.error
#import contextlib
#import socket
#import requests
import json

from helpers import generate_daily,_effect

def get_cumsum(dates, precip, input_end_date, max_ts, missing='cumulative'):
    '''
    Calculates the reverse cumulative precipitation totals ending on a specified date.

    Inputs:
    dates: numpy array of datetime64 objects
    precip: numpy array of precipitation floats (missing values as np.nan)
    input_end_date: standard python datetime object
    max_ts: integer, maximum time steps to go backward
    missing: 'cumulative' or 'instance'

    Outputs:
    spi, spi_eff, spi_runoff (numpy arrays), cmfr (numpy array), end_dt (datetime object)
    '''
    # Safely align the target datetime object with the datetime64 array
    target_dt = np.datetime64(input_end_date, 'D')
    dates_days = dates.astype('datetime64[D]')

    if target_dt in dates_days:
        edloc = np.where(dates_days == target_dt)[0][0]
    else:
        edloc = len(dates) - 1
        target_dt = dates_days[edloc]
        print(f"{input_end_date} does not exist, using most recent date: {target_dt.item()}")

    # Determine the start index for the requested sequence
    sloc = max(edloc - max_ts, 0)

    # To calculate the 6-day rolling effect for the first day, we need up to 5 prior days
    pad_start = max(sloc - 5, 0)
    data_block = precip[pad_start : edloc + 1]

    # If we are too close to the start of the array to get 5 prior days, pad with NaNs
    pad_len = 6 - (sloc - pad_start + 1)
    if pad_len > 0:
        data_block = np.concatenate([np.full(pad_len, np.nan), data_block])

    # Create rolling 6-day windows (shape: N, 6) and apply the effect function
    windows = sliding_window_view(data_block, 6)
    eff_q = np.array([_effect(w) for w in windows])

    # Extract the base arrays and reverse them so the most recent day is index 0
    raw_p = windows[:, -1][::-1]
    peff = eff_q[:, 0][::-1]
    prunoff = eff_q[:, 1][::-1]

    # Calculate cumulative sums (Note: np.cumsum propagates NaNs.
    # If you prefer NaNs to be treated as 0 in the sum, use np.nancumsum instead)
    spi = np.cumsum(raw_p)
    spi_eff = np.cumsum(peff)
    spi_runoff = np.cumsum(prunoff)

    # Calculate missing statistics using vectorized operations
    nans = np.isnan(raw_p).astype(float)
    if missing == 'cumulative':
        # Cumulative sum of NaNs divided by the step number (1, 2, 3...)
        cmfr = np.cumsum(nans) / np.arange(1, len(nans) + 1)
    elif missing == 'instance':
        cmfr = nans
    else:
        raise ValueError(f"Unknown missing type: {missing}")

    # .item() converts the np.datetime64 object back to a standard python datetime/date object
    return spi, spi_eff, spi_runoff, cmfr, target_dt.item()

def Jan_1(input_end_dt,ndays):
    # Figure out which cumulative days correspond to changing years

    ny = []
    yoffset = 0
    while True:
        tdif = input_end_dt - dt.datetime(input_end_dt.year-yoffset,1,1)
        ddif = tdif.days
        if ddif > ndays:
            return ny
        ny.append(ddif)
        yoffset += 1

def day_1(input_end_dt,ndays,nmos):
    # Figure out which cumulative days correspond to changing months

    nm = []
    moffset = 0
    yoffset = 0
    i = 0
    motxt = []
    while True:
        midofmonth = dt.datetime(
            input_end_dt.year-yoffset,
            input_end_dt.month-moffset,
            15
            )
        tdif = input_end_dt - midofmonth
        ddif = tdif.days
        # Fail-safe; should never happen unless nmos is bad input
        if ddif > ndays:
            return nm, motxt
        if ddif >= 0:
            nm.append(ddif)
            motxt.append(dt.datetime.strftime(midofmonth,'%b'))
            i += 1
        moffset += 1
        if i == nmos:
            return nm, motxt
        if input_end_dt.month - moffset < 1:
            moffset -= 12
            yoffset += 1

def fill_plots(ax1, ax2, y1, y2, label, color, short=180, x=0, zorder=1):
    """
    Utility subroutine to fill the area between two (percentile) lines
    on each of two axes (panels).
    """
    if x==0:
        ax1.fill_between(
            range(short),
            y1[:short],
            y2[:short],
            label=label,
            alpha=1.0,
            color=color,
            zorder=zorder
            )
        ax2.fill_between(
            range(len(y1)),
            y1,
            y2,
            label=label,
            alpha=1.0,
            color=color,
            zorder=zorder,
            )
    else:
        if x < short:
            ax1.fill_between(
                [x,x+1],
                y1,
                y2,
                alpha=1.0,
                label=label,
                color=color,
                zorder=zorder
                )
        ax2.fill_between(
            [x,x+1],
            y1,
            y2,
            alpha=1.0,
            label=label,
            color=color,
            zorder=zorder,
            )
    return

if __name__=="__main__":
    zarr_path = Path("data/zarr/crn_subhourly_2006-2026.zarr")

    get_station = "GA_Newton_8_W"

    daily_nan_thresh = 0.3
    long_panel_years = 5

    start_day = None
    end_day = None

    """ ------------( end normal configuration )------------ """

    ## open zarr store and extract labels
    zgrp = zarr.open(zarr_path, mode="r")
    l_station = zgrp.attrs["labels"]["stations"]
    l_var = zgrp.attrs["labels"]["variables"]
    l_stats = zgrp.attrs["labels"]["stats"]

    ## determine axis indices for relevant data
    six = l_station.index(get_station)
    mix_mean = l_stats.index("mean")
    mix_count = l_stats.index("count")
    vix = l_var.index("precip")

    ## extract times and determine appropriate bounds
    times = zgrp["time"][...].astype("M8[h]")
    dtimes = times.astype(datetime).tolist()
    if start_day is None:
        start_day = dtimes[0]
    if end_day is None:
        end_day = dtimes[-1]

    ## convert precipitation to daily
    pcph = zgrp["obs"][six,vix,:,mix_mean] * zgrp["obs"][six,vix,:,mix_count]
    pcpd,dates = [],[]
    for d,vs in generate_daily(times, pcph):
        d = d.item()
        d = datetime(d.year, d.month, d.day)
        if d < start_day:
            continue
        if d > end_day:
            break
        dates.append(d)
        nan_frac = np.count_nonzero(np.isnan(vs) / vs.size)
        if nan_frac > daily_nan_thresh:
            pcpd.append(np.nan)
        else:
            pcpd.append(np.nansum(vs))
    pcpd = np.array(pcpd) / 25.4 ## convert to inches
    dates = np.array(dates, dtype="M8[D]")

    spi,spi_e,spi_ro,mfrac,date_f = get_cumsum(
        dates=dates,
        precip=pcpd,
        input_end_date=end_day,
        max_ts=365*long_panel_years,
        missing="cumulative",
        )
    print(spi)

    exit(0)


    #********************** User Input Data for This Cell *****************************#
    input_end_date = 'latest'     # Use the most recent available data as the end date for the plot
    #input_end_date = '1956-06-25'     # or specify the end date for the plot
    #end_force = True                  # Not implemented: require specified end date even if data doesn't exist
    #form = 'SPIeff'
    form = 'SPI'
    #form = 'SPIrunoff'
    display = '2panel'     # otherwise, separate plots
    topwet = False         # True shows full range in 2nd plot, False shows up to actual
    end_force = False
    if form == 'SPIeff':
        maxyr = 2                         # The number of years to plot in the long-term panel
    else:
        maxyr = 5
    max_mis_frac = 0.10               # Choose maximum fraction of missing data
    missing = 'instance'
    #missing = 'cumulative'
    #*********************** End of User Input Data for This Cell ************************#

    ptype = {'SPI' : '',
             'SPIeff' : 'EFFECTIVE',
             'SPIrunoff':'RUNOFF'}
    max_ts = 365*maxyr
    if input_end_date == 'latest':
        end_date = raw_data.index[-1]
    else:
        end_date = input_end_date
    input_end_dt = dt.datetime.strptime(end_date,'%Y-%m-%d')
    cser, mser, end_dt = get_cumsum(raw_data, input_end_dt, max_ts,
                                    missing=missing, enforce_date=end_force,
                                    form=form)   # Added form 2-25-25
    try:
        end_date = dt.datetime.strftime(end_dt,'%Y-%m-%d')
    except:
        end_date = end_dt
        end_dt = dt.datetime.strptime(end_date,'%Y-%m-%d')

    other_years=[]
    if historical == None:  # added block with Else; should convert to subroutine
        sdate = raw_data.index[0][:4]+end_date[4:]      # The starting date for the earliest period is either the specified
        sdt = dt.datetime.strptime(sdate,'%Y-%m-%d')          # month and day of the earliest year in the data, or the year after
        if sdate < raw_data.index[0]:
            sdt += relativedelta(years=1)
        while True:
            if sdt > dt.datetime.strptime(raw_data.index[-1],'%Y-%m-%d'):     # All done if we try to access data in the future
                break
            if sdt == end_dt:                                           # Don't include the requested year in the percentile calculations
                sdt += relativedelta(years=1)
                continue
            ocser, omser, dtime = get_cumsum(raw_data, sdt, max_ts, form=form)  # Added form 2-25-25
            ocser = [x if y<max_mis_frac else np.nan for x,y in zip(ocser, omser)]    # If too much
            year = dt.datetime.strftime(sdt,'%Y')
            other_years.append(ocser)
            sdt += relativedelta(years=1)
    else:
        sdate = hist_data.index[0][:4]+end_date[4:]      # The starting date for the earliest period is either the specified
        sdt = dt.datetime.strptime(sdate,'%Y-%m-%d')          # month and day of the earliest year in the data, or the year after
        if sdate < hist_data.index[0]:
            sdt += relativedelta(years=1)
        while True:
            if sdt > dt.datetime.strptime(hist_data.index[-1],'%Y-%m-%d'):     # All done if we try to access data in the future
                break
            if sdt == end_dt:                                           # Don't include the requested year in the percentile calculations
                sdt += relativedelta(years=1)
                continue
            ocser, omser, dtime = get_cumsum(hist_data, sdt, max_ts, form=form)  # Added form 2-25-25
            ocser = [x if y<max_mis_frac else np.nan for x,y in zip(ocser, omser)]    # If too much
            year = dt.datetime.strftime(sdt,'%Y')
            other_years.append(ocser)
            sdt += relativedelta(years=1)

    oyl = max(map(len, other_years))
    oy = np.array([x+[np.nan]*(oyl-len(x)) for x in other_years])         # Make numpy array, dealing with some short years

    ## extract historical percentiles
    pctls = [100, 98, 95, 90, 80, 70, 50, 30, 20, 10, 5, 2, 0]
    p100,p98,p95,p90,p80,p70,p50,p30,p20,p10,p05,p02,p00 = np.nanpercentile(
        oy, pctls, axis=0, method='median_unbiased')

    if display == '2panel':
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10, 8))   # two-panel plot
    else:
        fig, (ax1) = plt.subplots(figsize=(10, 8))   # separate plots (added 2-25-25)
        fig, (ax2) = plt.subplots(figsize=(10, 8))

    ax1.plot(cser[:180],label=end_date,lw=2,c='black',zorder=2.5)               # Plot desired cumulative precipitation
    ax2.plot(cser,label=end_date,lw=2,c='black',zorder=2.5)

    ax1.plot(p100[:180],label='maximum',lw=1,c='white')
    ax2.plot(p100,label='maximum',lw=1,c='white')

    fill_plots(ax1, ax2, [np.nanmax(p100)]*len(cser), p98, 'W4', 'blue')
    fill_plots(ax1, ax2, p98, p95, 'W3', 'royalblue')
    fill_plots(ax1, ax2, p95, p90, 'W2', 'cornflowerblue')
    fill_plots(ax1, ax2, p90, p80, 'W1', 'lightskyblue')
    fill_plots(ax1, ax2, p80, p70, 'W0', 'cyan')

    ax1.plot(p50[:180],label='median',linestyle= (0, (3, 1, 1, 1, 1, 1)),lw=1,c='black')  # Linestyle added 2-25-25
    ax2.plot(p50,label='median',linestyle= (0, (3, 1, 1, 1, 1, 1)),lw=1,c='black')   # Linestyle added 2-25-25
    fill_plots(ax1, ax2, p20, p30, 'D0', 'yellow')
    fill_plots(ax1, ax2, p10, p20, 'D1', 'tan')
    fill_plots(ax1, ax2, p05, p10, 'D2', 'orange')
    fill_plots(ax1, ax2, p02, p05, 'D3', 'red')
    fill_plots(ax1, ax2, [0]*len(cser), p02, 'D4', 'maroon')

    ax1.plot(p00[:180],label='minimum',lw=1,c='white')                     # Plot historic minimum
    ax2.plot(p00,label='minimum',lw=1,c='white')

    #ax2.set_ylim(top=max(cser))  #long range y limit set to max precip sum added 2-25-25
    ax1top = max(np.nanmax(p100[:180])*(21/20),np.nanmax(cser[:180])*(21/20))   # Changed 4 lines to nanmax 2-25-25
    ax1bot = min(np.nanmax(p100[:180])*(-1/20),np.nanmax(cser[:180])*(-1/20))
    if topwet:                  # added 2-25-25
        ax2top = max(np.nanmax(p100),np.nanmax(cser))
    else:
        ax2top = max(np.nanmax(cser),np.nanmax(p50))
        ax2bot = ax2top*(-1/21.)

    mlabel = 'missing'
    for i,m in enumerate(mser):                                      # White out data gaps
        if m==1:
            fill_plots(ax1, ax2, [ax2top]*2, [0,0], mlabel, 'white', x=i, zorder=2.6)
            mlabel = None     # only one 'missing' label in legend

    newyears = Jan_1(end_dt,max_ts)                                  # Identify year boundaries
    ys = int(end_date[:4])
    yloc = 0 - np.nanmax(p100)/100.                       # Changed to nanmax 2-25-25
    for ny in newyears:
        ax2.plot([ny, ny], [0, ax2top], color="black", lw=0.5, zorder=2.7)
        if ny > 182:
            ax2.text(ny-182,yloc,str(ys),horizontalalignment='center',verticalalignment='top')
        else:
            ax2.text(0,yloc,str(ys),horizontalalignment='center',verticalalignment='top')
        ys -= 1
        # *****  Added 1-10-2025 J N-G  *****
        if ny == newyears[-1]:
            if ny-182 < max_ts-365:
                ax2.text(ny+182,yloc,str(ys),horizontalalignment='center',verticalalignment='top')


    for ny in range(30,200,30):                                        # Identify 30-day boundaries
        ax1.plot([ny, ny], [ax1bot, ax1top], color="black", lw=0.5, zorder=2.7)

    yloc = 0 - np.max(p100[:180])/100.                                      # Plot month labels
    xmos, motxt = day_1(end_dt,max_ts,6)
    for xm, mt in zip(xmos, motxt):
        if xm > 0:
            ax1.text(xm,yloc,mt,horizontalalignment='center',verticalalignment='top')

    ax1.set_ylim([ax1bot,ax1top])   # Make plots prettier
    ax1.xaxis.set_ticks(range(0,190,30))
    ax1.set_xlabel('Days of Accumulated Precipitation')
    ax1.set_ylabel(ptype[form]+' Precipitation (inches)')  # Added ptype 2-25-25
    ax1.legend(loc='upper left')
    if display == '2panel':                # Added if 2-25-25
        ax1.set_title('Short Range')

    ax2.set_ylim([ax2bot,ax2top])
    ax2.xaxis.set_ticks(range(0,maxyr*365+5,365))        # Changed 2-25-25
    ax2.set_xlabel('Days of Accumulated Precipitation')
    ax2.set_ylabel(ptype[form]+' Precipitation (inches)')  # Added ptype 2-25-25
    if display == '2panel':               # Added if 2-25-25
        ax2.set_title('Long Range')
        ax2.legend(loc='upper left')
    else:
        ax2.legend(loc='lower right', facecolor = 'lightgray')


    gyears = str(int((raw_data['Missing'] == 0).sum()/365.25))
    relto = ''        # Aug 13 added relto functionality to next 8 lines of code
    if historical != None:
        relto = 'using '+historical+' for '
        gyears = str(int((hist_data['Missing'] == 0).sum()/365.25))
    if display == '2panel':                       # Added if 2-25-25
        fig.suptitle(ptype[form]+' Accumulated Precipitation Ending '+
                     end_date+' for '+sname+', '+relto+gyears+' years of data') #Added ptype 2-25-25
    else:
        ax1.set_title(ptype[form]+' Accumulated Precipitation Ending '+
                      end_date+' for '+sname+', '+relto+gyears+' years of data\n Short Range') #Added ptype 2-25-25
        ax2.set_title(ptype[form]+' Accumulated Precipitation Ending '+
                      end_date+' for '+sname+', '+relto+gyears+' years of data\n Long Range') #Added ptype 2-25-25

    plt.tight_layout()
    if len(savefigprefix) > 0:
        plt.savefig(savefigprefix+station+ptype[form]+'.png',dpi=300)   # Added 2-25-25
    plt.show()
