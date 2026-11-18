import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
#import urllib.request, urllib.error
#import contextlib
#import socket
#import requests
import json

def get_cumsum(raw_data, input_end_date, max_ts, missing='cumulative',
        enforce_date = False, form='SPI'):
    # Calculate the cumulative precipitation total ending on specified date
    # Added form 2-25-25

    if missing not in ('cumulative','instance'):
        print('I do not know what missing = ',missing,'means -- aborting.')
        return
    end_dt = input_end_date
    try:      # Get ending row number; date might be string or datetime object
        edloc = raw_data.index.get_loc(input_end_date)
    except:
        try:
            edloc = raw_data.index.get_loc(
                dt.datetime.strftime(input_end_date,'%Y-%m-%d')
                )
        except:
            if end_force:
                # add missing data to raw_data to get to input_end_date
                print('sorry, not implemented')
            #else:
            # If the requested date doesn't exist, use the latest available
            edloc = len(raw_data) - 1
            print(input_end_date,'does not exist, using most recent date')
            end_dt = raw_data.index[-1]

    # Figure out how far back to go
    sloc = edloc - max_ts
    sloc = max(sloc,0)

    # Get the precip data and missing data info;
    # rearrange so first element is most recent day
    subset = raw_data[sloc:edloc+1]
    pser = subset['Precipitation'].tolist()
    mser = subset['Missing'].tolist()

    # Added next block of code 2-25-25 John N-G
    if form == 'SPIeff':
        df = pd.DataFrame(pser, columns=['pcpn'])
        pser = df.pcpn.rolling(6).apply(
            effect_wrapper,kwargs={'output':'fi'}
            ).to_list()
    elif form == 'SPIrunoff':
        df = pd.DataFrame(pser, columns=['pcpn'])
        pser = df.pcpn.rolling(6).apply(
            effect_wrapper,kwargs={'output':'q'}
            ).to_list()

    pser.reverse()
    mser.reverse()

    cser = [pser[0],]
    cm = mser[0]
    cmfr = [cm,]
    for i in range(1,len(pser)):
        # Calculate cumulative precipitation sum
        cser.append(cser[i-1] + pser[i])
        if missing == 'cumulative':
            cm += mser[i]
            # Calculate fraction of missing obs
            cmfr.append(cm/(i+1.))
        elif missing == 'instance':
            cmfr.append(mser[i])
    return cser, cmfr, end_dt

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

def cleanup(df,space=''):
    """
    Get raw dataframe into ship shape by handling traces, missing precip,
    and multi-day totals
    """
    df.loc[df.Precipitation == space+'T','Precipitation'] = '0.00'
    df['Missing'] = 0
    df.loc[df.Precipitation == space+'M','Missing'] = 1
    df.loc[df.Precipitation == space+'M','Precipitation'] = '0.00'
    plst = df['Precipitation'].to_list()
    # Spread multiday reports out over the multi days
    for i,p in enumerate(plst):
        if p[-1] == 'A':
            ptot = float(p[:-2])
            j=i-1
            while j >= 0:
                if plst[j] != space+'S':
                    break
                j -= 1
            pday = ptot/(i-j)
            for k in range(j,i):
                plst[k+1] = f'{pday:6.3f}'
    df.Precipitation = plst
    # Sometimes there's an 'S' without a corresponding 'A'
    df.loc[df.Precipitation == space+'S','Missing'] = 1
    df.loc[df.Precipitation == space+'S','Precipitation'] = '0.00'
    df.Precipitation = df.Precipitation.astype(float)
    return df

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

# Calculate effective precipitation and runoff
def effect(p):
    '''
    Calculates effective precipitation and runoff using a modified curve number
    approach Based on M.G. Bos et al. 2009 Water Requirements for Irrigation
    and the Environment (Springer) but using inches rather than millimeters

    Except:
    Assumes uniform soil conditions with reference curve number cnref
    Does not differentiate between growing and dormant seasons, using
    intermediate values instead Estimates that initial precip Io may be lost to
    evaporation, where Io = pl * min (ne, preceding consecutive days <= pl)

    Input:
    p: current and five preceding days of precipitation, in numpy array
    (current day is p[-1])

    Output:
    Dictionary with effective precipitation (fi) and runoff (q)

    Future work:
    Differentiate between growing and dormant seasons
    Make initial precipitation loss dependent on climatological vpd or other
    measure of pet or growing/dormant seasons Use curve number reference value
    from reliable source (USDA? DSSAT?)
    '''
    pl = 0.05 # precipitation reference value in initial precip formula
    ne = 3    # max number of days in initial precip formula; cannot be larger
              # than input array - 1
              # Note: maximum initial precip loss is ne*pl
    cnref = 74  # reference curve number value
    cnlow = 55
    cnhigh = 88
    amclow = 1.   # if 5-day precip less than this, use low curve number
    amchigh = 1.6 # if 5-day precip more than this, use high curve number

    # shortcuts
    psum = np.sum(p[:-1])
    #pob = p[-1]  # Dunno why this doesn't work
    pob = np.sum(p[-1:])
    if np.isnan(psum) or np.isnan(pob):
        return({'fi':np.nan,'q':np.nan})
    if pob == 0:
        return({'fi':0.,'q':0.})

    plist = list(p)
    if psum < amclow:
        cn = cnlow
    elif psum > amchigh:
        cn = cnhigh
    else:
        cn = cnref
    s = -1. + 100/cn

    Io = 0
    for i in range(2,5):
        #if p[-i] <= pl:
        #if np.sum(p[-i]) <= pl:     # Same dunno problem as above
        if plist[-i] <= pl:
            Io += pl
        else:
            break

    a = max(pob - Io, 0)
    Ia = 0.2 * s
    if pob <= Ia:
        return({'fi':a,'q':0.})
    q = (pob-Ia)*(pob-Ia)/(pob-Ia+s)
    fi = a - q
    return({'fi':fi,'q':q})

def effect_wrapper(p,output='both'):
    """
    Needed until rolling.apply can handle dictionary function output
    """
    efdict = effect(p)
    if output == 'fi':
        return(efdict['fi'])
    elif output == 'q':
        return(efdict['q'])
    elif output == 'both':
        return(efdict)
    else:
        print('effect_wrapper does not know what to do')
        return(-1)

if __name__=="__main__":
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

    if display == '2panel':
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10, 8))   # two-panel plot
    else:
        fig, (ax1) = plt.subplots(figsize=(10, 8))   # separate plots (added 2-25-25)
        fig, (ax2) = plt.subplots(figsize=(10, 8))

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

    ax1.plot(cser[:180],label=end_date,lw=2,c='black',zorder=2.5)               # Plot desired cumulative precipitation
    ax2.plot(cser,label=end_date,lw=2,c='black',zorder=2.5)
                                                    # Future option: fill missing values with median increments

    p100 = np.nanpercentile(oy,100,axis=0,method='median_unbiased')        # Plot historic maximum
    ax1.plot(p100[:180],label='maximum',lw=1,c='white')
    ax2.plot(p100,label='maximum',lw=1,c='white')

    p98 = np.nanpercentile(oy,98,axis=0,method='median_unbiased')          # Plot shaded historic wet percentiles
    fill_plots(ax1, ax2, [np.nanmax(p100)]*len(cser), p98, 'W4', 'blue')
    p95 = np.nanpercentile(oy,95,axis=0,method='median_unbiased')
    fill_plots(ax1, ax2, p98, p95, 'W3', 'royalblue')
    p90 = np.nanpercentile(oy,90,axis=0,method='median_unbiased')
    fill_plots(ax1, ax2, p95, p90, 'W2', 'cornflowerblue')
    p80 = np.nanpercentile(oy,80,axis=0,method='median_unbiased')
    fill_plots(ax1, ax2, p90, p80, 'W1', 'lightskyblue')
    p70 = np.nanpercentile(oy,70,axis=0,method='median_unbiased')
    fill_plots(ax1, ax2, p80, p70, 'W0', 'cyan')

    p50 = np.nanpercentile(oy,50,axis=0,method='median_unbiased')         # Plot historic median
    ax1.plot(p50[:180],label='median',linestyle= (0, (3, 1, 1, 1, 1, 1)),lw=1,c='black')  # Linestyle added 2-25-25
    ax2.plot(p50,label='median',linestyle= (0, (3, 1, 1, 1, 1, 1)),lw=1,c='black')   # Linestyle added 2-25-25
    p30 = np.nanpercentile(oy,30,axis=0,method='median_unbiased')
    p20 = np.nanpercentile(oy,20,axis=0,method='median_unbiased')
    fill_plots(ax1, ax2, p20, p30, 'D0', 'yellow')
    p10 = np.nanpercentile(oy,10,axis=0,method='median_unbiased')
    fill_plots(ax1, ax2, p10, p20, 'D1', 'tan')
    p05 = np.nanpercentile(oy,5,axis=0,method='median_unbiased')
    fill_plots(ax1, ax2, p05, p10, 'D2', 'orange')
    p02 = np.nanpercentile(oy,2,axis=0,method='median_unbiased')
    fill_plots(ax1, ax2, p02, p05, 'D3', 'red')
    p00 = np.nanpercentile(oy,0,axis=0,method='median_unbiased')
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
