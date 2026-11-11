import numpy as np
import scipy.stats as spstats

def calc_anderson_darling(x, dist, params):
    """
    calculate the anderson-darling fit for a distribution

    :@param x: observed samples
    :@param dist: scipy distribution object
    :@param params: distributional parameters
    """
    x_sorted = np.sort(np.asarray(x))
    n = len(x_sorted)
    i = np.arange(1, n + 1)

    ## logcdf(x) is log(CDF), logsf(x) is log(1 - CDF)
    logcdf = dist.logcdf(x_sorted, *params)
    logsf = dist.logsf(x_sorted, *params)

    ## check for invalid values caused by values outside supported bounds
    if np.any(np.isnan(logcdf)) or np.any(np.isnan(logsf)):
        return np.nan

    ad_stat = -n - np.sum((2 * i - 1) * (logcdf + logsf[::-1])) / n
    return float(ad_stat)

    '''
    x = np.sort(np.asarray(x))
    n = len(x)

    ## theoretical cumulative probabilities
    cdf = dist.cdf(x, *params)

    ## Avoid log(0) and log(1)
    eps = np.finfo(float).eps
    cdf = np.clip(cdf, eps, 1 - eps)

    ## Anderson-Darling stats
    i = np.arange(1, n + 1)
    return -n - np.sum(
        (2 * i - 1) *
        (np.log(cdf) + np.log(1 - cdf[::-1]))
        ) / n
    '''

def calc_sse(dist, params, bin_edges, hist_y):
    """
    calculate sum of squared errors for the distribution given a data histogram

    :@param dist: scipy distribution object
    :@param params: distributional parameters
    :@param hist_x: bin center values for the histogram
    :@param hist_y: probability density for each bin based on observations
    """
    #pdf = dist.pdf(hist_x, *params)
    #return np.sum((hist_y - pdf)**2) ## sum of squared errors
    ## integrated bin probabilities / bin widths yields exact mean density
    bin_widths = np.diff(bin_edges)
    expected_prob = dist.cdf(bin_edges[1:], *params) \
            - dist.cdf(bin_edges[:-1], *params)
    expected_density = expected_prob / bin_widths
    return float(np.sum((hist_y - expected_density) ** 2))

def calc_aic(x, dist, params, num_params):
    """
    calculate the akaike information criterion for a distribution

    :@param x: observed samples
    :@param dist: scipy distribution object
    :@param params: distributional parameters
    :@param num_params: number of free parameters fitted
    """
    return 2 * num_params - 2 * np.sum(dist.logpdf(x, *params))

def fit_and_check_dists(
        x:np.array, distributions:list,
        hist_bins:int, hist_bounds:tuple, nparams:int,
        return_hist=False, scipy_fit_kwargs={}
        ):
    """
    Fit multiple distributions to the input data and return a list of
    parameters and error metrics ordered like:
    (sse, anderson-darling, akaike information criterion)
    for each distribution in order. Optionally also returns histogram used
    to calculate SSE
    """
    hy,hx = np.histogram(x, bins=hist_bins, range=hist_bounds, density=True)
    print(np.round(np.amax(x)))
    #print(np.unique(hx))
    #print(np.unique(hy))
    xcent = (hx[:-1]+hx[1:]) / 2
    all_sse = []
    err = [
        [], ## sse
        [], ## ad
        [], ## aic
        ]
    all_params = []
    for dk in distributions:
        dist = getattr(spstats, dk)
        params = dist.fit(x, **scipy_fit_kwargs)
        #err[0].append(calc_sse(dist, params, xcent, hy))
        err[0].append(calc_sse(dist, params, hx, hy))
        err[1].append(calc_anderson_darling(x, dist, params))
        err[2].append(calc_aic(x, dist, params, nparams))
        all_params.append(params)
    if return_hist:
        return all_params,err,(xcent,hy)
    return all_params,err

def generate_daily(times, data):
    dates = times.astype("datetime64[D]")
    ## indices where calendar day changes
    boundaries = np.flatnonzero(dates[1:] != dates[:-1]) + 1
    boundaries = np.concatenate(([0], boundaries, [len(data)]))

    ## yield each day
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        yield (dates[start], data[start:end])

def doy_window_masks(times, window_days):
    """
    generate boolean masks for trailing day-of-year windows.

    Parameters
    ----------
    times : array_like, shape (N,)
        One-dimensional datetime64 array with arbitrary time resolution.

    window_days : int
        Number of days in each trailing day-of-year window.

    Yields
    ------
    mask : ndarray, shape (N,), dtype=bool
        Boolean mask identifying timestamps whose day-of-year
        falls within the trailing window ending on the current
        day of year.

    Notes
    -----
    - Generates exactly 366 masks, starting January 1st.
    - Day-of-year is calculated independently for each timestamp.
    - Windows wrap around the end of the year.
    - February 29th is handled naturally.
    - Time-of-day is ignored; all timestamps on a calendar day
      are treated identically.
    """

    times = np.asarray(times)

    if times.ndim != 1:
        raise ValueError("times must be a 1D array.")

    if not np.issubdtype(times.dtype, np.datetime64):
        raise TypeError("times must have datetime64 dtype.")

    if not isinstance(window_days, (int, np.integer)):
        raise TypeError("window_days must be an integer.")

    if not 1 <= window_days <= 366:
        raise ValueError("window_days must be between 1 and 366.")

    # Extract month and day, using a leap year as the reference.
    # This ensures February 29th has its own day-of-year.
    year = times.astype("datetime64[Y]")

    # Determine leap-year day-of-year using a reference leap year.
    reference_dates = (
        np.datetime64("2000-01-01")
        + (times - year).astype("timedelta64[D]")
        )

    # Convert dates to zero-based day-of-year (0 to 365).
    reference_year = reference_dates.astype("datetime64[Y]")

    doy = (
        reference_dates.astype("datetime64[D]")
        - reference_year.astype("datetime64[D]")
        ).astype(int)

    # Generate one mask for each calendar day.
    for day in range(366):

        # Trailing circular day-of-year window.
        window_start = (day - window_days + 1) % 366

        if window_start <= day:
            mask = (doy >= window_start) & (doy <= day)
        else:
            # Window crosses the end of the calendar year.
            mask = (doy >= window_start) | (doy <= day)

        yield mask

def doy_window_slices(times, window_days):
    """
    Wrapper around `doy_window_masks` that yields a list of slice objects for
    each contiguous sequence of True values in the day-of-year window masks.

    Slices that overlap the beginning or end of `times` and do not fully span
    a `window_days` period are omitted.

    Parameters
    ----------
    times : array_like, shape (N,)
        One-dimensional datetime64 array with arbitrary time resolution.
    window_days : int
        Number of days in each trailing day-of-year window.

    Yields
    ------
    slices : list of slice
        List of slice objects corresponding to full trailing window sequences.
    """
    times = np.asarray(times)
    n_total = len(times)

    # Compute DoY for each timestamp once to evaluate slice span completeness
    year = times.astype("datetime64[Y]")
    reference_dates = (
        np.datetime64("2000-01-01")
        + (times - year).astype("timedelta64[D]")
    )
    reference_year = reference_dates.astype("datetime64[Y]")
    doy = (
        reference_dates.astype("datetime64[D]")
        - reference_year.astype("datetime64[D]")
    ).astype(int)

    for mask in doy_window_masks(times, window_days):
        if not np.any(mask):
            yield []
            continue

        # Find contiguous sequences of True values by detecting transitions
        padded = np.empty(n_total + 2, dtype=bool)
        padded[0] = False
        padded[1:-1] = mask
        padded[-1] = False

        diff = np.diff(padded.view(np.int8))
        starts = np.flatnonzero(diff == 1)
        stops = np.flatnonzero(diff == -1)

        valid_slices = []
        for st, sp in zip(starts, stops):
            # Check if slice is cut off by the dataset boundaries
            touches_boundary = (st == 0) or (sp == n_total)

            if touches_boundary:
                # Count unique calendar days of year in the truncated slice
                unique_days = len(np.unique(doy[st:sp]))
                if unique_days < window_days:
                    continue  # Omit incomplete window slice

            valid_slices.append(slice(st, sp))

        yield valid_slices
