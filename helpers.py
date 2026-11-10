import numpy as np
import scipy.stats as spstats

def fit_and_check_dists(x:np.array, distributions:list, hist_bins:int,
        return_hist=False, scipy_fit_kwargs={}):
    """ """
    hy,hx = np.histogram(x, bins=hist_bins, density=True)
    xcent = (hx[:-1]+hx[1:]) / 2
    all_sse = []
    all_params = []
    for dk in distributions:
        dist = getattr(spstats, dk)
        params = dist.fit(x, **scipy_fit_kwargs)
        pdf = dist.pdf(xcent, *params)
        sse = np.sum((hy - pdf)**2) ## sum of squared errors
        all_sse.append(sse)
        all_params.append(params)
    if return_hist:
        return all_params,all_sse,(xcent,hy)
    return all_params,all_sse

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
