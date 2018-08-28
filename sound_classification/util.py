import numpy as np

def octave_bands(band_min=-1, band_max=9):
    p = np.arange(band_min-5., band_max-4.)
    fcentre = np.power(10.,3) * np.power(2.,p)
    fd = np.sqrt(2.)
    flow = fcentre / fd
    fhigh = fcentre * fd
    return fcentre,flow,fhigh

def print_octave_bands_json(band_min, band_max):
    fcentre,flow,fhigh = octave_bands(min_band_no, max_band_no)
    print("\"frequency_bands\": [")
    n = len(flow)
    for i in range(n):
        print("{")
        print("\t\"name\": \"{0:.0f}Hz\",".format(fcentre[i]))
        print("\t\"range\": [\"{1:.1f}Hz\", \"{2:.1f}Hz\"]".format(flow[i],fhigh[i]))
        endpar = "}"
        if i < n-1:
            endpar += ","
        print(endpar)
    print("]")

def morlet_func(t, frequency, sigma, displacement, norm=True):
    """ Morlet wavelet function

        Args:
            t: float
               Time in seconds at which the function is to be evaluated
            frequency: float
                Wavelet frequency in Hz
            sigma: float
                Wavelet width in seconds (1-sigma width of the Gaussian envelope function)
            displacement: float
                Wavelet centroid in seconds
            norm: bool
                Include [pi^1/4*sqrt(sigma)]^-1 normalization factor

        Returns:
            y: float
                Value of Morlet wavelet function at time t
    """

    w = 2 * np.pi * frequency
    s = sigma
    l = displacement
    x = (t-l)/s

    if w > 5:
        y = np.exp(1j*w*x) * np.exp(-0.5*(x**2))
    else:
        y = (np.exp(1j*w*x) - np.exp(-0.5*(w**2))) * np.exp(-0.5*(x**2))

    if norm:
        y *= np.pi**-0.25 * 1. / np.sqrt(s)

    return np.real(y)
