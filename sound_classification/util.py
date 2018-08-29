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

def morlet_func(time, frequency, width, displacement, norm=True):
    """ Morlet wavelet function

        The function is implemented as in Eq. (15) in John Ashmead, "Morlet Wavelets in Quantum Mechanics",
        Quanta 2012; 1: 58-70, with the replacement f -> 2*pi*f*s, to allow f to be identified with the 
        physical frequency.

        Args:
            time: float
               Time in seconds at which the function is to be evaluated
            frequency: float
                Wavelet frequency in Hz
            width: float
                Wavelet width in seconds (1-sigma width of the Gaussian envelope function)
            displacement: float
                Wavelet centroid in seconds
            norm: bool
                Include [pi^1/4*sqrt(sigma)]^-1 normalization factor

        Returns:
            y: float
                Value of Morlet wavelet function at time t
    """
    assert frequency > 0, "Frequency must be a strictly positive float"
    assert width > 0, "Width must be a strictly positive float"

    t = time
    w = 2 * np.pi * frequency * width
    s = width
    l = displacement
    x = (t-l)/s

    y = (np.exp(1j*w*x) - np.exp(-0.5*(w**2))) * np.exp(-0.5*(x**2))

    if norm:
        y *= (s * np.sqrt(np.pi) * (1 + np.exp(-w**2) - 2*np.exp(-0.75*w**2)) )**-0.5

    return np.real(y)
