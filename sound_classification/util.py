import numpy as np
import matplotlib.pyplot as plt

def plot_labeled_spec(spec, label, pred=None, feat=None, conf=None):

    nrows = 2
    if (pred is not None): 
        nrows += 1
    if (feat is not None): 
        nrows += 1
    if (conf is not None): 
        nrows += 1

    fig, ax = plt.subplots(nrows=nrows, ncols=1, figsize=(9, 2+2*nrows))

    # spectrogram
    x = spec.image
    img_plot = ax[-1].imshow(x.T, aspect='auto', origin='lower', extent=(0, spec.duration(), spec.fmin, spec.fmax()))
    ax[-1].set_xlabel('Time (s)')
    ax[-1].set_ylabel('Frequency (Hz)')
    fig.colorbar(img_plot, ax=ax[-1], format='%+2.0f dB')

    row = 0

    # time axis
    t_axis = np.zeros(len(label))
    for i in range(1,len(label)):
        t_axis[i] = t_axis[i-1] + spec.tres
    t_axis += 0.5 * (x.shape[0] - len(label)) * spec.tres

    # confidence
    if conf is not None:
        ax[row].plot(t_axis, conf, color='C1')
        ax[row].set_xlim(0, spec.duration())
        ax[row].set_ylim(-0.1, 1.1)
        ax[row].set_ylabel('confidence')
        fig.colorbar(img_plot, ax=ax[row]).ax.set_visible(False)  # add one more colorbar, but make them invisible
        row += 1

    # predictions
    if pred is not None:
        ax[row].plot(t_axis, pred, color='C1')
        ax[row].set_xlim(0, spec.duration())
        ax[row].set_ylim(-0.1, 1.1)
        ax[row].set_ylabel('prediction')
        fig.colorbar(img_plot, ax=ax[row]).ax.set_visible(False)  # add one more colorbar, but make them invisible
        row += 1

    # feat
    if feat is not None:
        m = np.max(feat, axis=0)
        idx = np.argwhere(m > 0)
        idx = np.squeeze(idx)
        x = feat[:,idx]
        x = x / np.max(x, axis=0)
        img_plot = ax[row].imshow(x.T, aspect='auto', origin='lower', extent=(0, spec.duration(), 0, 1))
        ax[row].set_ylabel('feature #')
        fig.colorbar(img_plot, ax=ax[row])
        row += 1

    # labels
    ax[row].plot(t_axis, label, color='C2')
    ax[row].set_xlim(0, spec.duration())
    ax[row].set_ylim(-0.1, 1.1)
    ax[row].set_ylabel('label')
    fig.colorbar(img_plot, ax=ax[row]).ax.set_visible(False)
    row += 1

    return fig

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

def morlet_func(time, frequency, width, displacement, norm=True, dfdt=0):
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
            dfdt: float
                Rate of change in frequency as a function of time in Hz per second.
                If dfdt is non-zero, the frequency is computed as 
                    
                    f = frequency + (time - displacement) * dfdt 

        Returns:
            y: float
                Value of Morlet wavelet function at time t
    """
    if dfdt != 0:
        frequency += (time - displacement) * dfdt
    
    assert np.all(frequency > 0), "Frequency must be a strictly positive float"
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
