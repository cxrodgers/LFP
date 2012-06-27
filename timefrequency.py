import numpy as np
import matplotlib.mlab as mlab
from ns5_process.myutils import Spectrogrammer, printnow

def psd(data, meth='avg_db', normalization=None, drop_DC=True,
    detrend=mlab.detrend_mean, pow2_warn=True, **psd_kwargs):
    """Returns the spectrum of a list of signals.
    
    Applies psd to array data, collects frequency bins, optionally collapses
    across replicates in one of a few ways.
    
    data : array data. If 2d, should contain replicates on the rows
    meth : 'avg_first', 'avg_db', 'all' (see psd_average)
    normalization :
        multiply power in each bin by freq**normalization
        So a 1/f spectrum (in amplitude) is normalized by 2.0
    
    Other kwargs to pass to mlab.psd:
        Fs
        NFFT
        window
        noverlap
    
    Returns: (Pxx, freqs)
    If meth is 'all', Pxx has shape (N_signals, NFFT/2)
    If meth is 'avg_db', Pxx has shape (NFFT/2,)
    """
    if pow2_warn:
        if data.ndim == 2 and np.mod(data.shape[1], 2) != 0:
            print "warning: data has non-power 2 length, possibly transposed?"
        elif data.ndim == 1 and np.mod(data.shape[0], 2) != 0:
            print "warning: data has non-power 2 length"
    
    Pxx_list = []
    for sig in data:
        Pxx, freqs = mlab.psd(sig, detrend=detrend, **psd_kwargs)
        Pxx = Pxx.flatten()
        if normalization is not None:
            Pxx = Pxx * freqs**normalization
        # Normalize and add to list
        Pxx_list.append(Pxx.flatten())
    
    Pxxa = np.array(Pxx_list)
    
    if drop_DC:
        Pxxa = Pxxa[:, 1:]
        freqs = freqs[1:]
    
    Pxxaa = psd_average_db(Pxxa, meth=meth)
    
    return Pxxaa, freqs
    

def psd_average_db(Pxx, meth='avg_db'):
    """Collapse Pxx in specified way and convert to decibels.
    
    Pxx : spectra, replicates in rows
    meth : how to collapse
        'avg_first' : average signals, compute spectrum, convert to dB
        'avg_db' : compute spectrum, convert to dB, average spectra
        'all' : return all spectra in dB without averaging
    """
    Pxx = np.asarray(Pxx)
    
    if Pxx.ndim == 1 or meth == 'all':
        res = 10 * np.log10(Pxx)
        
    elif meth == 'avg_db':
        Pxx = 10 * np.log10(Pxx)
        if np.any(np.isinf(Pxx)):
            print "warning: discarding Pxx with inf before averaging"
        Pxx = Pxx[~np.any(np.isinf(Pxx), axis=1), :]
        
        # Average the decibels
        res = np.mean(Pxx, axis=0)
    
    elif meth == 'avg_first':
        if np.any(np.isinf(Pxx)):
            print "warning: discarding Pxx with inf before averaging"
        Pxx = Pxx[~np.any(np.isinf(Pxx), axis=1), :]
        
        # Decibel the average
        res = 10 * np.log10(np.mean(Pxx, axis=0))
    
    else:
        raise ValueError("unsupported method %s" % meth)
    
    return res