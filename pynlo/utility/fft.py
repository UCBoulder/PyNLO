# -*- coding: utf-8 -*-
"""
Aliases to fast FFT implementations and associated helper functions.

"""

# __all__ = ["fft", "ifft", "rfft", "irfft",
        #    "fftshift", "ifftshift"]


# %% Imports

import os
from scipy.fft import next_fast_len, fftshift, ifftshift 

try: # Attempt to set backend as FFTW3
    import pyfftw.interfaces.scipy_fft as backend
    import scipy.fft as _fft
    _fft.set_global_backend(backend)

    import pyfftw
    pyfftw.interfaces.cache.enable()
    pyfftw.config.NUM_THREADS = os.cpu_count()
    print('Using FFTW FFT backend')

except ImportError: # If FFTW3 is not installed, fall back to native Scipy
    import scipy.fft as _fft
    print('Using Scipy FFT backend')


# %% Transforms
# 
# ---- FFTs
def fft(x, fsc=1.0, n=None, axis=-1, overwrite_x=False):
    """
    Use MKL to perform a 1D FFT of the input array along the given axis.

    # 
    ----------
    x : array_like
        Input array, can be complex.
    fsc : float, optional
        The forward transform scale factor. The default is 1.0.
    n : int, optional
        Length of the transformed axis of the output. If `n` is smaller than
        the length of the input, the input is cropped. If it is larger, the
        input is padded with zeros.
    axis : int, optional
        Axis over which to compute the FFT. The default is the last axis.
    overwrite_x : bool, optional
        If True, the contents of x may be overwritten during the computation.
        The default is False.

    Returns
    -------
    complex ndarray
        The transformed array.

    """
    return fsc * _fft.fft(x, n=n, axis=axis, overwrite_x=overwrite_x, norm='backward')
# 
def ifft(x, fsc=1.0, n=None, axis=-1, overwrite_x=False):
    """
    Use MKL to perform a 1D IFFT of the input array along the given axis.

    Parameters
    ----------
    x : array_like
        Input array, can be complex.
    fsc : float, optional
        The forward transform scale factor. Internally, this function sets the
        reverse transform scale factor as ``1/(n*fsc)``. The default is 1.0.
    n : int, optional
        Length of the transformed axis of the output. If `n` is smaller than
        the length of the input, the input is cropped. If it is larger, the
        input is padded with zeros.
    axis : int, optional
        Axis over which to compute the inverse FFT. The default is the last
        axis.
    overwrite_x : bool, optional
        If True, the contents of x may be overwritten during the computation.
        The default is False.

    Returns
    -------
    complex ndarray
        The transformed array.

    """
    return 1/fsc * _fft.ifft(x, n=n, axis=axis, overwrite_x=overwrite_x, norm='backward')

# ---- Real FFTs
def rfft(x, fsc=1.0, n=None, axis=-1):
    """
    Use MKL to perform a 1D FFT of the real input array along the given axis.
    The output array is complex and only contains positive frequencies.

    The length of the transformed axis is ``n//2 + 1``.

    Parameters
    ----------
    x : array_like
        Input array, must be real.
    fsc : float, optional
        The forward transform scale factor. The default is 1.0.
    n : int, optional
        Number of points to use along the transformed axis of the input. If
        `n` is smaller than the length of the input, the input is cropped. If
        it is larger, the input is padded with zeros.
    axis : int, optional
        Axis over which to compute the FFT. The default is the last axis.

    Returns
    -------
    complex ndarray
        The transformed array.

    """
    return fsc * _fft.rfft(x, n=n, axis=axis, norm='backward')
# 
def irfft(x, fsc=1.0, n=None, axis=-1):
    """
    Use MKL to perform a 1D IFFT of the input array along the given axis. The
    input is assumed to contain only positive frequencies, and the output is
    always real.

    If `n` is not given the length of the transformed axis is ``2*(m-1)``,
    where `m` is the length of the transformed axis of the input. To get an odd
    number of output points, `n` must be specified.

    Parameters
    ----------
    x : array_like
        Input array, can be complex.
    fsc : float, optional
        The forward transform scale factor. Internally, this function sets the
        reverse transform scale factor as ``1/(n*fsc)``. The default is 1.0.
    n : int, optional
        Length of the transformed axis of the output. For `n` output points,
        ``n//2+1`` input points are necessary. If the input is longer than
        this, it is cropped. If it is shorter than this, it is padded with
        zeros.
    axis : int, optional
        Axis over which to compute the inverse FFT. The default is the last
        axis.

    Returns
    -------
    ndarray
        The transformed array.

    """
    return 1/fsc * _fft.irfft(x, n=n, axis=axis, norm='backward')
