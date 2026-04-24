# -*- coding: utf-8 -*-
"""
TODO: module docs... testing for pynlo.light methods and classes
"""

# %% Imports

import numpy as np
from scipy import constants
if __name__ == "__main__":
    import matplotlib.pyplot as plt

from pynlo import light
from pynlo.utility import fft


# %% Constants

pi = constants.pi


# %% Pulse

def test_pulse_shapes():
    v_min = 100e12
    v_max = 500e12
    n = 2**8 + 0
    v_0 = 300e12
    e_p = 1
    t_fwhm = 100e-15

    #--- Gaussian Pulse
    test = light.Pulse.Gaussian(n, v_min, v_max, v_0, e_p, t_fwhm)
    t_w = test.t_width()
    assert np.isclose(e_p, test.e_p)
    assert np.isclose(test.v_grid[test.p_v.argmax()], v_0, atol=test.dv, rtol=0)
    assert np.isclose(t_w.fwhm, t_fwhm, atol=test.dv, rtol=0)

    #--- Sech**2 Pulse
    test = light.Pulse.Sech(n, v_min, v_max, v_0, e_p, t_fwhm)
    t_w = test.t_width()
    assert np.isclose(e_p, test.e_p)
    assert np.isclose(test.v_grid[test.p_v.argmax()], v_0, atol=test.dv, rtol=0)
    assert np.isclose(t_w.fwhm, t_fwhm, atol=test.dv, rtol=0)

    #--- Parabolic Pulse
    test = light.Pulse.Parabolic(n, v_min, v_max, v_0, e_p, t_fwhm)
    t_w = test.t_width()
    assert np.isclose(e_p, test.e_p)
    assert np.isclose(test.v_grid[test.p_v.argmax()], v_0, atol=test.dv, rtol=0)
    assert np.isclose(t_w.fwhm, t_fwhm, atol=test.dv, rtol=0)

    #--- Lorentzian**2 Pulse
    test = light.Pulse.Lorentzian(n, v_min, v_max, v_0, e_p, t_fwhm)
    t_w = test.t_width()
    assert np.isclose(e_p, test.e_p)
    assert np.isclose(test.v_grid[test.p_v.argmax()], v_0, atol=test.dv, rtol=0)
    assert np.isclose(t_w.fwhm, t_fwhm, atol=test.dv, rtol=0)

    #--- CW Light
    p_p = 1
    test = light.Pulse.CW(n, v_min, v_max, v_0, p_p)
    assert np.isclose(p_p, test.e_p/test.t_window)
    assert np.isclose(test.v_grid[test.p_v.argmax()], v_0, atol=test.dv, rtol=0)

def test_pulse_properties():
    """
    Test Pulse properties on a generated gaussian pulse.
    """
    v_min = 100e12
    v_max = 500e12
    n = 2**8 + 0
    v_0 = 300e12
    e_p = 1
    t_fwhm = 100e-15

    test = light.Pulse.Gaussian(n, v_min, v_max, v_0, e_p, t_fwhm)
    tbw = 4 * np.log(2)/ (2*pi)
    ac_conversion = 2**.5

    #--- Frequency Domain Properties
    assert all(test.a_v == fft.fftshift(test._a_v))
    assert all(test.p_v == fft.fftshift(test._p_v))
    assert all(test.phi_v == fft.fftshift(test._phi_v))
    assert np.allclose(np.log10(test.p_v**0.5 * np.exp(1j*test.phi_v)), np.log10(test.a_v), equal_nan=True)

    #--- Time Domain Properties
    assert all(test.a_t == fft.fftshift(test._a_t))
    assert all(test.p_t == fft.fftshift(test._p_t))
    assert all(test.phi_t == fft.fftshift(test._phi_t))
    assert np.allclose(np.log10(test.p_t**0.5 * np.exp(1j*test.phi_t)), np.log10(test.a_t), equal_nan=True)
    assert all(test.ra_t == fft.fftshift(test._ra_t))
    assert all(test.rp_t == fft.fftshift(test._rp_t))
    assert np.allclose(np.log10(test.rp_t), np.log10(test.ra_t**2), equal_nan=True)

    #--- Methods
    v_w = test.v_width()
    t_w = test.t_width()
    tbw_tol =((test.dv * t_fwhm)**2 + (test.dt* tbw / t_fwhm)**2)**.5
    assert np.isclose(tbw, v_w.fwhm * t_w.fwhm, atol=tbw_tol)
    ac = test.autocorrelation()
    ac_tol = ac_conversion * test.dt
    assert np.isclose(ac.rms, ac_conversion * t_w.rms, atol=ac_tol)

#--- Exploratory
if __name__ == "__main__":
    test_pulse_shapes()
    test_pulse_properties()

    v_min = 100e12
    v_max = 500e12
    n = 2**8 + 0
    v_0 = 300e12
    e_p = 1
    t_fwhm = 40e-15
    test = light.Pulse.Gaussian(n, v_min, v_max, v_0, e_p, t_fwhm)

    # test.phi_v += 2*pi*(.1*1e-12)*test.v_grid

    #--- Complex Envelope
    plt.figure("Complex Envelope")
    plt.clf()
    ax0 = plt.subplot2grid((2,1), (0,0))
    ax1 = plt.subplot2grid((2,1), (1,0), sharex=ax0)
    ax0.semilogy(1e12*test.t_grid, test.p_t)
    ax1.plot(1e12*test.t_grid, 1e-12*test.vg_t)
    plt.tight_layout()

    #--- Power Spectrum
    plt.figure("Power Spectrum")
    plt.clf()
    ax0 = plt.subplot2grid((2,1), (0,0))
    ax1 = plt.subplot2grid((2,1), (1,0), sharex=ax0)
    ax0.semilogy(1e-12*test.v_grid, test.p_v)
    ax1.plot(1e-12*test.v_grid, 1e12*test.tg_v)
    plt.tight_layout()

    #--- Autocorrelation
    ac = test.autocorrelation()

    plt.figure("Autocorrelation")
    plt.clf()
    plt.plot(ac.t_grid, ac.ac_t)

    #--- Spectrogram
    spg = test.spectrogram()
    plt.figure("Spectrogram")
    plt.clf()
    plt.imshow(10*np.log10(spg.spg), origin="lower", extent=spg.extent, aspect="auto")
    plt.tight_layout()