"""
TODO: module docs... testing for pynlo.light methods and classes
"""

# %% Imports
import numpy as np
import pytest
from scipy import constants

from pynlo import light
from pynlo.utility import fft

# %% Constants
pi = constants.pi


PULSE_CLASSES = [
    light.Pulse.Gaussian,
    light.Pulse.Sech,
    light.Pulse.Parabolic,
    light.Pulse.Lorentzian,
]

# %% Pulse

@pytest.mark.parametrize("PulseClass", PULSE_CLASSES)
def test_pulse_shapes(PulseClass):
    """
    Test creation and fundamental properties of each Pulse class
    """
    v_min = 100e12
    v_max = 500e12
    n = 2**8 + 0
    v_0 = 300e12
    e_p = 1
    t_fwhm = 100e-15

    pulse = PulseClass(n, v_min, v_max, v_0, e_p, t_fwhm)
    t_w = pulse.t_width()

    assert np.isclose(e_p, pulse.e_p)
    assert np.isclose(pulse.v_grid[pulse.p_v.argmax()], v_0, atol=pulse.dv)
    assert np.isclose(t_w.fwhm, t_fwhm, atol=pulse.dt)


def test_cw_pulse():
    v_min = 100e12
    v_max = 500e12
    n = 2**8
    v_0 = 300e12
    p_p = 1

    pulse = light.Pulse.CW(n, v_min, v_max, v_0, p_p)

    assert np.isclose(p_p, pulse.e_p / pulse.t_window)
    assert np.isclose(pulse.v_grid[pulse.p_v.argmax()], v_0, atol=pulse.dv)

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

    pulse = light.Pulse.Gaussian(n, v_min, v_max, v_0, e_p, t_fwhm)

    tbw = 4 * np.log(2) / (2 * pi)
    ac_conversion = 2**0.5

    # --- Frequency domain consistency
    assert np.array_equal(pulse.a_v, fft.fftshift(pulse._a_v))
    assert np.array_equal(pulse.p_v, fft.fftshift(pulse._p_v))
    assert np.array_equal(pulse.phi_v, fft.fftshift(pulse._phi_v))

    assert np.allclose(
        pulse.a_v,
        pulse.p_v**0.5 * np.exp(1j * pulse.phi_v),
        equal_nan=True,
    )

    # --- Time domain consistency
    assert np.array_equal(pulse.a_t, fft.fftshift(pulse._a_t))
    assert np.array_equal(pulse.p_t, fft.fftshift(pulse._p_t))
    assert np.array_equal(pulse.phi_t, fft.fftshift(pulse._phi_t))

    assert np.allclose(
        pulse.a_t,
        pulse.p_t**0.5 * np.exp(1j * pulse.phi_t),
        equal_nan=True,
    )

    # --- Real-valued representations
    assert np.array_equal(pulse.ra_t, fft.fftshift(pulse._ra_t))
    assert np.array_equal(pulse.rp_t, fft.fftshift(pulse._rp_t))
    assert np.allclose(pulse.rp_t, pulse.ra_t**2, equal_nan=True)

    # --- Width relations
    v_w = pulse.v_width()
    t_w = pulse.t_width()

    tbw_tol = ((pulse.dv * t_fwhm) ** 2 + (pulse.dt * tbw / t_fwhm) ** 2) ** 0.5
    assert np.isclose(tbw, v_w.fwhm * t_w.fwhm, atol=tbw_tol)

    # --- Autocorrelation scaling
    ac = pulse.autocorrelation()
    ac_tol = ac_conversion * pulse.dt
    assert np.isclose(ac.rms, ac_conversion * t_w.rms, atol=ac_tol)

@pytest.mark.parametrize("n", [2**7, 2**8, 2**9, 2**10])
def test_tbw_convergence(n):
    v_min = 100e12
    v_max = 500e12
    v_0 = 300e12
    e_p = 1
    t_fwhm = 100e-15

    pulse = light.Pulse.Gaussian(n, v_min, v_max, v_0, e_p, t_fwhm)

    tbw_expected = 4 * np.log(2) / (2 * np.pi)

    v_w = pulse.v_width()
    t_w = pulse.t_width()

    tbw = v_w.fwhm * t_w.fwhm

    # tolerance scales with discretization
    tbw_tol = ((pulse.dv * t_fwhm) ** 2 + (pulse.dt * tbw / t_fwhm) ** 2) ** 0.5

    assert np.isclose(tbw, tbw_expected, atol=tbw_tol)