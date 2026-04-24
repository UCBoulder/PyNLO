# -*- coding: utf-8 -*-
"""
TODO: module docs... testing for pynlo.utility methods and classes
"""

# %% Imports

import numpy as np
from scipy import constants, integrate

from pynlo import utility
from pynlo.utility import fft


# %% Constants

pi = constants.pi


# %% Resampling

class TestResampleT():
    #--- Real-Valued Input, Even Number of Points
    def test_r1(self):
        n = 2**6
        dt = 0.5

        t_grid = dt*(np.arange(n) - n//2)
        a_t = np.exp(-0.5*(t_grid/(5*dt))**2)

        #--- Even
        n_rs = 2*n
        rs_1 = utility.resample_t(t_grid, a_t, n_rs)
        rs_2 = utility.resample_t(rs_1.t_grid, rs_1.f_t, n)
        a_t_ident = rs_2.f_t
        assert np.allclose(a_t_ident, a_t)

        #--- Odd
        n_rs = 2*n + 1
        rs_1 = utility.resample_t(t_grid, a_t, n_rs)
        rs_2 = utility.resample_t(rs_1.t_grid, rs_1.f_t, n)
        a_t_ident = rs_2.f_t
        assert np.allclose(a_t_ident, a_t)

    #--- Real-Valued Input, Odd Number of Points
    def test_r2(self):
        n = 2**6 + 1
        dt = 0.5

        t_grid = dt*(np.arange(n) - n//2)
        a_t = np.exp(-0.5*(t_grid/(5*dt))**2)

        #--- Even
        n_rs = 2*n
        rs_1 = utility.resample_t(t_grid, a_t, n_rs)
        rs_2 = utility.resample_t(rs_1.t_grid, rs_1.f_t, n)
        a_t_ident = rs_2.f_t
        assert np.allclose(a_t_ident, a_t)

        #--- Odd
        n_rs = 2*n + 1
        rs_1 = utility.resample_t(t_grid, a_t, n_rs)
        rs_2 = utility.resample_t(rs_1.t_grid, rs_1.f_t, n)
        a_t_ident = rs_2.f_t
        assert np.allclose(a_t_ident, a_t)

    #--- Complex Envelope Input, Even Number of Points
    def test_c1(self):
        n = 2**6
        dt = 0.5

        t_grid = dt*(np.arange(n) - n//2)
        a_t = np.exp(-0.5*(t_grid/(5*dt))**2) + 1j*np.exp(-0.5*(t_grid/(5*dt))**2)

        #--- Even
        n_rs = 2*n
        rs_1 = utility.resample_t(t_grid, a_t, n_rs)
        rs_2 = utility.resample_t(rs_1.t_grid, rs_1.f_t, n)
        a_t_ident = rs_2.f_t
        assert np.allclose(a_t_ident, a_t)

        #--- Odd
        n_rs = 2*n + 1
        rs_1 = utility.resample_t(t_grid, a_t, n_rs)
        rs_2 = utility.resample_t(rs_1.t_grid, rs_1.f_t, n)
        a_t_ident = rs_2.f_t
        assert np.allclose(a_t_ident, a_t)

    #--- Complex Envelope Input, Odd Number of Points
    def test_c2(self):
        n = 2**6 + 1
        dt = 0.5

        t_grid = dt*(np.arange(n) - n//2)
        a_t = np.exp(-0.5*(t_grid/(5*dt))**2) + 1j*np.exp(-0.5*(t_grid/(5*dt))**2)

        #--- Even
        n_rs = 2*n
        rs_1 = utility.resample_t(t_grid, a_t, n_rs)
        rs_2 = utility.resample_t(rs_1.t_grid, rs_1.f_t, n)
        a_t_ident = rs_2.f_t
        assert np.allclose(a_t_ident, a_t)

        #--- Odd
        n_rs = 2*n + 1
        rs_1 = utility.resample_t(t_grid, a_t, n_rs)
        rs_2 = utility.resample_t(rs_1.t_grid, rs_1.f_t, n)
        a_t_ident = rs_2.f_t
        assert np.allclose(a_t_ident, a_t)

class TestResampleV():
    #--- Real-Valued Input, Even Number of Points
    def test_r1(self):
        n = 2**6
        rn = 2*(n - 1)
        dv = 0.5

        v_grid = dv*np.arange(n)
        a_v = np.exp(-0.5*(v_grid/(5*dv))**2)

        #--- Even
        n_rs = 2*rn
        rs_1 = utility.resample_v(v_grid, a_v, n_rs)
        rs_2 = utility.resample_v(rs_1.v_grid, rs_1.f_v, rn)
        a_v_ident = rs_2.f_v
        assert np.allclose(a_v_ident, a_v)

        #--- Odd
        n_rs = 2*rn + 1
        rs_1 = utility.resample_v(v_grid, a_v, n_rs)
        rs_2 = utility.resample_v(rs_1.v_grid, rs_1.f_v, rn)
        a_v_ident = rs_2.f_v
        assert np.allclose(a_v_ident, a_v)

    #--- Real-Valued Input, Odd Number of Points
    def test_r2(self):
        n = 2**6
        rn = 2*(n - 1) + 1
        dv = 0.5

        v_grid = dv*np.arange(n)
        a_v = np.exp(-0.5*(v_grid/(5*dv))**2) + 0j
        print(a_v[-1])
        a_v[-1] += 1j # force number of points in the time domain to be odd

        #--- Even
        n_rs = 2*rn
        rs_1 = utility.resample_v(v_grid, a_v, n_rs)
        rs_2 = utility.resample_v(rs_1.v_grid, rs_1.f_v, rn)
        a_v_ident = rs_2.f_v
        assert np.allclose(a_v_ident, a_v)

        #--- Odd
        n_rs = 2*rn + 1
        rs_1 = utility.resample_v(v_grid, a_v, n_rs)
        rs_2 = utility.resample_v(rs_1.v_grid, rs_1.f_v, rn)
        a_v_ident = rs_2.f_v
        assert np.allclose(a_v_ident, a_v)

    #--- Complex Envelope Input, Even Number of Points
    def test_c1(self):
        n = 2**6
        dv = 0.5

        v_grid = dv*(np.arange(n) - n//2)
        a_v = np.exp(-0.5*(v_grid/(5*dv))**2) + 1j*np.exp(-0.5*(v_grid/(5*dv))**2)

        #--- Even
        n_rs = 2*n
        rs_1 = utility.resample_v(v_grid, a_v, n_rs)
        rs_2 = utility.resample_v(rs_1.v_grid, rs_1.f_v, n)
        a_v_ident = rs_2.f_v
        assert np.allclose(a_v_ident, a_v)

        #--- Odd
        n_rs = 2*n + 1
        rs_1 = utility.resample_v(v_grid, a_v, n_rs)
        rs_2 = utility.resample_v(rs_1.v_grid, rs_1.f_v, n)
        a_v_ident = rs_2.f_v
        assert np.allclose(a_v_ident, a_v)

    #--- Complex Envelope Input, Odd Number of Points
    def test_c2(self):
        n = 2**6 + 1
        dv = 0.5

        v_grid = dv*(np.arange(n) - n//2)
        a_v = np.exp(-0.5*(v_grid/(5*dv))**2) + 1j*np.exp(-0.5*(v_grid/(5*dv))**2)

        #--- Even
        n_rs = 2*n
        rs_1 = utility.resample_v(v_grid, a_v, n_rs)
        rs_2 = utility.resample_v(rs_1.v_grid, rs_1.f_v, n)
        a_v_ident = rs_2.f_v
        assert np.allclose(a_v_ident, a_v)

        #--- Odd
        n_rs = 2*n + 1
        rs_1 = utility.resample_v(v_grid, a_v, n_rs)
        rs_2 = utility.resample_v(rs_1.v_grid, rs_1.f_v, n)
        a_v_ident = rs_2.f_v
        assert np.allclose(a_v_ident, a_v)

# %% TFGrid

def test_TFGrid():
    test = utility.TFGrid(2**6 + 1, 1e12, 200e12)

    #--- Complex Envelope Frequency Grid
    assert all(test.v_grid == fft.fftshift(test._v_grid))
    assert test.v_ref == test._v_ref
    assert test.dv == test._dv
    assert test.dv == test.rdv
    assert test.v_window == test.dv*test.n

    #--- Complex Envelope Time Grid
    assert all(test.t_grid == fft.fftshift(test._t_grid))
    assert test.t_ref == test._t_ref
    assert test.dt == test._dt
    assert test.t_window == test.n * test.dt

    #--- Real-Valued Time and Frequency Domain Grid
    assert test.rv_ref == 0
    assert test.rdv == test.dv
    assert test.rv_window == len(test.rv_grid) * test.rdv
    assert all(test.rt_grid == fft.fftshift(test._rt_grid))
    assert test.rt_ref == test.t_ref
    assert test.rdt == test._rdt
    assert test.rt_window == test.t_window

    #--- rtf_grids
    #TODO: test this method

#TODO: test utility.TFGrid.FromFreqRange(n_points, v_min, v_max)