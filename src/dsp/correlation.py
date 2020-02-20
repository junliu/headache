# Copyright (c) 2019 Jun Liu <jliu@mpifr-bonn.mpg.de>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function

__author__ = 'Jun LIU'
__copyright__ = 'Copyright (c) 2019 Jun Liu <jliu@mpifr-bonn.mpg.de>'
__license__ = 'GPL v3'
__version__ = '1.0'

import numpy as np


# TODO
# 1. Documentation on functions


def DCF_EK(ts0, ts1, bs, lgl=None, lgh=None):

  """Discrete correlation function via the Edelson-Krolik method

    Parameters
    ----------
    t : array_like
        times of observation.  Assumed to be in increasing order.
    y : array_like
        values of each observation.  Should be same shape as t
    dy : float or array_like
        errors in each observation.
    bins : int or array_like (optional)
        if integer, the number of bins to use in the analysis.
        if array, the (nbins + 1) bin edges.
        Default is bins=20.

    Returns
    -------
    ACF : ndarray
        The auto-correlation function and associated times
    err : ndarray
        the error in the ACF
    bins : ndarray
        bin edges used in computation

  DCF_EK(ts0, ts01, bs, lgl=None, lgh=None)
  discrete correlation function via Edelson & Krolik (1988) algorithm.

  ts0 - the first time series (t0, y0, dy0)
  ts1 - the second time series (t1, y1, dy1)
  bs - bin size
  lgl - lower limit for time lag
  lgh - higher limit for time lag

  return:
    time lags
    coefficients of DCF
    errors of coefficients
  """

  t0, y0, dy0 = ts0
  t1, y1, dy1 = ts1

  t0 = np.asarray(t0)
  y0 = np.asarray(y0)

  t1 = np.asarray(t1)
  y1 = np.asarray(y1)

  max_lgl = min(t0) - max(t1)
  max_lgh = max(t0) - min(t1)

  if (lgl == None) or lgl < max_lgl:
    lgl = max_lgl

  if (lgh == None) or lgh > max_lgh:
    lgh = max_lgh

  s0 = np.std(y0, ddof=1)
  dy0 = np.asarray(dy0)*np.ones(y0.shape)
  if 0 in dy0:
    dy0 = np.zeros(y0.shape)
    mu0 = np.mean(y0)
  else:
    mu0 = np.average(y0, weights=dy0**-2)

  s1 = np.std(y1, ddof=1)
  dy1 = np.asarray(dy1)*np.ones(y1.shape)
  if 0 in dy1:
    dy1 = np.zeros(y1.shape)
    mu1 = np.mean(y1)
  else:
    mu1 = np.average(y1, weights=dy1**-2)

  dt = t0 - t1[:,None]
  udcf = ((y0-mu0)*(y1-mu1)[:,None])/((s0**2-dy0**2)*(s1**2-dy1**2))**0.5
  bins = np.append(np.flip(np.arange(-bs/2., lgl, -bs)), np.arange(bs/2., lgh, bs))

  vdcf = np.zeros(len(bins)-1)
  ddcf = np.zeros(len(bins)-1)
  m = np.zeros(len(bins)-1)
  for i in range(len(bins)-1):
    flag = (dt >= bins[i])*(dt < bins[i+1])
    if True in flag:
      m[i] = flag.sum()
      vdcf[i] = np.mean(udcf[flag])
      ddcf[i] = np.std(udcf[flag])/m[i]**0.5 # standard error

  return (bins[:-1]+bins[1:])/2., vdcf, ddcf


def MDCF_EK(ts0, ts1, bs, lgl=None, lgh=None):

  """Discrete correlation function via the modified Edelson-Krolik method

    Parameters
    ----------
    t : array_like
        times of observation.  Assumed to be in increasing order.
    y : array_like
        values of each observation.  Should be same shape as t
    dy : float or array_like
        errors in each observation.
    bins : int or array_like (optional)
        if integer, the number of bins to use in the analysis.
        if array, the (nbins + 1) bin edges.
        Default is bins=20.

    Returns
    -------
    ACF : ndarray
        The auto-correlation function and associated times
    err : ndarray
        the error in the ACF
    bins : ndarray
        bin edges used in computation

  DCF_EK(ts0, ts01, bs, lgl=None, lgh=None)
  discrete correlation function via Edelson & Krolik (1988) algorithm.

  ts0 - the first time series (t0, y0, dy0)
  ts1 - the second time series (t1, y1, dy1)
  bs - bin size
  lgl - lower limit for time lag
  lgh - higher limit for time lag

  return:
    time lags
    coefficients of DCF
    errors of coefficients
  """

  t0, y0, dy0 = ts0
  t1, y1, dy1 = ts1

  t0 = np.asarray(t0)
  y0 = np.asarray(y0)

  t1 = np.asarray(t1)
  y1 = np.asarray(y1)

  max_lgl = min(t0) - max(t1)
  max_lgh = max(t0) - min(t1)

  if (lgl == None) or lgl < max_lgl:
    lgl = max_lgl

  if (lgh == None) or lgh > max_lgh:
    lgh = max_lgh

  s0 = np.std(y0, ddof=1)
  mu0 = np.mean(y0)

  s1 = np.std(y1, ddof=1)
  mu1 = np.mean(y1)

  bins = np.array([])
  dt = t0 - t1[:,None]
  udcf = (y0-mu0)*(y1-mu1)[:,None]
  bins = np.append(np.flip(np.arange(-bs/2., lgl, -bs)), np.arange(bs/2., lgh, bs))

  vdcf = np.zeros(len(bins)-1)
  ddcf = np.zeros(len(bins)-1)
  m = np.zeros(len(bins)-1)
  for i in range(len(bins)-1):
    flag = (dt >= bins[i])*(dt < bins[i+1])
    if True in flag:
      m[i] = flag.sum()
      vdcf[i] = np.mean(udcf[flag])
      ddcf[i] = np.std(udcf[flag])/m[i]**0.5 # standard error

  return (bins[:-1]+bins[1:])/2., vdcf, ddcf





