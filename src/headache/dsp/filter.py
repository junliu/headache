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

__author__ = 'Jun LIU'
__copyright__ = 'Copyright (c) 2019 Jun Liu <jliu@mpifr-bonn.mpg.de>'
__license__ = 'GPL v3'
__version__ = '1.0'


import numpy as np
from scipy import interpolate

def despike_spline(t, y, bs, endpt=None, tol=3.0):

  flt_full = np.ones(t.shape, dtype=bool)
  while 1:
    if endpt == None:
      endpt = [0, 0, 0, 0]
      endpt[0] = min(t) - 3*bs
      endpt[2] = max(t) + 3*bs
      endpt[1] = endpt[3] = np.mean(y[flt_full])

    tt, yy = [], []
    slots = np.arange(endpt[0], endpt[2]+bs, bs)

    for sl in slots:
      flt = (t[flt_full] >= sl)*(t[flt_full] < sl+bs)
      if True in flt:
        tt.append(np.mean(t[flt_full][flt]))
        yy.append(np.mean(y[flt_full][flt]))

    tt = np.asarray(tt)
    yy = np.asarray(yy)

    tck = interpolate.splrep(tt, yy)
    dtval = y - interpolate.splev(t, tck)
    rms = np.std(dtval[flt_full], ddof=1)

    flt_full_new = abs(dtval) <= tol*rms
    if not np.array_equal(flt_full, flt_full_new):
      flt_full = flt_full_new[:]
    else:
      break

  return flt_full

