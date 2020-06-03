# Copyright (c) 2020 Jun Liu <jliu@mpifr-bonn.mpg.de>
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
__copyright__ = 'Copyright (c) 2020 Jun Liu <jliu@mpifr-bonn.mpg.de>'
__license__ = 'GPL v3'
__version__ = '1.0'

import numpy as np

def mod_transform(modfile, dflux=0, dx=0, dy=0, dr=0, dtheta=0):
  """
  mod_transform(modfile, dflux=0, dx=0, dy=0, dr=0, dtheta=0)
  """

  flux, radius, theta = np.genfromtxt(modfile, comments='!').transpose()
  flux += dflux
  radius += dr
  theta += dtheta
  xy_complex = radius*np.exp(1j*np.deg2rad(theta))
  x = xy_complex.imag + dx
  y = xy_complex.real +dy

  radius = (x**2 + y**2) ** 0.5
  theta = np.angle(y + 1j*x, deg=True)
  np.savetxt('mod_trans.mod', np.transpose([flux, radius, theta]), fmt='%.8g')


__all__ = ['mod_transform']

if __name__ == '__main__':
  pass



