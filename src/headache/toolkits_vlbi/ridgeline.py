#! /usr/bin/env python
# -*- coding: utf-8 -*-

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
__version__ = '1.6'



"""
    ridgeline
    ---------

    a sub-module for toolkits_vlbi, to look for ridgline in AGN jets

"""


import os
import argparse
import numpy as  np
from numpy import rad2deg, deg2rad, sin, cos, log10
from numpy import array as arr
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator, LogLocator
from scipy.interpolate import interp1d

from headache.coord import polar
from headache import fits_utils
from headache.plotter import set_theme
set_theme('sci')


def main():

    parser = argparse.ArgumentParser(
    description = 'ridgeline.py   version 1.6 (2019-02-22)\n\n'
        'Written by Jun LIU <jliu@mpifr-bonn.mpg.de>\n\n'
        'ridgeline comes with ABSOLUTELY NO WARRANTY\n'
        'You may redistribute copies of fits_utils\n'
        'under the terms of the GNU General Public License.',
        formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument('-i', dest='fitsname', required=True,
                        help='input fits file', metavar='image.fits')
    parser.add_argument('-c', dest='core', required=False, type=str,
                        default='None', help='location of the core (list or None)',
                        metavar='None')
    parser.add_argument('-m', dest='method', required=False, type=str,
                        default='equal', help='location of the core',
                        metavar='equal')
    parser.add_argument('--ts', dest='twosided', required=False,
                        action='store_true', help='twosided jets')
    parser.add_argument('-pa', dest='pa', required=False, type=float,
                        default=0.0, metavar=0.0,
                        help='initial guess for jet position angle in deg.')
    parser.add_argument('-dpa', dest='dpa', required=False, type=float,
                        default=90.0, metavar='90.0',
                        help='deviation of PA in deg.')
    parser.add_argument('-dpai', dest='dpa_iter', required=False, type=float,
                        default=60.0, metavar='60.0',
                        help='deviation of PA in each iteration')
    parser.add_argument('-noise', dest='noise', required=False, type=float,
                        default=0, metavar=0,
                        help='noise in the image; 0-> auto calculation')
    parser.add_argument('-dthresh', dest='detect_thresh', required=False,
                        type=float, default=10.0, metavar=10.0,
                        help='factor*<noise> for ridgeline detection')
    parser.add_argument('-rmin', dest='min_radius', required=False, type=float,
                        default=0.0, metavar=0.0,
                        help='starting radius for ridgline detection (in pixel)')
    parser.add_argument('-rmax', dest='max_radius', required=False, type=float,
                        default=120.0, metavar=120.0,
                        help='maximum radius for ridgline detection (in pixel)')
    parser.add_argument('-step', dest='step', required=False, type=float,
                        default=5, metavar=5,
                        help='step for ridgline detection (in pixel)')
    parser.add_argument('-smooth', dest='smooth', required=False, type=float,
                        default=5, metavar=5,
                        help='smoothing for ridgline detection (in pixel)')
    parser.add_argument('-o', dest='out_data', required=False, type=str,
                        default=None, metavar='None',
                        help='output file name; None-> same as the input')
    parser.add_argument('--noplot', dest='plot_fig', required=False,
                        action='store_true',
                        help='do not plot the image')
    parser.add_argument('-pthresh', dest='plot_thresh', required=False,
                        type=float, default=5.0, metavar=5.0,
                        help='factor*<noise> as lowest level to show the images')
    parser.add_argument('-pw', dest='plot_window', required=False, type=str,
                        default='[2,-2,-2,2]', metavar='[2,-2,-2,2]',
                        help='xyrange for image plotting')
    args = parser.parse_args()


    if args.out_data == 'None':
        args.out_data = None

    onesided = not args.twosided
    plot_fig = not args.plot_fig

    get_ridgeline(args.fitsname,
                core = eval(args.core),
                method = args.method,
                onesided = onesided,
                pa = args.pa,
                dpa = args.dpa,
                dpa_iter = args.dpa_iter,
                noise = args.noise,
                detect_thresh = args.detect_thresh,
                min_radius = args.min_radius,
                max_radius = args.max_radius,
                step = args.step,
                smooth = args.smooth,
                out_data = args.out_data,
                plot_fig = plot_fig,
                plot_thresh = args.plot_thresh,
                plot_window = eval(args.plot_window))


def ridgeline_init(imgobj, core, method, pa, dpa, dpa_iter,
                   noise, detect_thresh, min_radius, max_radius, step):

    Pa = deg2rad(pa)
    Dpa = deg2rad(dpa)
    Dpa_iter = deg2rad(dpa_iter)

    _core = core*1.0
    if method == 'peak':
        _core[0] += step*sin(Pa)*2
        _core[1] -= step*cos(Pa)*2
    pdata, rcrd, tcrd = polar.reproject_image_into_polar(imgobj.data,
                                                         dr=0.708,
                                                         origin=_core)
    NR, NT = pdata.shape
    if not max_radius:
        max_radius = rcrd.max()

    ridge_r, ridge_t = [], []
    for r in range(int(min_radius), int(max_radius), step):
        bflt = (rcrd>r)*(rcrd<=r+step)
        flt0 = np.fabs(tcrd - Pa) <= Dpa
        flt = bflt*flt0

        nr = pdata[bflt].shape[0]/pdata.shape[0]
        nt = pdata[flt0].shape[0]/pdata.shape[1]

        ##################################################
        # finding peaks
        ##################################################
        if method == 'peak':
            sub_data = np.where(flt*(pdata>noise*detect_thresh), pdata, -999)
            idx = np.unravel_index(sub_data.argmax(), sub_data.shape)
            local_peak = sub_data[idx]
            if local_peak > noise*detect_thresh:
                if rcrd[idx] < 2*step or (Dpa_iter > 0 and \
                    len(ridge_r) > 2 and \
                    abs(ridge_t[-1] - tcrd[idx]) > Dpa_iter):
                        pass
                else:
                    ridge_r.append(rcrd[idx])
                    ridge_t.append(tcrd[idx])

        ##################################################
        # finding equal
        ##################################################
        elif method == 'equal':
            flt *= pdata>= noise*detect_thresh/3.0
            sub_data = np.where(flt, pdata, np.NAN)
            sub_data = np.nansum(sub_data, axis=0)
            halfsum = np.sum(sub_data)/2.0
            n = len(sub_data[sub_data>0])
            if halfsum < noise*detect_thresh*step:
                continue
            subsum = 0
            for i in range(len(sub_data)):
                if sub_data[i] > noise*detect_thresh:
                    subsum += sub_data[i]
                if subsum >= halfsum:
                    break
            subsum = 0
            tmp_data = sub_data[::-1]
            for j in range(len(tmp_data)):
                if tmp_data[j] > noise*detect_thresh:
                    subsum += tmp_data[j]
                if subsum >= halfsum:
                    break

#            if i != NT-j-1:
#                print(i-(NT-j)+1)
            idx = (i+NT-j+1)/2
            if Dpa_iter>0 and \
                len(ridge_r)>1 and \
                abs(ridge_t[-1] - tcrd[0][idx]) > Dpa_iter:
                    pass
            else:
                ridge_r.append(r+step/2)
                ridge_t.append(tcrd[0][idx])

        else:
            print('invalid method, should be either "equal" or "peak".')

    ridge_r, ridge_t = np.array(ridge_r), np.array(ridge_t)
    ridge_a, ridge_b = polar.polar2cart(ridge_r, ridge_t, _core)
    ridge_x, ridge_y = imgobj.pix2world(arr([ridge_a, ridge_b]).transpose(),
                                        1).transpose()
    if min_radius == 0 and method == 'peak':
        ridge_x[0] = 0
        ridge_y[0] = 0
    ridge_a, ridge_b = imgobj.world2pix(arr([ridge_x, ridge_y]).transpose(),
                                        1).transpose()
    ridge_r, ridge_t = polar.cart2polar(ridge_a, ridge_b, core)

    # r,t: polar plaen; x, y: physical plane; a, b: image plane
    return ridge_r, ridge_t, ridge_x, ridge_y, ridge_a, ridge_b


def get_ridgeline(infits,
                  core = None,
                  method = 'peak',
                  onesided = True,
                  pa = 0,
                  dpa = 30,
                  dpa_iter = 60,
                  noise = 0,
                  detect_thresh = 3,
                  min_radius = 0,
                  max_radius = 0,
                  step = 5,
                  smooth = 5,
                  out_data = None,
                  plot_fig = True,
                  plot_thresh = 3,
                  plot_window = None):

    image = fits_utils.fits_image(infits)
    image.img_load('mas')
    pixscale = image.wcs.wcs.cdelt[1]*3600*1e3

    if not core:
        if method == 'peak':
            core = list(np.unravel_index(image.data.argmax(), image.data.shape))
            core.reverse()
            core = arr(core)
        elif method == 'equal':
            core = image.wcs.wcs.crpix
    else:
        core = arr(core)

    if not noise:
        image.img_noise()
        noise = image.noise

    if onesided == True: palist = [pa]
    else: palist = [pa, pa-180 if pa>=0 else pa+180]
    ridge_r, ridge_t = arr([]), arr([])
    ridge_x, ridge_y = arr([]), arr([])
    ridge_a, ridge_b = arr([]), arr([])
    ridge_lr, ridge_lt = arr([]), arr([])
    ridge_lx, ridge_ly = arr([]), arr([])
    ridge_la, ridge_lb = arr([]), arr([])

    switch = 0
    for pal in palist:
        _ridge_r, _ridge_t, _ridge_x, _ridge_y, _ridge_a, _ridge_b = \
        ridgeline_init(image, core, method, pal, dpa, dpa_iter,
                       noise, detect_thresh, min_radius, max_radius, step)

        #######################################################
        # smooth the ridgeline in polar coordinate.
        # oscillations sometimes occurs in the low SNR region,
        # so the smooth starts from the core.
        #######################################################
        npt = int(smooth/2)
        rt = np.convolve(_ridge_t[1:],
                1.0*np.ones(smooth,)/smooth, mode='valid')
        rr = _ridge_r[npt+1:-npt]
        rr = np.insert(rr, 0, _ridge_r[0])
        rr = np.append(rr, _ridge_r[-1])
        rt = np.insert(rt, 0, np.mean(rt[1:1+int(smooth/2)]))
        rt = np.append(rt, rt[-1])

        fsmooth = interp1d(rr, rt, kind='cubic')
        _ridge_lr = np.arange(_ridge_r.min(), _ridge_r.max(), step/2.0)
        _ridge_lt = fsmooth(_ridge_lr)
        _ridge_la, _ridge_lb = polar.polar2cart(_ridge_lr, _ridge_lt, core)
        _ridge_lx, _ridge_ly = \
        image.pix2world(arr([_ridge_la, _ridge_lb]).transpose(), 1).transpose()

        if not switch:
            ridge_r = np.append(ridge_r, _ridge_r)
            ridge_t = np.append(ridge_t, _ridge_t)
            ridge_x = np.append(ridge_x, _ridge_x)
            ridge_y = np.append(ridge_y, _ridge_y)
            ridge_a = np.append(ridge_a, _ridge_a)
            ridge_b = np.append(ridge_b, _ridge_b)
            ridge_lr = np.append(ridge_lr, _ridge_lr)
            ridge_lt = np.append(ridge_lt, _ridge_lt)
            ridge_lx = np.append(ridge_lx, _ridge_lx)
            ridge_ly = np.append(ridge_ly, _ridge_ly)
            ridge_la = np.append(ridge_la, _ridge_la)
            ridge_lb = np.append(ridge_lb, _ridge_lb)
        else:
            ridge_r = np.append(_ridge_r[::-1], ridge_r)
            ridge_t = np.append(_ridge_t[::-1], ridge_t)
            ridge_x = np.append(_ridge_x[::-1], ridge_x)
            ridge_y = np.append(_ridge_y[::-1], ridge_y)
            ridge_a = np.append(_ridge_a[::-1], ridge_a)
            ridge_b = np.append(_ridge_b[::-1], ridge_b)
            ridge_lr = np.append(_ridge_lr[::-1], ridge_lr)
            ridge_lt = np.append(_ridge_lt[::-1], ridge_lt)
            ridge_lx = np.append(_ridge_lx[::-1], ridge_lx)
            ridge_ly = np.append(_ridge_ly[::-1], ridge_ly)
            ridge_la = np.append(_ridge_la[::-1], ridge_la)
            ridge_lb = np.append(_ridge_lb[::-1], ridge_lb)

        switch += 1

    if not out_data:
        outname = os.path.splitext(os.path.basename(infits))[0]
    else:
        outname = os.path.splitext(out_data)[0]

    rw = (ridge_x**2 + ridge_y**2)**0.5
    outf = open(outname+'.ridge.txt', 'w')
    print('%s%9s %10s %10s %8s %8s  %8s' \
            %('#', 'ridge_x', 'ridge_y', 'r', 'PA', 'ridge_a', 'ridge_b'),\
            file=outf)
    for (a, b, c, d, e, f) in zip(ridge_x, ridge_y, rw,
                                  rad2deg(ridge_t), ridge_a, ridge_b):
        print('%10.6f %10.6f %10.6f %8.2f %8.1f %8.1f' \
                %(a, b, c, d, e, f), file=outf)
    outf.close()

    lrw = (ridge_lx**2 + ridge_ly**2)**0.5
    outf = open(outname+'.ridge_smooth.txt', 'w')
    print('%s%9s %10s %10s %8s %8s  %8s' \
        %('#', 'ridge_sx', 'ridge_sy', 'r', 'sPA', 'ridge_sa', 'ridge_sb'),\
        file=outf)
    for (a, b, c, d, e, f) in zip(ridge_lx, ridge_ly, lrw,
                                  rad2deg(ridge_lt), ridge_la, ridge_lb):
        print('%10.6f %10.6f %10.6f %8.2f %8.1f %8.1f' \
                %(a, b, c, d, e, f), file=outf)
    outf.close()

    ##################################################
    # plot the image and ridgeline
    ##################################################
    if plot_fig == True:
        ny, nx = image.data.shape
        xpix, ypix = np.arange(nx), np.arange(ny)
        xpix, ypix = np.meshgrid(xpix, ypix)

        image.img_ext()
        xcrd = np.linspace(image.xlim[0], image.xlim[1], image.dim[0])
        ycrd = np.linspace(image.ylim[0], image.ylim[1], image.dim[1])
        xcrd, ycrd = np.meshgrid(xcrd, ycrd)

        # plot image in wcs
        snr = image.peak/noise
        levs = np.linspace(log10(plot_thresh), log10(snr), 10)
        levs[-1] -= 0.2*(levs[1]-levs[0])
        levs = 10**levs
        levs = arr(levs)*noise

        fig = plt.figure(figsize=(5, 5.4))
        ax = fig.add_subplot(111, rasterized=True, aspect='equal')
        img = plt.pcolormesh(xcrd, ycrd, image.data, cmap=plt.cm.jet,
            norm=matplotlib.colors.SymLogNorm(
                linthresh=-1.2*image.data.min(),
                vmin = plot_thresh*noise,
                vmax = 1.1*image.data.max()))

        plt.contour(xcrd, ycrd, image.data,
                linewidths=0.8,
                levels=levs,
                colors='w')

        ax.scatter(ridge_lx, ridge_ly, color='yellow', s=10, zorder=3)
        ax.scatter(0, 0, color='k', s=120, marker='+', zorder=3)
        ax.plot(ridge_lx, ridge_ly, 'w-', zorder=2, linewidth=2)  # smoothed

        ax.set_xlabel('relative RA (mas)')
        ax.set_ylabel('relative Dec (mas)')
        if plot_window:
            ax.set_xlim(plot_window[0], plot_window[1])
            ax.set_ylim(plot_window[2], plot_window[3])

        # plot the colorbar
        minor_loc = LogLocator(base=10.0, subs=(1.0,))
        idx0 = int(np.floor(np.log10(8*noise)))
        idx1 = int(np.ceil(np.log10(image.peak)))+1
        idxs = np.arange(idx0, idx1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(img, cax=cax, ticks=10.0**idxs,
                orientation='vertical')
        cbar.ax.yaxis.set_minor_locator(minor_loc)
        cbar.set_label('Jy/beam (log)')

        plt.tight_layout()
        plt.savefig(outname+'_wcs.eps', bbox_inches='tight')
        ###################### wcs end ############################

        # plot image in cart. coord.
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, rasterized=True, aspect='equal')
        img = ax.pcolormesh(xpix, ypix, image.data, cmap=plt.cm.jet,
            norm=matplotlib.colors.SymLogNorm(
                linthresh=-1.2*image.data.min(),
                vmin = plot_thresh*noise,
                vmax = 1.1*image.data.max()))

        plt.contour(xpix, ypix, image.data,
                linewidths=0.8,
                levels=levs,
                colors='w')

        ax.scatter(ridge_a, ridge_b, color='yellow', s=10, zorder=3)
        ax.scatter(core[0], core[1], color='k', s=120, marker='+', zorder=3)
        ax.plot(ridge_la, ridge_lb, 'w-', zorder=2, linewidth=2)  # smoothed
        plt.xlabel('X (pixel)')
        plt.ylabel('Y (pixel)')
        if plot_window:
            win_cart = [[plot_window[0], plot_window[2]], \
                        [plot_window[1], plot_window[3]]]
            win_cart = image.world2pix(win_cart).transpose()
            plt.xlim(win_cart[0])
            plt.ylim(win_cart[1])
        plt.tight_layout()
        plt.savefig(outname+'_cart.eps', bbox_inches='tight')
        ###################### cart. end ############################

        # plot image with polar coord.
        pdata, rcrd, tcrd = polar.reproject_image_into_polar(image.data,
                                                         dr=0.708,
                                                         origin=core)
        polar2wcs = (image.dim[0]*image.dim[1])**0.5/len(rcrd)*pixscale
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, rasterized=True)
        img = ax.pcolormesh(rad2deg(tcrd), rcrd*polar2wcs, pdata,
                cmap=plt.cm.jet,
                norm = matplotlib.colors.SymLogNorm(
                linthresh=-1.2*pdata.min(),
                vmin = plot_thresh*noise,
                vmax = 1.1*pdata.max()))

        ax.scatter(rad2deg(ridge_t), ridge_r*polar2wcs, color='yellow', s=10)
        ax.plot(rad2deg(ridge_lt), ridge_lr*polar2wcs, 'w-', zorder=2, linewidth=2)
        plt.xlabel('PA (deg.)')
        plt.ylabel('r (mas)')
        plt.xlim(-180, 180)
        if plot_window:
            if onesided == True:
                plt.xlim(pa-dpa, pa+dpa)
            plt.ylim(0.7*min_radius, max(ridge_lr*polar2wcs)*1.5)
        plt.tight_layout()
        plt.savefig(outname+'_polar.eps', bbox_inches='tight')
        plt.close('all')

__all__ = ['get_ridgeline']

if __name__ == '__main__':

    main()
    # get_ridgeline('fitsfile',
    #               core = None,
    #               method = 'equal',
    #               onesided = True,
    #               pa = 45,
    #               dpa = 90,
    #               dpa_iter = 20,
    #               noise = 0.0003,
    #               detect_thresh = 5,
    #               min_radius = 0,
    #               max_radius = 160,
    #               step = 5,
    #               smooth = 5,
    #               out_data = None,
    #               plot_fig = True,
    #               plot_thresh = 3,
    #               plot_window = [2, -2, -1.4, 2.4])

