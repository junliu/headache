#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import argparse
import numpy as np
from numpy import array as arr
from numpy import rad2deg, deg2rad, sin, cos, log, log10
import astropy
from astropy.io import fits as pyfits
from astropy.wcs import WCS
from astropy import units
from astropy.modeling.functional_models import Gaussian2D
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import MultipleLocator, LogLocator
from matplotlib.patches import Circle, Ellipse, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rcParams
try:
    from colorspacious import cspace_converter as csc
    CSC_EXIST = 1
except:
    CSC_EXIST = 0

rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['text.usetex'] = True
if matplotlib.__version__ > '2.0.0':
    rcParams['xtick.top'] = True
    rcParams['ytick.right'] = True

def config_parser(fconfig):

    with open(fconfig) as f:
        lines = f.readlines()
    lines = [l.rstrip().lstrip() for l in lines]
    lines = [l for l in lines if not (l.startswith('#') or len(l)==0)]
    lines = [l.split('#')[0].rstrip().lstrip() for l in lines]

    for i in range(len(lines)-1, -1, -1):
        if (lines[i][-1] == ',') or (not lines[i].split()[0].isupper()):
            lines[i-1] += ' '+lines[i]
            lines[i] = ''

    except_keys = ['TITLE', 'XLABEL', 'YLABEL', 'ANNOT']
    lines = [re.sub(',', ' ', l).split() if l.split()[0] not in except_keys \
            else l.split() for l in lines if len(l)>0]
    keys = [l[0] for l in lines]
    vals = [l[1:] if len(l) > 2 else l[1] for l in lines]

    conf = dict(list(zip(keys, vals)))
    for key in except_keys:
        if type(conf[key]) == list:
            conf[key] = ' '.join(conf[key])

    for key in conf:
        val = conf[key]
        if type(val) == str:
            if val.lower() == 'none':
                conf[key] = None
            elif val.lower() == 'false':
                conf[key] = False

    conf['POL_VEC'][:3] = [float(v) for v in conf['POL_VEC'][:3]]
    conf['POL_VEC'][3:] = [int(v) for v in conf['POL_VEC'][3:]]
    conf['FIX_VEC_LEN'] = float(conf['FIX_VEC_LEN'])
    conf['EVPA'] = float(conf['EVPA'])
    conf['FITS_DIM'] = [int(v) for v in conf['FITS_DIM']]
    if conf['CRPIX'].lower() == 'auto':
        conf['CRPIX'] = [v*0.5 for v in conf['FITS_DIM']]
    conf['ROTATE'] = float(conf['ROTATE'])
    conf['PLOT_STYLE'] = int(conf['PLOT_STYLE'])
    conf['NOISE_THRESH'] = float(conf['NOISE_THRESH'])
    conf['LIN_THRESH'] = float(conf['LIN_THRESH'])
    conf['DATA_UNITS'] = conf['DATA_UNITS'].upper()
    conf['MAP_FUNC'] = conf['MAP_FUNC'].lower()
    conf['CONT_FUNC'] = conf['CONT_FUNC'].lower()
    conf['LEV_PARAM'] = [float(v) for v in conf['LEV_PARAM']]
    conf['BEAM'] = [float(v) for v in conf['BEAM']]
    conf['BEAM_LOC'] = conf['BEAM_LOC'].lower()
    conf['CBAR_LOC'] = conf['CBAR_LOC'].lower()
    conf['FIG_SIZE'] = [float(v) for v in conf['FIG_SIZE']]
    conf['LINEWIDTH'] = float(conf['LINEWIDTH'])
    conf['FONTSIZE'] = int(conf['FONTSIZE'])
    conf['ANNOT_LOC'] = conf['ANNOT_LOC'].lower()

    if conf['XLABEL'].lower() == 'auto':
        conf['XLABEL'] = 'Relative RA (%s)' %conf['MAP_UNITS']
    if conf['YLABEL'].lower() == 'auto':
        conf['YLABEL'] = 'Relative Dec (%s)' %conf['MAP_UNITS']

    return conf


def line_creator(cpt, angle, length):

    xlen = length*0.5*cos(angle)
    ylen = length*0.5*sin(angle)
    xx = [cpt[0] - xlen, cpt[0] + xlen]
    yy = [cpt[1] - ylen, cpt[1] + ylen]

    xx = arr(xx).transpose()
    yy = arr(yy).transpose()

    return xx, yy


def get_edgecolor(cmap, vmin, lev0):
    if CSC_EXIST:
        x = np.linspace(0, 1, 100)
        rgb = matplotlib.cm.get_cmap(plt.get_cmap(cmap))(x)[np.newaxis,:,:3]
        lab = csc('sRGB1', 'CAM02-UCS')(rgb)
        L = lab[0,:,0]
        if L[0] > L[-1]: ec = 'k'
        else: ec = 'w'
    else:
        if vmin > lev0: ec = 'k'
        else: ec = 'w'

    return ec

class difmap_model:

    def __init__(self, fconf):


        self.flux = arr([0])
        self.radius = arr([0])
        self.theta = arr([0])
        self.major = arr([0])
        self.axialr = arr([1])
        self.phi = arr([0])

        self.conf = config_parser(fconf)


    def load_mod(self):
        print('****** loading models ******')
        with open(self.conf['MOD_NAME']) as fmodel:
            lines = fmodel.readlines()

        # first remove all the tentative models
        line = [l for l in lines if 'Tentative' in l]
        if len(line) > 0:
            print('tentative models found')
            idx = lines.index(line[0])
            lines = lines[:idx-1]

        cra, cdec = 1.0, 45.0
        center = [l[:-1] for l in lines if 'Center RA' in l]
        if center:
            center = center[0].split(',')
            cra, cdec = [c.split(': ')[1].split() for c in center]
            cra = float(cra[0]) + float(cra[1])/60.0 + float(cra[2])/3600.0
            cra *= 15.0
            cdec = float(cdec[0]) + float(cdec[1])/60.0 + float(cdec[2])/3600.0
            print(('found RA: %0.6f, Dec: %0.6f' %(cra, cdec)))
        self.mod_crval = arr([cra, cdec])
        del center, cra, cdec

        lines = [l[:-1] for l in lines if not l.startswith('!')]
        # check the model types (delta, circular/elliptical Gaussian, or hybrid)
        # self.major = 0 --> delta
        # self.axialr = 1 --> circular Gaussian
        # self.axialr != 1 --> elliptical Gaussian
        model = []
        for line in lines:
            line = [float(l[:-1]) if l.endswith('v') else float(l) \
                    for l in line.split()]
            if len(line) > 6: line = line[:6]
            elif len(line) == 3: line += [0, 1, 45]
            if line[4] == 1: line[5] = 45
            model.append(line)
        print(('collected %d models' %len(model)))
        self.flux, self.radius, self.theta, self.major, \
        self.axialr, self.phi = arr(model).transpose()
        del model

        if self.conf['TEMPLATE'] == None and \
            self.conf['FITS_SIZE'].lower() == 'auto':
            print('calculating image extent')
            x = self.radius*sin(deg2rad(self.theta))
            y = self.radius*cos(deg2rad(self.theta))
            x0, x1 = x-self.major, x+self.major
            y0, y1 = y-self.major, y+self.major
            xdim = int(max(x1) - min(x0))+1.0
            ydim = int(max(y1) - min(y0))+1.0
            self.conf['FITS_SIZE'] = [xdim, ydim]
            print('FITS_SIZE = %0.1f, %0.1f' %(xdim, ydim))

        elif self.conf['TEMPLATE'] == None:
            self.conf['FITS_SIZE'] = [float(v) for v in self.conf['FITS_SIZE']]


    def set_wcs(self):

        # get new attributes: wcs, dim, p2w_scale
        # check crval, crpix value in configure file
        self.load_mod()

        print('****** setting up coordinate system ******')
        if self.conf['CRVAL'].lower() == 'auto':
            self.conf['CRVAL'] = self.mod_crval

        # get the wcs information from the template
        if self.conf['TEMPLATE'] != None:
            print('****** reading from template *****')
            hdu = pyfits.open(self.conf['TEMPLATE'])
            hdr = hdu[0].header
            hdu.close()

            try:
                _wcs = WCS(hdr)
            except astropy.wcs._wcs.SingularMatrixError:
                n = hdr['naxis']
                for i in range(n):
                    if not hdr['cdelt%d' %(i+1)]:
                        hdr['cdelt%d' %(i+1)] = 1
                _wcs = WCS(hdr)

            if hdr['naxis'] > 2:
                wcs = WCS(naxis=2)
                wcs.wcs.ctype = list(_wcs.wcs.ctype)[:2]
                wcs.wcs.crval = _wcs.wcs.crval[:2]
                wcs.wcs.crpix = _wcs.wcs.crpix[:2]
                wcs.wcs.cdelt = _wcs.wcs.cdelt[:2]
                if _wcs.wcs.has_crota():
                    wcs.wcs.crota = _wcs.wcs.crota[:2]
                wcs.wcs.cunit = list(_wcs.wcs.cunit)[:2]
                self.wcs = wcs
            else:
                self.wcs = _wcs

            self.dim = arr([hdr['naxis1'], hdr['naxis2']])

        else:
            self.dim = arr(self.conf['FITS_DIM'])
            self.scale = float(1.0*getattr(units, self.conf['MAP_UNITS'])/units.deg)
            wcs = WCS(naxis=2)
            wcs.wcs.ctype = ['RA---SIN', 'DEC--SIN']
            wcs.wcs.crval = self.conf['CRVAL']
            wcs.wcs.crpix = self.conf['CRPIX']
            wcs.wcs.cdelt = arr(self.conf['FITS_SIZE'])/self.dim*self.scale
            wcs.wcs.cdelt[0] *= -1.0
            wcs.wcs.cunit = ['deg', 'deg']
            self.wcs = wcs

        # check the crval to avoid zero values
        if not self.wcs.wcs.crval[0]:
            self.wcs.wcs.crval[0] = 1e-2
        if not self.wcs.wcs.crval[1]:
            self.wcs.wcs.crval[1] = 1e-2

        self.p2w_scale = arr([1, 1])
        for i in range(2):
            self.p2w_scale[i] = float(1*getattr(units,
                                str(self.wcs.wcs.cunit[i]))/(1*getattr(units,
                                self.conf['MAP_UNITS'])))
        self.p2w_scale[0] *= cos(deg2rad(self.wcs.wcs.crval[1]))

        print('WCS setup done')


    def mod2fits(self):

        self.set_wcs()

        print('****** converting models to FITS ******')
        print('image ROTATE = %0.2f degrees' %self.conf['ROTATE'])
        theta = deg2rad(self.theta - self.conf['ROTATE'])
        minor = self.major*self.axialr
        lenax = arr([self.major, minor]).transpose()
        modx, mody = self.radius*sin(theta), self.radius*cos(theta)

        # get xy grids in wcs
        pixcrd = arr([arr([0, 0]), self.dim-1], np.float_)
        xlim, ylim = self.pix2world(pixcrd).transpose()

        x = np.linspace(xlim[0], xlim[1], self.dim[0])
        y = np.linspace(ylim[0], ylim[1], self.dim[1])
        x, y = np.meshgrid(x, y)
        data = np.zeros(x.shape, np.float32)
        fbeam = Gaussian2D()
        beam = self.conf['BEAM']

        print('****** convolving with beam %0.2fx%0.2f, PA %0.2f ******' \
            %(beam[0], beam[1], beam[2]))

        for i in range(len(theta)):
            # for clean (delta) models
            if self.major[i] == 0:
                data += fbeam.evaluate(x, y, self.flux[i], modx[i], mody[i],
                    beam[0]/(8*log(2))**0.5, beam[1]/(8*log(2))**0.5,
                    -deg2rad(beam[2]+90))
            # for circular or elliptical Gaussian models
            else:
                data += fbeam.evaluate(x, y, self.flux[i], modx[i], mody[i],
                    self.major[i]/(8*log(2))**0.5, minor[i]/(8*log(2))**0.5,
                    deg2rad(self.phi[i]+90))
        print('convolution done')

        # write to fits image
        if os.path.isfile(self.conf['OUT_FITS']):
            os.remove(self.conf['OUT_FITS'])
        scale = float(1.0*getattr(units, self.conf['MAP_UNITS'])/units.deg)
        hdr = self.wcs.to_header()

        hdr['date-obs'] = self.conf['DATE_OBS']
        hdr['epoch'] = 2000.0
        hdr['object'] = self.conf['SRC_NAME']
        hdr['obsra'] = self.mod_crval[0]
        hdr['obsdec'] = self.mod_crval[1]
        hdr['bunit'] = self.conf['DATA_UNITS']
        hdr['instrume'] = 'MODEL'
        hdr['bmaj'] = beam[0]*scale
        hdr['bmin'] = beam[1]*scale
        hdr['bpa'] = beam[2]

        hdr.comments['date-obs'] = 'Observation date'
        hdr.comments['epoch'] = 'Equinox of coordinate'
        hdr.comments['object'] = 'Name of observed source'
        hdr.comments['obsra'] = 'Antenna pointing RA'
        hdr.comments['obsdec'] = 'Antenna pointing Dec'
        hdr.comments['bunit'] = 'Unit of measurements'
        hdr.comments['instrume'] = 'Instrument used'
        hdr.comments['bmaj'] = 'Clean beam major axis diameter (deg)'
        hdr.comments['bmin'] = 'Clean beam minor axis diameter (deg)'
        hdr.comments['bpa'] = 'Clean beam position angle (deg)'

        pyfits.writeto(self.conf['OUT_FITS'], data, hdr)
        print('written to %s' %self.conf['OUT_FITS'])


    def pix2world(self, pixcrd, start=0):

        wcrd = self.wcs.wcs_pix2world(pixcrd, start)
        wcrd -= self.wcs.wcs.crval
        wcrd *= self.p2w_scale

        return wcrd


    def world2pix(self, wcscrd, start=0):

        _wcscrd = arr(wcscrd)/self.p2w_scale
        pcrd = self.wcs.wcs_world2pix(_wcscrd + self.wcs.wcs.crval, start)

        return pcrd


class fits_image(object):

    def __init__(self, fits_name):

        self.fits_name = fits_name

    def img_load(self, map_unit='mas'):
        '''img_load(map_unit='mas')
        load the image data and wcs system
        '''

        print('****** loading data for %s ******' %self.fits_name)
        hdu = pyfits.open(self.fits_name)
        self.data = hdu[0].data
        self.hdr = hdu[0].header
        hdu.close()

        self.ndim = len(self.data.shape)
        if self.ndim > 2:
            self.data = self.data.reshape(self.data.shape[-2:])
        self.dim = arr(self.data.shape)[::-1]

        self.img_wcs()
#        self.img_noise()
        self.peak = self.data.max()

        # scaling factor from pixel to wcs
        self.p2w_scale = arr([1.0, 1.0])
        for i in range(2):
            try:
                self.p2w_scale[i] = float(1*getattr(units,
                                str(self.wcs.wcs.cunit[i]))/(1*getattr(units,
                                map_unit)))
            except AttributeError:
                print('Warning: missing WCS in FITS header!!!')
        self.p2w_scale[0] *= np.cos(np.deg2rad(self.wcs.wcs.crval[1]))


    def img_wcs(self):

        # FITS header sanity check for WCS
        print('****** loading WCS for %s ******' %self.fits_name)
        try:
            _wcs = WCS(self.hdr)
        except astropy.wcs._wcs.SingularMatrixError:
            n = self.hdr['naxis']
            for i in range(n):
                if not self.hdr['cdelt%d' %(i+1)]:
                    self.hdr['cdelt%d' %(i+1)] = 1
            _wcs = WCS(self.hdr)

        if len(_wcs.wcs.ctype) > 2:
            wcs = WCS(naxis=2)
            wcs.wcs.ctype = list(_wcs.wcs.ctype)[:2]
            wcs.wcs.crval = _wcs.wcs.crval[:2]
            wcs.wcs.crpix = _wcs.wcs.crpix[:2]
            wcs.wcs.cdelt = _wcs.wcs.cdelt[:2]
#            wcs.wcs.naxis = _wcs.wcs.naxis[:2]
            if _wcs.wcs.has_crota():
                wcs.wcs.crota = _wcs.wcs.crota[:2]
            wcs.wcs.cunit = list(_wcs.wcs.cunit)[:2]
            self.wcs = wcs
        else:
            self.wcs = _wcs

        # check the crval to avoid zero values
        if not self.wcs.wcs.crval[0]:
            self.wcs.wcs.crval[0] = 0.1
        if not self.wcs.wcs.crval[1]:
            self.wcs.wcs.crval[1] = 0.1


    def img_noise(self):

        print('****** calculating image noise for %s ******' \
            %self.fits_name)
        if 'NOISE' in list(self.hdr.keys()):
            self.noise = self.hdr['noise']
        else:
            # estimate the noise from by iteratively filtering
            # out the image peaks

            # old method, sometimes sensitive and unstable
            # n1, n2 = self.dim
            # flt0 = np.ones(n1*n2, dtype=bool).reshape(n2, n1)
            # while 1:
            #     noise = np.std(self.data[flt0])
            #     _flt = self.data < 3*noise

            #     if np.array_equal(_flt, flt0):
            #         break
            #     else:
            #         flt0 = _flt*1
            # self.noise = noise
            #
            # new method, seems better
            peak = self.data.max()
            flt0 = self.data > 0
            flt1 = self.data < 0
            if not True in flt1:
                noise0 = np.std(self.data)
                while 1:
                    peak*=0.9
                    flt = self.data < peak
                    noise1 = np.std(self.data[flt])
                    if np.fabs(1-noise1/noise0) < 1e-2:
                        break
                    else:
                        noise0 = noise1*1.0
                self.noise = noise1

            else:
                while 1:
                    peak *= 0.95
                    flt = (self.data <= peak)
                    n0 = len(self.data[flt0*flt])
                    n1 = len(self.data[flt1*flt])
                    if n0 <= n1 and np.sum(self.data[flt]) <= 0:
                        break

                self.noise = np.std(self.data[flt])


    def pix2world(self, pixcrd, start=0):

        wcrd = self.wcs.wcs_pix2world(pixcrd, start)
        wcrd -= self.wcs.wcs.crval
        wcrd *= self.p2w_scale

        return wcrd


    def world2pix(self, wcscrd, start=0):

        _wcscrd = arr(wcscrd)/self.p2w_scale
        pcrd = self.wcs.wcs_world2pix(_wcscrd + self.wcs.wcs.crval, start)

        return pcrd


    def img_ext(self, start=0):

        pixcrd = arr([arr([start, start]), self.dim-1+start], np.float_)
        self.xlim, self.ylim = self.pix2world(pixcrd, start).transpose()


class plot_utils:

    def __init__(self, fconf):

        self.fconf = fconf
        self.conf = config_parser(fconf)

    def vec_creator(self, xx, yy, tdata, pdata, pangle, polvec, vec_len, evpa):

        xdn, ydn = polvec[3:5]
        _xx = xx[::ydn][:,::xdn]
        _yy = yy[::ydn][:,::xdn]
        _tdata = tdata[::ydn][:,::xdn]
        _pdata = pdata[::ydn][:,::xdn]
        _pangle = pangle[::ydn][:,::xdn]
        flt = (_tdata>polvec[1])*(_pdata>polvec[2])
        xpol, ypol = _xx[flt], _yy[flt]
        angle = _pangle[flt]
        if vec_len > 0: length = np.ones(len(xpol))*vec_len
        else: length = _pdata[flt]/polvec[0]

        cpt = [xpol, ypol]
        cxx, cyy = line_creator(cpt, deg2rad(90-angle-evpa), length)

        return cxx, cyy


    def run(self):

        # do conversion if it required
        if self.conf['MOD_NAME']:
            self.mod = difmap_model(self.fconf)
            if self.conf['CONVERT']:
                self.mod.mod2fits()
        else:
            if self.conf['CONVERT']:
                print('Error: Model file should be specified.')
                exit(0)

        self.timg = fits_image(self.conf['I_IMAGE'])
        self.timg.img_load(self.conf['MAP_UNITS'])
        self.timg.img_ext()
        self.timg.img_noise()
        print('I_NOISE = %f %s' %(self.timg.noise, self.conf['DATA_UNITS']))
        print('I_PEAK = %f %s' %(self.timg.peak, self.conf['DATA_UNITS']))
        if self.conf['PLOT_STYLE'] > 0:
            if self.conf['P_IMAGE'] and self.conf['PANG_IMAGE']:
                self.pimg = fits_image(self.conf['P_IMAGE'])
                self.paimg = fits_image(self.conf['PANG_IMAGE'])
                self.pimg.img_load(self.conf['MAP_UNITS'])
                self.pimg.img_noise()
                print('P_NOISE = %f %s' \
                    %(self.pimg.noise, self.conf['DATA_UNITS']))
                print('P_PEAK = %f %s' \
                    %(self.pimg.peak, self.conf['DATA_UNITS']))
                self.paimg.img_load(self.conf['MAP_UNITS'])

            elif self.conf['Q_IMAGE'] and self.conf['U_IMAGE']:
                print('****** converting Q/U to P/PA ******')
                qimg = fits_image(self.conf['Q_IMAGE'])
                uimg = fits_image(self.conf['U_IMAGE'])
                qimg.img_load(self.conf['MAP_UNITS'])
                uimg.img_load(self.conf['MAP_UNITS'])

                self.pimg = fits_image('none')
                self.paimg = fits_image('none')

                hdr = qimg.hdr
                if 'NOISE' in list(hdr.keys()):
                    del hdr['NOISE']
                self.pimg.hdr = hdr
                self.pimg.data = (qimg.data**2 + uimg.data**2)**0.5
                self.pimg.dim = arr(self.pimg.data.shape)[::-1]
                self.pimg.ndim = hdr['naxis']
                self.pimg.img_noise()
                self.pimg.img_wcs()
                self.pimg.peak = self.pimg.data.max()
                self.paimg.data = rad2deg(0.5*np.arctan2(uimg.data,
                                                         qimg.data))

                del qimg, uimg
                print('P_NOISE = %f %s' \
                    %(self.pimg.noise, self.conf['DATA_UNITS']))
                print('P_PEAK = %f %s' \
                    %(self.pimg.peak, self.conf['DATA_UNITS']))

        # speed up plotting when image is large
        if self.timg.dim[0]*self.timg.dim[1] > 1e6:
            plot_func = plt.pcolormesh
        else:
            plot_func = plt.pcolor

        xx = np.linspace(self.timg.xlim[0], self.timg.xlim[1], self.timg.dim[0])
        yy = np.linspace(self.timg.ylim[0], self.timg.ylim[1], self.timg.dim[1])
        xx, yy = np.meshgrid(xx, yy)

        if type(self.conf['LEVS']) == str and self.conf['LEVS'].lower() == 'auto':
            if self.conf['PLOT_STYLE'] <= 1:
                _noise = self.timg.noise
                _peak = self.timg.peak
            elif self.conf['PLOT_STYLE'] > 1:
                _noise = self.pimg.noise
                _peak = self.pimg.peak

            lev0 = self.conf['LEV_PARAM'][0]
            lev1 = self.conf['LEV_PARAM'][1]
            lev_incre = self.conf['LEV_PARAM'][2]
            if lev0 == 0: lev0 = 3.0*_noise
            if lev1 == 0 or lev1 > _peak: lev1 = _peak

            if 'log' in self.conf['CONT_FUNC']:
                nlev = np.log(lev1/lev0)/np.log(lev_incre)
                nlev = int(nlev) + 2
                levs = np.logspace(0, nlev-2, nlev-1, base=lev_incre)*lev0
            else:
                nlev = (lev1 - lev0)/lev_incre
                nlev = int(nlev) + 1
                levs = np.arange(0, nlev, 1)*lev_incre + lev0
            lev1 = levs[-1]

            print('****** calculating levels with')
            print('****** noise = %0.2e' %_noise)
            print('******  peak = %0.2e' %_peak)
            print('****** lev0 = %0.2e' %lev0)
            print('****** lev1 = %0.2e' %lev1)
            print('****** level increment = %0.1f' %lev_incre)

            levs = np.insert(levs, 0, -levs[0])
            levs /= _peak/100.0
            strlevs = ['%0.1e' %l if abs(l)<1 else '%0.1f' %l for l in levs]
            strlevs = ', '.join(strlevs)
            self.conf['LEVS'] = levs
            print('LEVS (%%) = %s' %strlevs)

        else:
            self.conf['LEVS'] = np.array([float(v) for v in self.conf['LEVS']])

        VMIN, VMAX = self.conf['IMG_MIN'], self.conf['IMG_MAX']
        if VMIN.lower() == 'auto': VMIN = 'auto'
        else: VMIN = float(VMIN)
        if VMAX.lower() == 'auto': VMAX = 'auto'
        else: VMAX = float(VMAX)

        if type(self.conf['XY_RANGE']) == list:
            self.conf['XY_RANGE'] = [float(v) for v in self.conf['XY_RANGE']]
            if self.conf['XY_RANGE'][0] < self.conf['XY_RANGE'][1]:
                self.conf['XY_RANGE'][:2] = self.conf['XY_RANGE'][:2][::-1]
            if self.conf['XY_RANGE'][2] > self.conf['XY_RANGE'][3]:
                self.conf['XY_RANGE'][2:] = self.conf['XY_RANGE'][2:][::-1]
        else:
            self.conf['XY_RANGE'] = list(self.timg.xlim) + list(self.timg.ylim)

        rcParams.update({'lines.linewidth': self.conf['LINEWIDTH']})
        rcParams.update({'font.size': self.conf['FONTSIZE']})
        fig = plt.figure(figsize=self.conf['FIG_SIZE'])
        ax = fig.add_subplot(111, rasterized=True, aspect='equal')
        ax.set_xlabel(self.conf['XLABEL'])
        ax.set_ylabel(self.conf['YLABEL'])
        ax.set_xlim(self.conf['XY_RANGE'][:2])
        ax.set_ylim(self.conf['XY_RANGE'][2:])

        if self.conf['PLOT_STYLE'] == 1 or self.conf['PLOT_STYLE'] == 2:
            cmap = 'gnuplot2_r'
            ec = 'k'
            # plot the polarizations (raster)
            if  self.conf['MAP_FUNC'] == 'log10':
                if VMIN == 'auto': VMIN = 5.0
                if VMAX == 'auto': VMAX = 10.0
                linthresh = self.conf['LIN_THRESH']*self.pimg.noise
                if linthresh + self.pimg.data.min() < 0:
                    linthresh = -self.pimg.data.min()*1.05

                img = plot_func(xx, yy, self.pimg.data, \
                cmap = plt.get_cmap(cmap), \
                norm = matplotlib.colors.SymLogNorm(
                linthresh = linthresh,
                vmin = VMIN*self.pimg.noise,
                vmax = VMAX*self.pimg.peak))

            elif self.conf['MAP_FUNC'] == 'linear':
                if VMIN == 'auto': VMIN = 5.0
                if VMAX == 'auto': VMAX = 1.2
                img = plot_func(xx, yy, self.pimg.data, \
                cmap = plt.get_cmap(cmap), \
                vmin = -VMIN*self.pimg.noise,
                vmax = VMAX*self.pimg.peak)

            if self.conf['PLOT_STYLE'] == 1:
                # plot the total intensity (contour plot)
                plt.contour(xx, yy, self.timg.data, \
                    linewidths = 0.8*self.conf['LINEWIDTH'], \
                    levels=self.conf['LEVS']*self.timg.peak/100, colors=ec)
            else:
                # lowest level contour of total intensity
                p2i_lev0 = self.conf['LEVS'][1]*self.pimg.peak/100
                p2i_lev0 /= self.pimg.noise
                plt.contour(xx, yy, self.timg.data, \
                        linewidths = 1.2*self.conf['LINEWIDTH'], \
                        levels = p2i_lev0*self.timg.noise,
                        linestyles = '--',
                        colors = 'b')

        if self.conf['PLOT_STYLE'] == 2 or self.conf['PLOT_STYLE'] == 3:
            ec = 'k' if self.conf['PLOT_STYLE'] == 2 else 'w'
            lw = 0.8 if self.conf['PLOT_STYLE'] == 2 else 1.2
            tmp_lev0 = self.pimg.noise/self.pimg.peak*100*VMIN
            tmp_levs = arr(self.conf['LEVS'])
            tmp_levs = tmp_levs[tmp_levs>tmp_lev0]
            plt.contour(xx, yy, self.pimg.data, \
                linewidths = lw*self.conf['LINEWIDTH'], \
                levels=tmp_levs*self.pimg.peak/100, colors=ec)

        if self.conf['PLOT_STYLE'] == 3:
            cmap = 'jet'
            ec = 'w'
             # plot the total intensity map (raster)
            if  self.conf['MAP_FUNC'] == 'log10':
                if VMIN == 'auto': VMIN = 5.0
                if VMAX == 'auto': VMAX = 10.0
                linthresh = self.conf['LIN_THRESH']*self.timg.noise
                if linthresh + self.timg.data.min() < 0:
                    linthresh = -self.timg.data.min()*1.05
                img = plot_func(xx, yy, self.timg.data, \
                cmap = plt.get_cmap(cmap), \
                norm = matplotlib.colors.SymLogNorm(
                linthresh = linthresh,
                vmin = -VMIN*self.timg.noise,
                vmax = VMAX*self.timg.peak/100))

            elif self.conf['MAP_FUNC'] == 'linear':
                if VMIN == 'auto': VMIN = 5.0
                if VMAX == 'auto': VMAX = 1.2
                img = plot_func(xx, yy, self.timg.data, \
                cmap = plt.get_cmap(cmap), \
                vmin = -VMIN*self.timg.noise,
                vmax = VMAX*self.timg.peak)

        if self.conf['PLOT_STYLE'] > 0:
            vxx, vyy = self.vec_creator(xx, yy, self.timg.data,
                                        self.pimg.data, self.paimg.data,
                                        self.conf['POL_VEC'],
                                        self.conf['FIX_VEC_LEN'],
                                        self.conf['EVPA'])

            for (xl, yl) in zip(vxx, vyy):
                plt.plot(xl, yl, 'w-', lw=2,
                    path_effects=[pe.Stroke(linewidth=3*self.conf['LINEWIDTH'],
                    foreground='k'), pe.Normal()])

        # in case of total intensity only
        else:
            if self.conf['COLOR_STYLE']:
                cmap = self.conf['COLOR_STYLE'].lower()
                if  self.conf['MAP_FUNC'] == 'log10':
                    if VMIN == 'auto': VMIN = 5.0
                    if VMAX == 'auto': VMAX = 10.0
                    linthresh = self.conf['LIN_THRESH']*self.timg.noise
                    if linthresh + self.timg.data.min() < 0:
                        linthresh = -self.timg.data.min()*1.05
                    img = plot_func(xx, yy, self.timg.data, \
                    cmap = plt.get_cmap(cmap), \
                    norm = matplotlib.colors.SymLogNorm(
                    linthresh = linthresh,
                    vmin = -VMIN*self.timg.noise,
                    vmax = VMAX*self.timg.peak/100))

                elif self.conf['MAP_FUNC'] == 'linear':
                    if VMIN == 'auto': VMIN = 5.0
                    if VMAX == 'auto': VMAX = 1.2
                    img = plot_func(xx, yy, self.timg.data, \
                    cmap = plt.get_cmap(cmap), \
                    vmin = -VMIN*self.timg.noise,
                    vmax = VMAX*self.timg.peak)
                ec = get_edgecolor(cmap, VMIN*self.timg.noise,
                                   self.conf['LEVS'][1]*self.timg.peak/100)
            else:
                ec = 'k'

            # plot contours
            if self.conf['SHOW_CONT']:
                plt.contour(xx, yy, self.timg.data, \
                    linewidths = 0.8*self.conf['LINEWIDTH'], \
                    levels=self.conf['LEVS']*self.timg.peak/100, colors=ec)

            # plot Gaussian models
            if self.conf['SHOW_MOD']:
                amp = self.mod.flux
                radius = self.mod.radius
                theta = deg2rad(self.mod.theta)
                major = self.mod.major
                minor = self.mod.major*self.mod.axialr

                flt = major > 0
                amp = amp[flt]
                radius = radius[flt]
                theta = theta[flt]
                major = major[flt]
                minor = minor[flt]

                mx = radius*np.sin(theta)
                my = radius*np.cos(theta)

                for i in range(len(mx)):
                    model = Circle((mx[i], my[i]), major[i]/2,
                                    ec=ec, fc='none', linewidth=1)
                    px = np.array([-major[i], major[i]])*2**0.5/4+mx[i]
                    py = np.array([-major[i], major[i]])*2**0.5/4+my[i]
                    plt.plot(px, py, '%s-' %ec, lw=self.conf['LINEWIDTH'])
                    plt.plot(px, np.flip(py,0), '%s-' %ec,
                             lw=1.2*self.conf['LINEWIDTH'])

                    ax.add_artist(model)

        # plot the restoring beam
        bmax, bmin, bpa = self.conf['BEAM']
        bpa = deg2rad(bpa)
        rwidth = ((bmax*sin(bpa))**2 + (bmin*cos(bpa))**2)**0.5
        rheight = ((bmax*cos(bpa))**2 + (bmin*sin(bpa))**2)**0.5
        dw = rwidth if rwidth > rheight else rheight
        rwidth += 0.5*dw
        rheight += 0.5*dw

        xy_range = self.conf['XY_RANGE']
        dx = xy_range[1] - xy_range[0]
        dy = xy_range[3] - xy_range[2]
        if 'lower' in self.conf['BEAM_LOC']:
            starty = xy_range[2]
        elif 'upper' in self.conf['BEAM_LOC']:
            starty = xy_range[3]
            rheight = -rheight
            dy = -dy
        if 'left' in self.conf['BEAM_LOC']:
            startx = xy_range[0]
            rwidth = -rwidth
        elif 'right' in self.conf['BEAM_LOC']:
            startx = xy_range[1]
            dx = -dx

        recx = startx + dx*0.025
        recy = starty + dy*0.025
        bx = recx + rwidth*0.5
        by = recy + rheight*0.5

        rec = Rectangle([recx, recy], rwidth, rheight, ec=ec, fill=False)
        if self.conf['PLOT_STYLE'] > 0 or self.conf['COLOR_STYLE'] != None:
            fc = 'b'
        else: fc = 'none'
        beam = Ellipse((bx, by), bmin, bmax, -rad2deg(bpa), ec=ec, fc=fc)
        ax.add_artist(rec)
        ax.add_artist(beam)

        # plot the colorbar
        if self.conf['PLOT_STYLE'] > 0 or self.conf['COLOR_STYLE'] != None:
            cbarloc = self.conf['CBAR_LOC'].lower()
            divider = make_axes_locatable(ax)
            if cbarloc == 'top' or cbarloc == 'bottom':
                orientation='horizontal'
            else:
                orientation='vertical'
            if cbarloc == 'left' or cbarloc == 'bottom':
                pad = 0.55*self.conf['FONTSIZE']/12.0
                if cbarloc == 'left' and xy_range[2] < 0:
                    pad = 0.75*self.conf['FONTSIZE']/12.0
            else: pad = 0.05*self.conf['FONTSIZE']/12.0

            cax = divider.append_axes(cbarloc, size='5%', pad=pad)
            # fig.add_axes(cax)  # deprecated method in matplotlib

            if self.conf['MAP_FUNC'] == 'log10':
                minor_loc = LogLocator(base=10.0, subs=(1.0,))
                idx0 = int(np.floor(log10(np.fabs(VMIN*self.timg.noise))))
                idx1 = int(np.ceil(log10(VMAX*self.timg.peak)))+1
                idxs = np.arange(idx0, idx1)
                cbar = fig.colorbar(img, cax=cax,
                                    ticks=10.0**idxs,
                                    orientation = orientation)
                if orientation == 'horizontal':
                    cbar.ax.set_xticklabels([str(i) for i in idxs])
                    cbar.ax.xaxis.set_ticks_position(cbarloc)
                    cbar.ax.xaxis.set_label_position(cbarloc)
                    # cbar.ax.xaxis.set_minor_locator(minor_loc)
                else:
                    cbar.ax.set_yticklabels([str(i) for i in idxs])
                    cbar.ax.yaxis.set_ticks_position(cbarloc)
                    cbar.ax.yaxis.set_label_position(cbarloc)
                    cbar.ax.yaxis.set_minor_locator(minor_loc)
                if cbarloc == 'right':
                    cbar.set_label('%s (log)' %self.conf['DATA_UNITS'])
            elif self.conf['MAP_FUNC'] == 'linear':
                minor_loc = MultipleLocator(0.5)
                cbar = fig.colorbar(img, cax=cax, orientation = orientation)
                if orientation == 'horizontal':
                    cbar.ax.xaxis.set_ticks_position(cbarloc)
                    cbar.ax.xaxis.set_label_position(cbarloc)
                    cbar.ax.xaxis.set_minor_locator(minor_loc)
                else:
                    cbar.ax.yaxis.set_ticks_position(cbarloc)
                    cbar.ax.yaxis.set_label_position(cbarloc)
                    cbar.ax.yaxis.set_minor_locator(minor_loc)
                if cbarloc == 'right':
                    cbar.set_label('%s' %self.conf['DATA_UNITS'])

        if self.conf['TITLE']:
            ax.set_title(r'%s' %self.conf['TITLE'])

        if self.conf['ANNOT']:
            dx = xy_range[1] - xy_range[0]
            dy = xy_range[3] - xy_range[2]
            if 'lower' in self.conf['ANNOT_LOC']:
                starty = xy_range[2]
                valign = 'bottom'
            elif 'upper' in self.conf['ANNOT_LOC']:
                starty = xy_range[3]
                dy = -dy
                valign = 'top'
            if 'left' in self.conf['ANNOT_LOC']:
                startx = xy_range[0]
                halign = 'left'
            elif 'right' in self.conf['ANNOT_LOC']:
                startx = xy_range[1]
                dx = -dx
                halign = 'right'

            tx = startx + dx*0.05
            ty = starty + dy*0.05

#            annot = self.conf['ANNOT'].decode('string_escape')
            annot = self.conf['ANNOT']
            ax.text(tx, ty, annot,
                horizontalalignment=halign,
                verticalalignment=valign,
                fontsize=1.2*self.conf['FONTSIZE'],
                fontweight='demi',
                color=ec)

        if self.conf['OUT_NAME']:
            plt.savefig(self.conf['OUT_NAME'], bbox_inches='tight', dpi=300)

        plt.tight_layout()
        plt.show()


def show_config():

    config = \
    '''
# Default setup file for fits_utils v%0.1f
# Jun Liu %s
#

#------------------------------ Files Input ---------------------------------
I_IMAGE         demo_i.fits    # name of total intensity image fits
Q_IMAGE         none           # name of the polarization Q image fits
U_IMAGE         none           # name of the polarization U image fits
P_IMAGE         demo_p.fits    # name of the polarization image fits
PANG_IMAGE      demo_pa.fits   # name of the polarization angle fits
POL_VEC         0.05, 0.0, 0.001, 8, 6
                               # Configure polarization vector plotting.
                               # See difmap command 'polvec' for more details.
FIX_VEC_LEN     0.4            # set fixed vector length for polarization
EVPA            0              # EVPA angle used to correct the plot for
                               # polarization angle
MOD_NAME        demo_i.mod     # name of the model file (created by difmap)

#------------------------ Model to FITS conversion --------------------------
CONVERT         false          # whether to convert the models to FITS image
TEMPLATE        none           # template for FITS file generation
OUT_FITS        demo.fits      # name of the output fits image
FITS_DIM        512, 512       # dimension of the output fits image
FITS_SIZE       auto           # data range for the FITS (in unit of MAP_UNITS)
CRPIX           auto           # reference position in image plane
CRVAL           auto           # source RA and Dec. in deg.
ROTATE          0              # rotation angle (in deg.) for conversion

#-------------------------- Plot Configurations -----------------------------
PLOT_STYLE      1              # higher level plotting control (overwrites
                               # CONTOUR, COLOR_STYLE) parameter, valid when
                               # PLOT_STYLE>0. Three styles are available:
                               # 1. I map shown in contours while P map in
                               #    pseudo-color and vectors.
                               # 2. I map shown in lowest level contour (dashed
                               #    line) while P in pseudo-color and vectors.
                               # 3. I map shown in pseudo-color while P map
                               #    as vectors and contours.
NOISE_THRESH    3.0            # <sigma> or <threshold> for noise detection
LIN_THRESH      4.0            # factor*[-<noise>, +<noise>] of linear range to
                               # to be stretched relative to the log10 range.
DATA_UNITS      jy/beam        # unit of measurements
MAP_UNITS       mas            # set the map coordinate units
MAP_FUNC        log10          # set the transfer function for fits image:
                               # log10|linear
IMG_MIN         5.0            # factor*<noise> as the minimum value for the
                               # data to be displayed
IMG_MAX         10.0           # factor*<peak> as the maximum value for the
                               # data to be displayed
XY_RANGE        auto           # Set the boundaries of the image
SHOW_CONT       true           # determine whether to plot the contours
CONT_FUNC       log10          # set the transfer function for contours:
                               # log10|linear
LEVS            auto           # levels for contour plot
LEV_PARAM       0, 0, 2        # parameters determing the contour levels,
                               # similar to the Difmap 'loglev' command. Valid
                               # only when LEVS is set to auto.
                               # LEV_PARAM(1): lowest positive contour level.
                               #   0 -> 3*noise;
                               # LEV_PARAM(2): highest contour level. It will
                               #   be automatically trancated when the levels
                               #   reaches the map peak. 0 -> peak;
                               # LEV_PARAM(3): increment between two continual
                               #   contour levels. Default value 2. Compatible
                               #   with formula expression, e.g. sqrt(2);
SHOW_MOD        true           # determine whether to plot the GAUSSIAN models
BEAM            1, 0.8, 60     # Set the beam (a, b, phi) for the clean models
BEAM_LOC        lowerleft      # the location of beam ellipse to be plotted:
                               # lowerleft|lowerright|upperleft|upperright
CBAR_LOC        right          # relative position of the color bar to the
                               # main image plot: left|right|top|bottom
COLOR_STYLE     jet            # color map to display the image: none|jet|
                               # binary|rainbow|gnuplot2_r ...
                               # Please check out the matplotlib manual for all
                               # available color maps.

#----------------------------- Miscellaneous ---------------------------------
FIG_SIZE        5, 5.4         # width and height for the figure in inches
LINEWIDTH       1              # linewidth for the figure in pts
FONTSIZE        12             # fontsize for the figure in pts
TITLE           source@xx\,GHz # title for the figure
XLABEL          auto           # label for x-axis
YLABEL          auto           # label for y-axis
''' %(__version__, __date__) + \
r'ANNOT           source@xx\,GHz\ndate: sxxxx.xx' + \
'''
                               # annotation for the figure
ANNOT_LOC       upperleft      # the location of beam ellipse to be plotted:
                               # lowerleft|lowerright|upperleft|upperright
SRC_NAME        xxx+xxx        # name of observed source
DATE_OBS        2018-01-01     # observation date
OUT_NAME        image.eps      # filename for figure output
#-------------------------------- END ----------------------------------------'''

    print(config)

def main():

    parser = argparse.ArgumentParser(
        description = 'fits utility   version 2.2 (2019-01-18)\n\n'
        'Written by Jun LIU <jliu@mpifr-bonn.mpg.de>\n\n'
        'fits_utils comes with ABSOLUTELY NO WARRANTY\n'
        'You may redistribute copies of fits_utils\n'
        'under the terms of the GNU General Public License.',
        formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument('-d', help='to dump a default configuration file',
                        action='store_true')
    parser.add_argument('-i', help='input configuration file')

    args = parser.parse_args()
    if args.d:
        show_config()

    if args.i:
        pu = plot_utils(args.i)
        pu.run()

__all__ = ['difmap_model', 'fits_image', 'plot_utils', 'config_parser', \
           'line_creator', 'get_edgecolor', 'show_config', 'main']

if __name__ == '__main__':

    main()
