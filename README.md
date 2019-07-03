# headache
**H**igh **E**fficiency **A**stronomical **D**ata **A**nalysis with **CH**ic Elegance

## installation

`python setup.py install`

with either `python2` or `python3`

## requirements

- argparse
- numpy
- scipy
- matplotlib
- astropy


## submodules and functions


### `headache.io` 

submodule for reading and writing various formats of data.

#### `headache.io.ascii`

- `readcol(fname, cols=None, fmt=None, start=0, stop=0, comment='#', flag=True)`

read an ascii file by column, returns a list of numpy arrays.

```python
from headache.io import ascii
srcname, mjd, flux, err = ascii.readcol('test.dat', cols=[0,3,4,5], fmt='sfff')
```

- `writecol(fname, data, fmt, hdr=None, tail=None)`

write to an ascii file by column


### `headache.coord`

submodule for coordination transformation.

#### `headache.coord.polar`

- `reproject_image_into_polar(data, origin=None, Jacobian=False, dr=1.0, dt=None)`


- `cart2polar(x, y, center)`


- `polar2cart(r, theta, center)`


#### `headache.plotter`

submodule for plotting (using matplotlib)

- `set_fig_fcolor('white')` 

- `set_fig_ecolor('black')`

- `set_color_palette`

- `set_latex(True)`

- `set_fontsize(15)`

- `set_linewidth(1)`

- `set_ticks`,

- `set_theme('sci')`

### `fits_utils`

submodule for operating and plotting fits image, can also run as an executable, e.g.

```bash
./fits_utils -d > plot.inp   # to dump a default configuration file for plotting
./fits_utils -i plot.inp     # input a configuration file for plotting
```

The default configuration is showing bellow
```bash
# Default setup file for fits_utils v2.2
# Jun Liu 2019-01-18
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
ANNOT           source@xx\,GHz\ndate: sxxxx.xx
                               # annotation for the figure
ANNOT_LOC       upperleft      # the location of beam ellipse to be plotted:
                               # lowerleft|lowerright|upperleft|upperright
SRC_NAME        xxx+xxx        # name of observed source
DATE_OBS        2018-01-01     # observation date
OUT_NAME        image.eps      # filename for figure output
#-------------------------------- END ----------------------------------------
```

### `headache.toolkits_vlbi`

toolkits for VLBI analysis

#### `get_ridgeline`

compute and plot the ridgline of AGN jets

- run as a function

```python
get_ridgeline(infits, core = None, method = 'peak', onesided = True, pa = 0,
               dpa = 30, dpa_iter = 60, noise = 0, detect_thresh = 3,
               min_radius = 0, max_radius = 0, step = 5, smooth = 5,
               out_data = None, plot_fig = True, plot_thresh = 3, plot_window = None)
```

- run as an executable
```bash
usage: ridgeline.py [-h] -i image.fits [-c None] [-m equal] [--ts] [-pa 0.0]
                    [-dpa 90.0] [-dpai 60.0] [-noise 0] [-dthresh 10.0]
                    [-rmin 0.0] [-rmax 120.0] [-step 5] [-smooth 5] [-o None]
                    [--noplot] [-pthresh 5.0] [-pw [2,-2,-2,2]]

ridgeline.py   version 1.6 (2019-02-22)

Written by Jun LIU <jliu@mpifr-bonn.mpg.de>

ridgeline comes with ABSOLUTELY NO WARRANTY
You may redistribute copies of fits_utils
under the terms of the GNU General Public License.

optional arguments:
  -h, --help       show this help message and exit
  -i image.fits    input fits file
  -c None          location of the core (list or None)
  -m equal         method of ridgline finding, equal or peak
  --ts             twosided jets
  -pa 0.0          initial guess for jet position angle in deg.
  -dpa 90.0        deviation of PA in deg.
  -dpai 60.0       deviation of PA in each iteration
  -noise 0         noise in the image; 0-> auto calculation
  -dthresh 10.0    factor*<noise> for ridgeline detection
  -rmin 0.0        starting radius for ridgline detection (in pixel)
  -rmax 120.0      maximum radius for ridgline detection (in pixel)
  -step 5          step for ridgline detection (in pixel)
  -smooth 5        smoothing for ridgline detection (in pixel)
  -o None          output file name; None-> same as the input
  --noplot         do not plot the image
  -pthresh 5.0     factor*<noise> as lowest level to show the images
  -pw [2,-2,-2,2]  xyrange for image plotting
```

