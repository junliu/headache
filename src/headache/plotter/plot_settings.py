#! /usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import rcParams as rc
from matplotlib.pyplot import style, get_cmap
from cycler import cycler
from numpy import array

def set_fig_fcolor(color=None):
    """set figure facecolor

    :color: matplotlib color

    """
    if color == None:
        rc['axes.facecolor'] = 'white'
        rc['figure.facecolor'] = 'white'
        rc['savefig.facecolor'] = 'white'
    else:
        rc['axes.facecolor'] = color
        rc['figure.facecolor'] = color
        rc['savefig.facecolor'] = color


def set_fig_ecolor(color=None):
    """set figure edgecolor

    :color: matplotlib color

    """
    if color == None:
        rc['axes.edgecolor'] = 'white'
        rc['figure.edgecolor'] = 'white'
        rc['savefig.edgecolor'] = 'white'
    else:
        rc['axes.edgecolor'] = color
        rc['figure.edgecolor'] = color
        rc['savefig.edgecolor'] = color


def set_color_palette(cmap=None):
    """set color palette for plot

    :cmap: name of matplotlib QUALITATIE colormap, e.g., Pastel1, Pastel2,
    Paired, Accent, Dark2, Set1, set2, Set3, tab10, tab20, tab20b, tab20c.
    see https://matplotlib.org/examples/color/colormaps_reference.html

    If cmap is set to None, color palette switch to default.

    """
    if not cmap:
        # the classic default color cycle
        # palette = 'bgrcmyk'
        # the v2.0 default color cycle
        palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', \
                    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', \
                    '#bcbd22', '#17becf']
    else:
        palette = []
        cmap0 = get_cmap(cmap)
        for rgb in array(cmap0.colors)*255:
            rgb = rgb.astype(int)
            palette.append('#{:02x}{:02x}{:02x}'.format(*tuple(rgb)))

    rc['axes.prop_cycle'] = cycler(color=palette)


def set_latex(use=True):
    """latex usage in plot (True or False), default: True

    """
    rc['text.usetex'] = use

def set_fontsize(fs=12):
    rc['font.size'] = fs

def set_linewidth(lw=1):
    rc['lines.linewidth'] = lw

def set_ticks(l=3, vis=True):
    """set axes ticks for plot

    :l:   1/4 of the length of major ticks
    :vis: visibility of ticks (True or False), default: True

    """
    rc['xtick.direction'] = 'in'
    rc['ytick.direction'] = 'in'

    rc['xtick.major.size']=4*l
    rc['xtick.minor.size']=2*l
    rc['xtick.minor.visible']=vis

    rc['ytick.major.size']=3*l
    rc['ytick.minor.size']=1.5*l
    rc['ytick.minor.visible']=vis

    rc['xtick.top'] = True
    rc['ytick.right'] = True


def set_theme(sty='sci'):
    """set theme for plot

    :sty: name of the them for plot.
    Available names are sci, tech and normal. Default value is sci.

    """
    set_latex(True)
    rc['figure.dpi'] = 100
    rc['savefig.dpi'] = 300
    rc['legend.numpoints'] = 1
    rc['legend.scatterpoints'] = 1
    rc['savefig.bbox'] = 'tight'
    rc['axes.facecolor'] = 'white'
    rc['axes.edgecolor'] = 'black'

    def default():
        fs = 10
        set_fontsize(fs)
        set_linewidth(1.5)
        set_ticks(3, True)
        rc['legend.fontsize'] = 0.9*fs
        rc['axes.labelsize'] = 1.2*fs
        rc['axes.grid'] = False
        rc['axes.linewidth'] = 1.0
        rc['grid.color'] = 'none'
        rc['grid.linewidth'] = 0.8
        rc['image.cmap'] = 'jet'

    if sty == 'sci':
        style.use('seaborn-bright')
        default()

    elif sty == 'tech':
        style.use('seaborn-dark')
        set_theme('normal')
        set_ticks(0, False)
        set_linewidth(1.2)
        rc['axes.grid'] = True
        rc['axes.facecolor'] = '#EAEAF2'
        rc['axes.edgecolor'] = 'white'
        rc['axes.linewidth'] = 0
        rc['grid.color'] = 'white'
        rc['grid.linewidth'] = 1.5

    elif sty == 'normal':
        set_theme('sci')
        fs = 12
        set_fontsize(fs)
        set_linewidth(1)
        set_ticks(2, True)
        rc['legend.fontsize'] = fs
        rc['axes.labelsize'] = fs
        rc['grid.linewidth'] = 1

    else:
        print('style %s is not available, switch to normal.' %sty)
        set_theme('normal')


__all__ = ['set_fig_fcolor', 'set_fig_ecolor', 'set_color_palette',
           'set_latex', 'set_fontsize', 'set_linewidth', 'set_ticks',
           'set_theme']


if __name__ == "__main__":
    pass
