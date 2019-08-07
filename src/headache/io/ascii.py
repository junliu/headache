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

"""
  module ascii

"""

__author__ = 'Jun LIU'
__copyright__ = 'Copyright (c) 2019 Jun Liu <jliu@mpifr-bonn.mpg.de>'
__license__ = 'GPL v3'
__version__ = '1.0'


__all__ = ['readcol', 'writecol']

import numpy as np


def comment_remover(line):

    if line.strip()[0]!='%' and line.strip()[0]!='#':
        return True

    else:
        return False

def type_guess(string):

    string = string.strip()

    try: dtype = type(eval(string))
    except: dtype = str

    return dtype

def type_specify(string):

    if string == 'F':
        dtype = float
    elif string == 'D':
        dtype = int
    elif string == 'S':
        dtype = str
    else:
        print('Wrong data format, use i, f, or s')
        return

    return dtype


def readcol(fname, cols=None, fmt=None,
                    start=0, stop=0,
                    comment='#', delimiter=None, flag=True):
    """
    readcol(fname, cols=None, fmt=None,
            start=0, stop=0, comment='#', delimiter=None, flag=True):

    """

    with open(fname) as fobj:
        lines = fobj.readlines()
    n = len(lines)
    if stop == 0: stop = n+1
    lines = lines[start:stop]
    if flag:
        lines = [line for line in lines if line.strip()[0]!=comment]
    else:
        lines = [line if line.strip()[0]!=comment else line.strip()[1:] for line in lines]

    if not cols:
        tmp = lines[0].split(delimiter)
        ncol = len(tmp)
#        if flag: ncol = len(tmp)
#        else:
#            if tmp[0] != comment: ncol = len(tmp) + 1
#            else: ncol = len(tmp)
        cols = range(ncol)
    else:
        ncol = len(cols)
#        if flag: ncol = len(cols)
#        else:
#            cols = [e+1 for e in cols]
#            cols.insert(0, 0)
#            ncol = len(cols)

    n = len(lines)

    if not fmt:
        dfmt = []
        line = lines[0].split(delimiter)
        if len(line) < ncol: line.insert(0, '#')
        for i in cols:
            dfmt.append(type_guess(line[i]))
    else:
        fmt = [e for e in fmt if e!=',' and e!='%']
        fmt = [e.strip().upper() for e in fmt]
        fmt = [e for e in fmt if e!='']
        dfmt = [type_specify(s) for s in fmt]

    if ncol == len(dfmt) + 1: dfmt.insert(0, str)

    data = [list(range(n)) for _ in range(ncol)]
    for i in range(n):
        line = lines[i].split(delimiter)
        for j in range(ncol):
            col = cols[j]
            data[j][i] = dfmt[j](line[col].lstrip().rstrip())

    # convert the data into numpy arrays
    for i in range(ncol):
        data[i] = np.array(data[i])

    if ncol == 1:
        data = data[0]

    return data



def writecol(fname, hdr, tail, data, fmt):
    """
    writecol(fname, hdr=None, tail=None,
            data, fmt):

    """

    ncol = len(data)
    try:
        nline = len(data[0])
    except:
        nline = ncol*1
        ncol = 1
        data = [data]

    dlist = []

    fobj = open(fname, 'w')
    if hdr:
        print >> fobj, hdr

    for i in range(nline):
        dlist = []
        for j in range(ncol):
            dlist.append(data[j][i])
        dtuple = tuple(dlist)
        print >> fobj, fmt %dtuple

    if tail:
        print >> fobj, tail

    fobj.close()

### end FileIO functions ###

def array_flat(array, norm=0, ref_array=None):
    """
    array = [array1, array2, array3 ....]
    """
    out_array = np.array([])
    n = len(array)
    locate = np.zeros(n)
    if (type(norm) == int) and (norm == 0):
        norm = np.ones(n)
    elif (type(norm) == int) and (norm == 1):
        norm = np.array([np.mean(e) for e in array])

    elif (type(norm) == int) and (norm > 1):
        order = copy.deepcopy(norm)
        norm = np.zeros(n)

        for i in range(n):
            par = np.polyfit(ref_array[i], array[i], order)
            func = lambda x: -np.polyval(par, x)
            mini = minimize(func, [48.0])
            norm[i] = -mini.fun
            loc = mini.x
            if len(loc) > 1:
                idx = np.argmin(np.fabs(loc-48))[0][0]
                locate[i] = loc[idx]
            else: locate[i] = loc[0]

    for i in range(n):
        if array[i].dtype =='S1':
            out_array = np.append(out_array, array[i])
        else:
            out_array = np.append(out_array, array[i]/norm[i])

    return out_array, norm, locate

