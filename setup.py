#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
from distutils.core import setup, Extension

name = 'headache'
version = '1.1.1'
description = \
'HEADACHE: High Efficiency Astronomical Data Analysis with Chic Elegance'
package_dir = {name:'src'}
packages = os.listdir(package_dir[name])
packages = [name+'.'+p for p in packages if os.path.isdir(package_dir[name]+'/'+p)]
packages.insert(0, name)

setup(name = name,
    version = version,
    description = description,
    author = 'Jun LIU',
    author_email = 'jliu@mpifr-bonn.mpg.de',
    url = 'https://www.github.com/jliu-radio/headache',
    packages = packages,
    package_dir = package_dir,
#    package_data = {'package' : files },
#    scripts = ["runner"],
    long_description = open('README.md').read(),
    long_description_content_type = 'text/markdown'
)
