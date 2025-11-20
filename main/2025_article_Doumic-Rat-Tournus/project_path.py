#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 12:59:33 2024

@author: anais

Contains the code for relative imports.

The script imports the path of the project: from any other script of the same
folder, import this module to be able, then, to import the modules of the
telomere package.

(Solution taken from https://stackoverflow.com/questions/34478398/import-local-
function-from-a-module-housed-in-another-directory-with-relative-im.)

"""

import sys
import os

project_path = os.path.abspath(os.path.join(os.pardir, os.pardir))
if project_path not in sys.path:
    sys.path.append(project_path)
