from utility import get_group_from_file

import pandas as pd
import numpy as np
import os
from shapely.geometry import Polygon
from math import sqrt, pi
import sys

'''
Looks for a number of random groups in each error range and plots their contours
'''

if len(sys.argv) < 2:
    print("usage:")
    print("python ", sys.argv[0], "<vascular data>")
    exit()

df = pd.read_csv('all_errors.csv')