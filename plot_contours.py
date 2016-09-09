from utility import *

import pandas as pd
import numpy as np
import os
from shapely.geometry import Polygon
from math import sqrt, pi
import sys
import plotly as py
import plotly.graph_objs as go

'''
Looks for a number of random groups in each error range and plots their contours
'''

if len(sys.argv) < 2:
    print("usage:")
    print("python ", sys.argv[0], "<vascular data>")
    exit()

df = pd.read_csv('all_errors.csv')

key = "error_edge96"

errs = np.arange(0, 1.2, 0.2)

members = []

for i in range(0,len(errs)-1):
	d = df.loc[(df[key] >= errs[i]) 
	& (df[key] < errs[i+1])
	& (df['error_image'] < 1)
	& (df[key] < 1)]
	
	ind = np.random.randint(d.shape[0])

	print ind
	members.append(d.iloc[ind,:])


Xs = []
Ys = []
legends = []
titles = []

for mem in members:
	print mem
	fn = get_groups_dir(sys.argv[1], mem['image'])
	fn = fn + '/'+mem['path']

	group = get_group_from_file(fn)[mem['group']]
	group_image = get_group_from_file(fn+'_image')[mem['group']]
	group_edge = get_group_from_file(fn+'_edge96')[mem['group']]
	
	q, mu = get_vec_shift(group)

	x,y = project_group(group, q, mu)
	x_image, y_image = project_group(group_image, q, mu)
	x_edge, y_edge = project_group(group_edge, q, mu)

	plot_data_plotly([x,x_image,x_edge],
		[y, y_image, y_edge],
		['user','image','edge'],
		mem['path']+', error= '+str(mem['error_edge96']),
		fn='./plots/'+mem['path']+'.html')

	Xs.append([x,x_image,x_edge])
	Ys.append([y,y_image,y_edge])
	legends.append(['user','image','edge'])
	titles.append(mem['path']+', error= '+str(mem['error_edge96']))

plot_data_subfigures(Xs,Ys,legends+['acceptable error'], titles,rows=3,cols=2, size=1000)
print mem
imstr = query_file(sys.argv[1]+mem['image'],
	[str(int(mem['group'])), mem['path'], 'edge', 'edge96', 'pot'
	])

print imstr

trace = make_image_trace(imstr,
	bounds=[-7.3,7.3,-7.3,7.3])
traces = make_traces([x,x_image,x_edge],
	[y,y_image,y_edge],
	['user','image','edge'], mode='lines')

traces.append(trace)
py.offline.plot(traces)