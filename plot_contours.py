from utility import get_group_from_file

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

def get_filename(d):
	fn = sys.argv[1]

	img = d['image']
	
	folder = fn+img

	for root, dirs, files in os.walk(folder):
		if 'groups-cm' in root:
			return root

def get_vec_shift(group):
	'''
	takes a list of 3d point tuples that lie on a plane
	and converts them to 2d points

	inputs:
	- group: list of (x,y,z) tuples
	'''
	group = [np.asarray(p) for p in group]
	group.append(group[0])
	l = len(group)
	p1 = group[0]
	p2 = group[int(np.floor(float(l)/4))]

	A = np.zeros((3,2))
	A[:,0] = p1[:]
	A[:,1] = p2[:]
	mu = np.mean(A,axis=1, keepdims=True)
	A = A - mu

	q,r = np.linalg.qr(A)

	return (q,mu)

def project_group(group, q, mu):
	group = [np.asarray(p) for p in group]
	group.append(group[0])
	group_centered = [(p-mu.T)[0] for p in group]
	
	twod_group = [p.dot(q) for p in group_centered]
	x = [p[0] for p in group_centered]
	y = [p[1] for p in group_centered]
	
	return (x,y)

for mem in members:
	print mem
	fn = get_filename(mem)
	fn = fn + '/'+mem['path']

	group = get_group_from_file(fn)[mem['group']]
	group_image = get_group_from_file(fn+'_image')[mem['group']]
	group_edge = get_group_from_file(fn+'_edge96')[mem['group']]
	
	q, mu = get_vec_shift(group)

	x,y = project_group(group, q, mu)
	x_image, y_image = project_group(group_image, q, mu)
	x_edge, y_edge = project_group(group_edge, q, mu)

	trace = go.Scatter(
	x = x,
	y = y,
	mode = 'lines+markers',
	name = 'group'
	)

	trace_image = go.Scatter(
	x = x_image,
	y = y_image,
	mode = 'lines+markers',
	name = 'group_image'
	)

	trace_edge = go.Scatter(
	x = x_edge,
	y = y_edge,
	mode = 'lines+markers',
	name = 'group_edge'
	)

	layout = go.Layout(
		title = mem['path']+', error= '+str(mem['error_edge96'])
		)

	fig = go.Figure(data=[trace, trace_image, trace_edge], layout=layout)

	py.offline.plot(fig, filename="./plots/plot"+mem['path']+".html")