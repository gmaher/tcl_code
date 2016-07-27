import pandas as pd
import numpy as np
import os
from shapely.geometry import Polygon
from math import sqrt, pi

def get_group_from_file(fn):
	group = {}
	f = open(fn, 'r')

	while True:
		name = f.readline()
		num = f.readline()
		meta = f.readline()

		if name in ['\n', '\r\n', ""]:
			print "end of file: ", fn
			return group

		else:
			num = int(num)
			group[num] = []

			#read empty line
			l = f.readline()
			#read first group point
			l = f.readline()
			while l not in ['\n', '\r\n']:
				l = l.split()
				l = [float(i) for i in l]

				group[num].append((l[0],l[1],l[2]))

				#read next line
				l = f.readline()

def calculate_errors(image,path,groups,groups_edges, err_name):
	df = pd.DataFrame(columns=["image","path","radius",err_name])

	for grp in groups.keys():
		#edges
		if grp not in groups_edges.keys():
			edge_err = 1.0

		else:
			a = Polygon(groups_edges[grp])
			b = Polygon(groups[grp])

			if not (a.is_valid and b.is_valid):
				edge_err = 1.0

			else:

				Adiff1 = a.difference(b).area
				Adiff2 = b.difference(a).area
				Adiff = Adiff1+Adiff2
				Aunion = a.union(b).area
				edge_err = Adiff/Aunion

		b = Polygon(groups[grp])

		df = df.append({
			"image": image,
			"path": path,
			"group":grp,
			"radius": sqrt(b.area/pi),
			err_name: edge_err
			}, ignore_index=True)

	return df


data = []
cols=["image","path","group","radius"]
apps = ['_edge48',
'_image48',
'_edge96',
'_image96']

for root, dirs, files in os.walk('.'):
	if '/groups' in root:
		#print root
		#print root.split('/')[2]
		#print files
		img = root.split('/')[2]
		data.append((root,img,files))

#loop over directories, calculate errors and store in dataframe
d = pd.DataFrame()
for directory,image,files in data:
	print directory
	print image

	for fn in files:
		for app in apps:
			if fn+app in files:
				g = get_group_from_file(directory+'/'+fn)
				ge = get_group_from_file(directory+'/'+fn+app)


				d = d.append(calculate_errors(image,fn,g,ge,'error'+app), ignore_index=True)
 

dataframes = []
for app in apps:
	dataframes.append(d.loc[pd.notnull(d['error'+app]), cols+['error'+app]])



for i in range(1,len(dataframes)):
	dataframes[0] = dataframes[0].merge(dataframes[i],how='inner', on=cols)

d = dataframes[0]

d.loc[(d['error_edge48']>=0.9)].to_csv('bad_errors.csv')

d.to_csv('all_errors.csv')

print d['path'].unique()