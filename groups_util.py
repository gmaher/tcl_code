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
			print num
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

import os
import pandas as pd
#get_group_from_file('/home/gabriel/projects/tcl_code/models/OSMSC0001/0001_0001/groups/arch_edge')
#get_group_from_file('/home/gabriel/projects/tcl_code/models/OSMSC0001/0001_0001/groups/arch_image')
#get_group_from_file('/home/gabriel/projects/tcl_code/models/OSMSC0001/0001_0001/groups/arch')
cwd = os.getcwd()
groups = {}
groups_edges = {}
groups_images = {}

dirs = []

dirs.append("/models/OSMSC0001/0001_0001/groups/")
dirs.append("/models/OSMSC0002/0002_0001/groups/")
dirs.append("/models/OSMSC0003/0003_0001/groups/")
dirs.append("/models/weiguang/groups/")
for adir in dirs:
	dir_str = adir.split('/')
	dir_str = dir_str[2]
	groups[dir_str] = {}
	groups_edges[dir_str] = {}
	groups_images[dir_str] = {}
	print dir_str
	print adir

	for fn in os.listdir(cwd+adir):
		if "." in fn:
			continue

		print fn

		if "_edge" in fn:
			groupname = fn.replace("_edge", "")
			groups_edges[dir_str][groupname] = get_group_from_file(cwd+ adir +fn)

		elif "_image" in fn:
			groupname = fn.replace("_image", "")
			groups_images[dir_str][groupname] = get_group_from_file(cwd+ adir +fn)

		else:
			groups[dir_str][fn] = get_group_from_file(cwd+ adir +fn)

print len(groups_edges.keys())
print len(groups_images.keys())
print len(groups.keys())

#count success groups
succ_edges  = 0
succ_images = 0
succ_user   = 0

print "Edgemap successes: ", succ_edges
print "Image successes: ", succ_images
print "Total groups: ", succ_user

#calculate area errors
from shapely.geometry import Polygon
import numpy as np

invalid_edges = 0
invalid_images = 0

df = pd.DataFrame(columns=['image', 'surface','group','error', 'type'])

#calculate threshold ratios
edge_total_errs = []
img_total_errs = []

for adir in groups.keys():
	for surf in groups[adir].keys():
			if surf not in groups_edges[adir].keys():
				continue
			for grp in groups[adir][surf].keys():
				#edges
				if grp not in groups_edges[adir][surf].keys():
					edge_total_errs.append(1.0)

					df = df.append({"image" : adir,
						"surface":surf,
						"group" :grp,
						"error":1.0,
						"type":'edge'}, ignore_index=True)

				else:
					a = Polygon(groups_edges[adir][surf][grp])
					b = Polygon(groups[adir][surf][grp])

					if not (a.is_valid and b.is_valid):
						edge_total_errs.append(1.0)

						df = df.append({"image" : adir,
							"surface":surf,
							"group" :grp,
							"error":1.0,
							"type":'edge'}, ignore_index=True)
					else:

						Adiff1 = a.difference(b).area
						Adiff2 = b.difference(a).area
						Adiff = Adiff1+Adiff2
						Aunion = a.union(b).area
						err = Adiff/Aunion
						edge_total_errs.append(err)

						df = df.append({"image":adir,
							"surface":surf,
							"group" :grp,
							"error":err,
							"type":'edge'}, ignore_index=True)

				#images
				if grp not in groups_images[adir][surf].keys():
					img_total_errs.append(1.0)
					df = df.append({"image":adir,
						"surface":surf,
						"group" :grp,
						"error":1.0,
						"type":'img'}, ignore_index=True)
				else:
					a = Polygon(groups_images[adir][surf][grp])
					b = Polygon(groups[adir][surf][grp])

					if not (a.is_valid and b.is_valid):
						img_total_errs.append(1.0)
						df = df.append({"image":adir,
							"surface":surf,
							"group" :grp,
							"error":1.0,
							"type":'img'}, ignore_index=True)
					else:

						Adiff1 = a.difference(b).area
						Adiff2 = b.difference(a).area
						Adiff = Adiff1+Adiff2
						Aunion = a.union(b).area
						err = Adiff/Aunion
						img_total_errs.append(err)

						df = df.append({"image":adir,
							"surface":surf,
							"group" :grp,
							"error":err,
							"type":'img'}, ignore_index=True)

#calculate threshold success rate
thresholds = np.arange(0.025,1.025,0.025)
edge_sucs = []
edge_fails = []
img_sucs = []
img_fails = []
for thresh in thresholds:
	e = float(len([i for i in edge_total_errs if i <= thresh]))
	l = len(edge_total_errs)
	edge_sucs.append(e/l)
	edge_fails.append(1.0-e/l)

	e = float(len([i for i in img_total_errs if i <= thresh]))
	l = len(img_total_errs)
	img_sucs.append(e/l)
	img_fails.append(1-e/l)

						
##make histogram plot
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
mpl.rc('font', size=15)

err_fig = plt.figure()
plt.plot(thresholds, edge_sucs, color = 'blue', marker='o', label='edge successes')
#plt.plot(thresholds, edge_fails, color = 'blue', marker='^', label='edge failures')
plt.plot(thresholds, img_sucs, color = 'green', marker='o', label='img successes')
#plt.plot(thresholds, img_fails, color = 'green', marker='^', label='img failures')
plt.xlabel(r'$\epsilon$ threshold')
plt.ylabel('proportion groups meeting threshold')
plt.legend(loc='center left')
plt.show()

fig = plt.figure()
print df.head(10)
err_edges = df.loc[(df['type'] == 'edge') & (df["image"] == "weiguang"), 'error'].values
err_images = df.loc[(df['type'] == 'img') & (df["image"] == "weiguang"), 'error'].values

edge_weights = np.ones_like(err_edges)/float(len(err_edges))
image_weights = np.ones_like(err_images)/float(len(err_images))
plt.hist(err_images, 30, weights=image_weights, facecolor='blue', alpha=0.5, label='img')
plt.hist(err_edges, 30, weights=edge_weights, facecolor='red', alpha=0.5, label='edge')
plt.legend()
plt.xlabel(r'$\epsilon$ error')
plt.ylabel('frequency')
plt.show()

# #edges
# def get_contour_points(srf,grp):
# 	zipped = zip(*groups_edges[srf][grp])
# 	x = [i for i in zipped[0]]
# 	x.append(x[0])
# 	y = [i for i in zipped[1]]
# 	y.append(y[0])
# 	z = [i for i in zipped[2]]
# 	z.append(z[0])

# 	#images
# 	zipped = zip(*groups_images[srf][grp])
# 	xi = [i for i in zipped[0]]
# 	xi.append(xi[0])
# 	yi = [i for i in zipped[1]]
# 	yi.append(yi[0])
# 	zi = [i for i in zipped[2]]
# 	zi.append(zi[0])

# 	#regular
# 	zipped = zip(*groups[srf][grp])
# 	xg = [i for i in zipped[0]]
# 	xg.append(xg[0])
# 	yg = [i for i in zipped[1]]
# 	yg.append(yg[0])
# 	zg = [i for i in zipped[2]]
# 	zg.append(zg[0])

# 	return (x,y,z,xi,yi,zi,xg,yg,zg)

# mpl.rc('font', size=14)

# fig2 = plt.figure()

# x,y,z,xi,yi,zi,xg,yg,zg = get_contour_points('LAD',46)
# ax = fig2.gca(projection='3d')
# ax.plot(x,y,z, color='blue', label='edge')
# ax.plot(xi,yi,zi, color='red', label='image')
# ax.plot(xg,yg,zg, color='green', label='user')
# plt.xlabel('x')
# plt.ylabel('y')
# ax.set_zlabel('z')
# plt.legend()
# plt.title('LAD, 46')
# plt.show()

# fig3 = plt.figure()

# x,y,z,xi,yi,zi,xg,yg,zg = get_contour_points('RPA33',37)
# ax = fig3.gca(projection='3d')
# ax.plot(x,y,z, color='blue', label='edge')
# ax.plot(xi,yi,zi, color='red', label='image')
# ax.plot(xg,yg,zg, color='green', label='user')
# plt.xlabel('x')
# plt.ylabel('y')
# ax.set_zlabel('z')
# plt.legend()
# plt.title('RPA33, 37')
# plt.show()

# fig4 = plt.figure()

# x,y,z,xi,yi,zi,xg,yg,zg = get_contour_points('LAD',14)
# ax = fig4.gca(projection='3d')
# ax.plot(x,y,z, color='blue', label='edge')
# ax.plot(xi,yi,zi, color='red', label='image')
# ax.plot(xg,yg,zg, color='green', label='user')
# plt.xlabel('x')
# plt.ylabel('y')
# ax.set_zlabel('z')
# plt.legend()
# plt.title('LAD, 14')
# plt.show()

# fig5 = plt.figure()

# x,y,z,xi,yi,zi,xg,yg,zg = get_contour_points('RPA33',0)
# ax = fig5.gca(projection='3d')
# ax.plot(x,y,z, color='blue', label='edge')
# ax.plot(xi,yi,zi, color='red', label='image')
# ax.plot(xg,yg,zg, color='green', label='user')
# plt.xlabel('x')
# plt.ylabel('y')
# ax.set_zlabel('z')
# plt.legend()
# plt.title('RPA33, 0')
# plt.show()
