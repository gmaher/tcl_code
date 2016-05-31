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
#get_group_from_file('/home/gabriel/projects/tcl_code/models/OSMSC0001/0001_0001/groups/arch_edge')
#get_group_from_file('/home/gabriel/projects/tcl_code/models/OSMSC0001/0001_0001/groups/arch_image')
#get_group_from_file('/home/gabriel/projects/tcl_code/models/OSMSC0001/0001_0001/groups/arch')
cwd = os.getcwd()
groups = {}
groups_edges = {}
groups_images = {}

for fn in os.listdir(cwd+'/models/weiguang/groups'):
	if "." in fn:
		continue
	print fn

	if "_edge" in fn:
		groupname = fn.replace("_edge", "")
		groups_edges[groupname] = get_group_from_file(cwd+"/models/weiguang/groups/"+fn)

	elif "_image" in fn:
		groupname = fn.replace("_image", "")
		groups_images[groupname] = get_group_from_file(cwd+"/models/weiguang/groups/"+fn)

	else:
		groups[fn] = get_group_from_file(cwd+"/models/weiguang/groups/"+fn)

print len(groups_edges.keys())
print len(groups_images.keys())
print len(groups.keys())

#count success groups
succ_edges  = 0
succ_images = 0
succ_user   = 0

for key in groups_edges.keys():
	succ_edges += len(groups_edges[key].keys())
	succ_images += len(groups_images[key].keys())
	succ_user += len(groups[key].keys())

print "checking empty image: ", groups_images["LCX_b7"].keys()

print "Edgemap successes: ", succ_edges
print "Image successes: ", succ_images
print "Total groups: ", succ_user

#calculate area errors
from shapely.geometry import Polygon
import numpy as np
err_edges  = []
err_images = []
invalid_edges = 0
invalid_images = 0

img_good_errors = []
img_bad_errors = []
edge_good_errors = []
edge_bad_errors = []

#edge errors
for surf in groups_edges.keys():

	if len(groups_edges[surf].keys()) == 0:
		continue
	else:
		for grp in groups_edges[surf].keys():

			a = Polygon(groups_edges[surf][grp])
			b = Polygon(groups[surf][grp])

			if not (a.is_valid and b.is_valid):
				invalid_edges += 1
				print surf, ", ", grp, " is invalid"
				edge_bad_errors.append((surf,grp))
				continue

			Adiff1 = a.difference(b).area
			Adiff2 = b.difference(a).area
			Adiff = Adiff1+Adiff2
			Aunion = a.union(b).area
			err = Adiff/Aunion
			err_edges.append(err)

			if err <= 0.1:
				edge_good_errors.append((surf,grp))
			elif err >=0.9:
				edge_bad_errors.append((surf,grp))

print "edges error mean: ", np.mean(err_edges)
print "edges error std : ", np.std(err_edges)
print "edges no. invalid: ", invalid_edges

#image errors
for surf in groups_images.keys():

	if len(groups_images[surf].keys()) == 0:
		continue
	else:
		for grp in groups_images[surf].keys():

			a = Polygon(groups_images[surf][grp])
			b = Polygon(groups[surf][grp])

			if not (a.is_valid and b.is_valid):
				invalid_images += 1
				print surf, ", ", grp, " is invalid"
				img_bad_errors.append((surf,grp))
				continue

			Adiff1 = a.difference(b).area
			Adiff2 = b.difference(a).area
			Adiff = Adiff1+Adiff2
			Aunion = a.union(b).area
			err = Adiff/Aunion
			err_images.append(err)

			if err <= 0.1:
				img_good_errors.append((surf,grp))
			elif err >= 0.9:
				img_bad_errors.append((surf,grp))

print "images error mean: ", np.mean(err_images)
print "images error std : ", np.std(err_images)
print "images no. invalid: ", invalid_images

#make histogram plot
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


fig = plt.figure()
edge_weights = np.ones_like(err_edges)/float(len(err_edges))
image_weights = np.ones_like(err_images)/float(len(err_images))
plt.hist(err_images, 30, weights=image_weights, facecolor='blue', alpha=0.5, label='images')
plt.hist(err_edges, 30, weights=edge_weights, facecolor='red', alpha=0.5, label='edges')
plt.legend()
plt.xlabel('area error')
plt.ylabel('frequency')
plt.show()

#print bad errors and good errors
print "good errors edges"
print edge_good_errors
print "bad errors edges"
print edge_bad_errors

print "\n"

print "good errors images"
print img_good_errors
print "bad errors images"
print img_bad_errors


#edges
def get_contour_points(srf,grp):
	zipped = zip(*groups_edges[srf][grp])
	x = [i for i in zipped[0]]
	x.append(x[0])
	y = [i for i in zipped[1]]
	y.append(y[0])
	z = [i for i in zipped[2]]
	z.append(z[0])

	#images
	zipped = zip(*groups_images[srf][grp])
	xi = [i for i in zipped[0]]
	xi.append(xi[0])
	yi = [i for i in zipped[1]]
	yi.append(yi[0])
	zi = [i for i in zipped[2]]
	zi.append(zi[0])

	#regular
	zipped = zip(*groups[srf][grp])
	xg = [i for i in zipped[0]]
	xg.append(xg[0])
	yg = [i for i in zipped[1]]
	yg.append(yg[0])
	zg = [i for i in zipped[2]]
	zg.append(zg[0])

	return (x,y,z,xi,yi,zi,xg,yg,zg)

fig2 = plt.figure()

x,y,z,xi,yi,zi,xg,yg,zg = get_contour_points('LAD',46)
ax = fig2.gca(projection='3d')
ax.plot(x,y,z, color='blue', label='edge')
ax.plot(xi,yi,zi, color='red', label='image')
ax.plot(xg,yg,zg, color='green', label='user')
plt.legend()
plt.show()

fig3 = plt.figure()

x,y,z,xi,yi,zi,xg,yg,zg = get_contour_points('RPA33',37)
ax = fig3.gca(projection='3d')
ax.plot(x,y,z, color='blue', label='edge')
ax.plot(xi,yi,zi, color='red', label='image')
ax.plot(xg,yg,zg, color='green', label='user')
plt.legend()
plt.show()

fig4 = plt.figure()

x,y,z,xi,yi,zi,xg,yg,zg = get_contour_points('LAD',14)
ax = fig4.gca(projection='3d')
ax.plot(x,y,z, color='blue', label='edge')
ax.plot(xi,yi,zi, color='red', label='image')
ax.plot(xg,yg,zg, color='green', label='user')
plt.legend()
plt.show()

fig5 = plt.figure()

x,y,z,xi,yi,zi,xg,yg,zg = get_contour_points('RPA33',0)
ax = fig5.gca(projection='3d')
ax.plot(x,y,z, color='blue', label='edge')
ax.plot(xi,yi,zi, color='red', label='image')
ax.plot(xg,yg,zg, color='green', label='user')
plt.legend()
plt.show()
