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
				continue

			Adiff1 = a.difference(b).area
			Adiff2 = b.difference(a).area
			Adiff = Adiff1+Adiff2
			Aunion = a.union(b).area
			err_edges.append(Adiff/Aunion)

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
				continue

			Adiff1 = a.difference(b).area
			Adiff2 = b.difference(a).area
			Adiff = Adiff1+Adiff2
			Aunion = a.union(b).area
			err_images.append(Adiff/Aunion)

print "images error mean: ", np.mean(err_images)
print "images error std : ", np.std(err_images)
print "images no. invalid: ", invalid_images

#make histogram plot
import matplotlib.pyplot as plt
edge_weights = np.ones_like(err_edges)/float(len(err_edges))
image_weights = np.ones_like(err_images)/float(len(err_images))
plt.hist(err_images, 30, weights=image_weights, facecolor='blue', alpha=0.5, label='images')
plt.hist(err_edges, 30, weights=edge_weights, facecolor='red', alpha=0.5, label='edges')
plt.legend()
plt.xlabel('area error')
plt.ylabel('frequency')
plt.show()