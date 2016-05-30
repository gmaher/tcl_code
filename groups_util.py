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

print groups_edges.keys()
print groups_images.keys()
print groups.keys()