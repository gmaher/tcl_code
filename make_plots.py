import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


apps = ['_edge48',
'_image',
'_edge96']

colors = {'_edge48':'red',
'_image':'blue',
'_edge96':'green'}

markers = {'_edge48':'o',
'_image':'v',
'_edge96':'x'}


df = pd.read_csv('all_errors.csv')

# plt.scatter(x=np.log(df['radius']), y=np.log(df['error_edge']), label='edge_error')
# plt.legend()
# plt.xlabel('log(radius)')
# plt.ylabel('log(edge_error)')
# plt.show()

# plt.scatter(x=np.log(df['radius']), y=np.log(df['error_image']), label='image_error')
# plt.legend()
# plt.xlabel('log(radius)')
# plt.ylabel('log(edge_error)')
# plt.show()

# df['error_image'].hist(bins=20, label='image error')
# plt.legend()
# plt.show()

# df['error_edge'].hist(bins=20, label='edge error')
# plt.legend()
# plt.show()

# df['error_image_bl05'].hist(bins=20, label='image bl05 error')
# plt.legend()
# plt.show()

# df['error_edge_bl05'].hist(bins=20, label='edge bl05 error')
# plt.legend()
# plt.show()

thresh_errs = {}
for app in apps:
	thresh_errs[app] = []

ts = np.arange(0,1.1,0.1)
for t in ts:
	for app in apps:
		frac = float(np.sum(df['error'+app]<=t))/len(df['error'+app])
		thresh_errs[app].append(frac)

for app in apps:
	plt.plot(ts,thresh_errs[app], color = colors[app], marker=markers[app], markersize=8, label=app, linewidth=2)

plt.legend(apps, loc='upper left')
plt.show()