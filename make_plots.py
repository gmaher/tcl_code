import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


apps = ['_edge','_image','_edge_bl05','_image_bl05']
colors = {'_edge':'red', '_image':'blue', '_edge_bl05':'green', '_image_bl05':'black'}

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

ts = np.arange(0,1.025,0.025)
for t in ts:
	for app in apps:
		frac = float(np.sum(df['error'+app]<=t))/len(df['error'+app])*4
		thresh_errs[app].append(frac)

for app in apps:
	plt.plot(ts,thresh_errs[app], color = colors[app], label=app)

plt.legend(apps, loc='lower right')
plt.show()