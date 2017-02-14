import numpy as np
import utility.utility as utility
import vtk
import argparse
from shapely.geometry import Polygon, Point
import os
from tqdm import tqdm
import configparser
np.random.seed(0)
#############################
#Parse input arguments
#############################
parser = argparse.ArgumentParser()
parser.add_argument('input_dir')
parser.add_argument('output_dir')
parser.add_argument('imsize')
parser.add_argument('-t','--type',default='all',choices=['all','ct','mr','jameson'])
args = parser.parse_args()
input_dir = args.input_dir
output_dir = args.output_dir
im_type = args.type
imsize = int(args.imsize)

#############################
#create output directories
#############################
if not os.path.exists(os.path.abspath(output_dir)):
    os.mkdir(os.path.abspath(output_dir))

dirs = {}
if im_type == 'all':
    if not os.path.exists(os.path.abspath(output_dir+'all')):
        os.mkdir(os.path.abspath(output_dir+'all'))

    dirs['train'] = output_dir+'all/train/'
    dirs['val'] = output_dir+'all/val/'
    dirs['test'] = output_dir+'all/test/'

if im_type == 'mr':
    if not os.path.exists(os.path.abspath(output_dir+'mr')):
        os.mkdir(os.path.abspath(output_dir+'mr'))

    dirs['train'] = output_dir+'mr/train/'
    dirs['val'] = output_dir+'mr/val/'
    dirs['test'] = output_dir+'mr/test/'

if im_type == 'ct':
    if not os.path.exists(os.path.abspath(output_dir+'ct')):
        os.mkdir(os.path.abspath(output_dir+'ct'))

    dirs['train'] = output_dir+'ct/train/'
    dirs['val'] = output_dir+'ct/val/'
    dirs['test'] = output_dir+'ct/test/'

if im_type == 'jameson':
    if not os.path.exists(os.path.abspath(output_dir+'jameson')):
        os.mkdir(os.path.abspath(output_dir+'jameson'))

    dirs['train'] = output_dir+'jameson/train/'
    dirs['val'] = output_dir+'jameson/val/'
    dirs['test'] = output_dir+'jameson/test/'

for k in dirs.keys():
    if not os.path.exists(os.path.abspath(dirs[k])):
        os.mkdir(os.path.abspath(dirs[k]))

#split = float(config['learn_params']['split'])
split = 0.15

################################
# Create variables to store data
################################
files = os.listdir(input_dir)
images = []
segmentations = []
meta_data = [[],[],[]]
contours = []
contours_ls = []
contours_edge = []
contours_seg = []
im_seg = []
mag_seg = []
names = open(input_dir+'names.txt','w')

eccentricity_limit = 0.2

#################################
# Make model train/val/test split
#################################
split_models = {}
if im_type == 'jameson':
    split_models['train'] = open('./data/jameson_train.txt').readlines()
    split_models['val'] = open('./data/jameson_val.txt').readlines()
    split_models['test'] = open('./data/jameson_test.txt').readlines()
    for k in split_models.keys():
        split_models[k] = [s.replace('\n','') for s in split_models[k]]
    models = split_models['train'] + split_models['val'] + split_models['test']
else:
    if im_type == 'all':
        models = [f.split('.')[0] for f in files if 'OSMSC' in f]
    if im_type == 'mr':
        models = open('./data/mr_images.list').readlines()
    if im_type == 'ct':
        models = open('./data/ct_images.list').readlines()

    models = [m.replace('\n','') for m in models]
    models = list(set(models))
    inds = np.random.permutation(len(models))
    cut = int(round(split*len(models)))


    split_models['test'] = [models[i] for i in inds[:cut]]
    if im_type == 'ct' or im_type == 'all':
        split_models['test'].append('OSMSC0002')
        split_models['val'] = [models[i] for i in inds[cut:2*cut] if models[i] != "OSMSC0002"]
        split_models['train'] = [models[i] for i in inds[2*cut:] if models[i] != "OSMSC0002"]
    else:
        split_models['val'] = [models[i] for i in inds[cut:2*cut]]
        split_models['train'] = [models[i] for i in inds[2*cut:]]



split_inds = {}
split_inds['train'] = []
split_inds['val'] = []
split_inds['test'] = []

count = 0
files = [f for f in files if 'truth.ls' in f and (not "OSMSC0159" in f) and any(s.lower() in f.lower() for s in models)]

segmentations = np.zeros((len(files),imsize,imsize))
images = np.zeros((len(files),imsize,imsize))
im_seg = np.zeros((len(files),imsize,imsize))
mag_seg = np.zeros((len(files),imsize,imsize))
for f in tqdm(files):
    if "truth.ls" in f:
        mag = f.replace('truth.ls.vtp','truth.mag.vts')
        ls = f
        ls_image = f.replace('truth','image')
        ls_edge = f.replace('truth','edge96')
        ls_seg = f.replace('truth','seg3d')


        contour = utility.VTKPDReadAndReorder(input_dir+ls)
        contour_image = utility.VTKPDReadAndReorder(input_dir+ls_image)
        contour_edge = utility.VTKPDReadAndReorder(input_dir+ls_edge)
        if utility.eccentricity(contour) < eccentricity_limit:
            continue

        #convert contours to 2d
        contour = contour[:,:2]
        contour_image = contour_image[:,:2]
        contour_edge = contour_edge[:,:2]
        poly = Polygon(contour)

        mag_sp = utility.readVTKSP(input_dir+mag)

        spacing = mag_sp.GetSpacing()
        origin = mag_sp.GetOrigin()
        dims = mag_sp.GetDimensions()

        seg = utility.contourToSeg(contour, origin, dims, spacing)
        mag_np = utility.VTKSPtoNumpy(mag_sp)[0]

        if mag_np.shape[0] != imsize:
            print "shape mismatch, continuing"
            print f
            continue

        if os.path.isfile(input_dir+ls_seg):
            print "Found 3d seg {}".format(ls_seg)
            c_seg = utility.VTKPDReadAndReorder(input_dir+ls_seg)
            c_seg = c_seg[:,:2]
            contours_seg.append(c_seg)

            mseg = mag.replace('truth','seg3d')
            pseg = mag.replace('truth.mag','seg3d.pot')

            msegnp = utility.VTKSPtoNumpyFromFile(input_dir+mseg)[0]
            psegnp = utility.VTKSPtoNumpyFromFile(input_dir+pseg)[0]

            im_seg[count,:,:] = psegnp
            mag_seg[count,:,:] = msegnp

        elif os.path.isfile(input_dir+ls_edge):
            contours_seg.append(contour_edge)
            im_seg[count,:,:] = mag_np
            mag_seg[count,:,:] = mag_np
        else:
            contours_seg.append(contour_image)
            im_seg[count,:,:] = mag_np
            mag_seg[count,:,:] = mag_np
        #segmentations.append(seg)
        #images.append(mag_np)
        segmentations[count,:,:] = seg
        images[count,:,:] = mag_np

        contours.append(contour)
        contours_ls.append(contour_image)
        contours_edge.append(contour_edge)

        meta_data[0].append(spacing)
        meta_data[1].append(origin)
        meta_data[2].append(dims)
        names.write(f+'\n')
        #convert file to just file name
        file_model = f.split('.')[0]
        for k in split_models.keys():
            if any(file_model in s for s in split_models[k]):
                split_inds[k].append(count)
        count+=1

# segmentations = np.asarray(segmentations)
# if len(segmentations.shape) != 3:
#     segmentations = segmentations.reshape((-1,segmentations[0].shape[0],segmentations[0].shape[1]))
# images = np.asarray(images)
# if len(images.shape) != 3:
#     image = images.reshape((-1,images[0].shape[0],images[0].shape[1]))
meta_data = np.asarray(meta_data)
names.close()

f = open(input_dir+'names.txt')
s = f.readlines()
for k in dirs.keys():
    group_names = open(dirs[k]+'names.txt','w')
    group_names.writelines([s[i] for i in split_inds[k]])
    group_names.close()
    np.save(dirs[k]+'segmentations', segmentations[split_inds[k]])
    np.save(dirs[k]+'images', images[split_inds[k]])
    np.save(dirs[k]+'images_seg', im_seg[split_inds[k]])
    np.save(dirs[k]+'mag_seg', mag_seg[split_inds[k]])
    np.save(dirs[k]+'metadata', meta_data[:,split_inds[k],:])
    np.save(dirs[k]+'contours', [contours[i] for i in split_inds[k]])
    np.save(dirs[k]+'ls_image', [contours_ls[i] for i in split_inds[k]])
    np.save(dirs[k]+'ls_edge', [contours_edge[i] for i in split_inds[k]])
    np.save(dirs[k]+'ls_seg', [contours_seg[i] for i in split_inds[k]])

f = open(dirs['train']+'train.txt','w')
for m in split_models['train']:
    f.write(m+'\n')
f.close()

f = open(dirs['val']+'val.txt','w')
for m in split_models['val']:
    f.write(m+'\n')
f.close()

f = open(dirs['test']+'test.txt','w')
for m in split_models['test']:
    f.write(m+'\n')
f.close()
#
# else:
#     images = np.load(outputDir+'images.npy')
#     segmentations = np.load(outputDir+'segmentations.npy')
#     meta_data = np.load(outputDir+'metadata.npy')
#     contours = np.load(outputDir+'contours.npy')
#     contours_ls = np.load(outputDir+'ls_image.npy')

# if plot:
#     for i in range(0,1):
#         index = np.random.randint(len(segmentations))
#         utility.heatmap(segmentations[index], fn='./plots/seg{}.html'.format(i))
#         utility.heatmap(images[index],
#         fn='./plots/mag{}.html'.format(i))
#
#         spacing = meta_data[0][index]
#         origin = meta_data[1][index]
#         dims = meta_data[2][index]
#
#         segCon = utility.segToContour(segmentations[index],
#         origin,
#         spacing)[0]
#
#         utility.plot_data_plotly([segCon[:,0]], [segCon[:,1]], ['segcon'],
#         fn='./plots/segcon{}.html'.format(i))
#
#         c = contours[index]
#         utility.plot_data_plotly([c[:,0]], [c[:,1]], ['truth_con'],
#         fn='./plots/truth_con{}.html'.format(i))
#
#         c_ls = contours_ls[index]
#         utility.plot_data_plotly([c_ls[:,0]], [c_ls[:,1]], ['ls_con'],
#         fn='./plots/ls_con{}.html'.format(i))
