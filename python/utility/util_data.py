import utility
import numpy as np
from tqdm import tqdm
from scipy.ndimage import rotate
from scipy.ndimage.interpolation import shift
import os
class VascData2D:

    def __init__(self, dataDir, normalize='global_max', rotate_data=True):
        '''
        stores data strings and loads 2D vascular datasets

        args:
            @a data_dir (string): directory containing .npy files of 2d vascular data
        '''
        self.data_dir = dataDir
        self.imString = dataDir + 'images.npy'
        self.segString = dataDir + 'segmentations.npy'
        self.metaString = dataDir + 'metadata.npy'
        self.contourString = dataDir + 'contours.npy'
        self.ls_string = dataDir + 'ls_image.npy'
        self.ls_edge = dataDir + 'ls_edge.npy'
        self.ls_seg = dataDir+'ls_seg.npy'
        #self.im_seg_str = dataDir+'im_seg.npy'
        self.mag_seg_str = dataDir+'mag_seg.npy'
        self.normalize = normalize
        self.names = open(dataDir+'names.txt').readlines()
        self.img_names = [k.split('.')[0] for k in self.names]
        # self.ct = open(dataDir+'../../../ct_images.list').readlines()
        # self.ct = [k.replace('\n','') for k in self.ct]
        # self.mr = open(dataDir+'../../../mr_images.list').readlines()
        # self.mr = [k.replace('\n','') for k in self.mr]
        #
        # self.ct_inds = [i for i in range(len(self.names)) \
        #     if any([self.img_names[i] == k for k in self.ct])]
        #
        # self.mr_inds = [i for i in range(len(self.names)) \
        #     if any([self.img_names[i] == k for k in self.mr])]

        print 'loading data'
        if os.path.isfile(self.imString):
            self.images = np.load(self.imString)
            self.images = self.images.astype(float)
            self.max = np.amax(self.images)
            self.min = np.amin(self.images)
        if os.path.isfile(self.segString):
            self.segs = np.load(self.segString)
        if os.path.isfile(self.metaString):
            self.meta = np.load(self.metaString)
        if os.path.isfile(self.contourString):
            self.contours = np.load(self.contourString)
        if os.path.isfile(self.ls_string):
            self.contours_ls = np.load(self.ls_string)

        #self.contours_edge = np.load(self.ls_edge)
        #self.contours_seg = np.load(self.ls_seg)
        if os.path.isfile(self.mag_seg_str):
            self.mag_seg = np.load(self.mag_seg_str)
            self.mag_seg[:,0:15,:] = 0
            self.mag_seg[:,:,0:15] = 0
            self.mag_seg[:,45:,:] = 0
            self.mag_seg[:,:,45:] = 0

        # if rotate_data:
        #     self.images = self.rotate_images(self.images)
        #     self.segs = self.rotate_images(self.segs)
        #     self.mag_seg = self.rotate_images(self.mag_seg)
        #self.im_seg = np.load(self.im_seg_str)
        self.data_dims = self.images.shape
        data_dims = self.data_dims

        if data_dims[1] == 1:
            self.images = self.images[:,0,:,:]
            self.data_dims = self.images.shape
            print self.data_dims

        # #self.normalize_modality()
        # #Keras/Tf require channel dimension so need to add this
        # print 'reshaping and normalizing images'
        # self.images_tf = self.images.reshape((self.data_dims[0],
        #     self.data_dims[1], self.data_dims[2],1))
        # self.segs_tf = self.segs.reshape((self.data_dims[0],
        #     self.data_dims[1], self.data_dims[2],1))
        # self.images_norm = utility.normalize_images(self.images_tf, normalize)
        # #self.images_norm = self.images_tf
        # if os.path.isfile(self.mag_seg_str):
        #     self.mag_seg_tf = self.mag_seg.reshape((self.data_dims[0],
        #          self.data_dims[1], self.data_dims[2],1))
        #self.im_seg_tf = self.im_seg.reshape((self.data_dims[0],
        #    self.data_dims[1], self.data_dims[2],1))
        # self.images = None
        # self.segs = None

    def createOBG(self,border_width=1):
        '''
        converts segmentations to background/object/boundary labels
        self.obg.shape = (N,W,H,3)
        args:
            @a border_width (int) - boundary width to use
        '''
        print 'creating OBG data'
        dims = self.images_norm.shape
        self.obg = np.zeros((dims[0],dims[1],dims[2],3))

        for i in tqdm(range(0,len(self.segs))):
            self.obg[i,:] = utility.segToOBG(self.segs_tf[i],border_width)

    # def normalize_modality(self):
    #     xct = self.images[self.ct_inds]
    #     bone = xct > 700
    #     xct[bone] = 0
    #     self.images[self.ct_inds] = xct
    #
    #     #ctmax = np.amax(self.images[self.ct_inds])
    #     #ctmin = np.amin(self.images[self.ct_inds])
    #     #self.images[self.ct_inds] = (self.images[self.ct_inds]-ctmin)/(ctmax-ctmin)
    #
    #     #mrmax = np.amax(self.images[self.mr_inds])
    #     #mrmin = np.amin(self.images[self.mr_inds])
    #     #self.images[self.mr_inds] = (self.images[self.mr_inds]-mrmin)/(mrmax-mrmin)

    def rotate_images(self,x, gen_angles=True):
        if gen_angles:
            self.angles = np.random.randint(360, size=x.shape[0])
        xrot = np.zeros(x.shape)
        for i in range(len(self.angles)):
            xrot[i] = rotate(x[i],self.angles[i], axes=(1,0), reshape=False)

        x = np.vstack((x,xrot))

        return x

    def translate_images(self,x,translate, gen_moves=True):
        N = x.shape[0]
        if gen_moves:
            self.moves = np.random.randint(-translate, translate,size=(N,2))

        xret = np.zeros(x.shape)
        for i in range(N):
            xret[i] = np.roll(x[i],(self.moves[i][0],self.moves[i][1]), axis=(1,2))

        return np.vstack((x,xret))

    def get_subset(self, N, rotate=False, translate=None, crop=None):

        inds = np.random.choice(self.data_dims[0], size=N, replace=False)
        x = self.images[inds]
        x = (x-self.min)/(self.max-self.min)
        s = x.shape
        x = x.reshape((s[0],s[1],s[2],1))
        y = self.segs[inds]
        y = y.reshape((s[0],s[1],s[2],1))

        if rotate:
            x = self.rotate_images(x)
            y = self.rotate_images(y, gen_angles=False)

        if translate != None:
            x = self.translate_images(x, translate)
            y = self.translate_images(y, translate, gen_moves=False)

        if crop != None:
            x = x[:,s[1]/2-crop/2:s[1]/2+crop/2,s[1]/2-crop/2:s[1]/2+crop/2]
            y = y[:,s[1]/2-crop/2:s[1]/2+crop/2,s[1]/2-crop/2:s[1]/2+crop/2]

        return (x,y)
