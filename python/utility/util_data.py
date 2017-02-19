import utility
import numpy as np
from tqdm import tqdm
class VascData2D:

    def __init__(self, dataDir, normalize='max'):
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
        print 'loading data'
        self.images = np.load(self.imString)
        self.images = self.images.astype(float)
        self.segs = np.load(self.segString)
        self.meta = np.load(self.metaString)
        self.contours = np.load(self.contourString)
        self.contours_ls = np.load(self.ls_string)
        #self.contours_edge = np.load(self.ls_edge)
        #self.contours_seg = np.load(self.ls_seg)
        #self.mag_seg = np.load(self.mag_seg_str)
        #self.im_seg = np.load(self.im_seg_str)
        self.data_dims = self.images.shape
        data_dims = self.data_dims

        if data_dims[1] == 1:
            self.images = self.images[:,0,:,:]
            self.data_dims = self.images.shape
            print self.data_dims


        #Keras/Tf require channel dimension so need to add this
        print 'reshaping and normalizing images'
        self.images_tf = self.images.reshape((self.data_dims[0],
            self.data_dims[1], self.data_dims[2],1))
        self.segs_tf = self.segs.reshape((self.data_dims[0],
            self.data_dims[1], self.data_dims[2],1))
        #self.images_norm = utility.normalize_images(self.images_tf, normalize)
        self.images_norm = self.images_tf
        # self.mag_seg_tf = self.mag_seg.reshape((self.data_dims[0],
        #     self.data_dims[1], self.data_dims[2],1))
        #self.im_seg_tf = self.im_seg.reshape((self.data_dims[0],
        #    self.data_dims[1], self.data_dims[2],1))

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
