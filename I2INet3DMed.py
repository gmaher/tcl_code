from __future__ import print_function
def makeEdgeMap(name, inputfile, netFile, caffeModel):
    # coding: utf-8

    # In[1]:


    #get_ipython().magic(u'matplotlib inline')
    import matplotlib.pylab as plt
    import numpy as np
    import SimpleITK as sitk
    import os,sys,h5py,tempfile,re
    caffe_prefix = os.path.expandvars('$HOME/projects/caffe-sv')
    caffe_root = os.path.expandvars('../../')
    caffe_pycaffe = os.path.join(caffe_root,'python')
    if not caffe_pycaffe in sys.path:
        sys.path.append(caffe_pycaffe)
    import caffe
    from caffe.proto import caffe_pb2
    from google.protobuf import text_format

    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'


    npa=np.array
    def sigmoid(x):
        return  1.0 / (1.0 + np.exp(-np.array(x,dtype=float)))


    import pandas


    # In[2]:

    def load_net(net_proto,caffeModel=None):
        f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        f.write(str(net_proto))
        f.close()
        if caffeModel is None:
            return caffe.Net(f.name, caffe.TEST)
        else:
            return caffe.Net(f.name, caffeModel ,caffe.TEST)
    def load_proto(netFile):
        net_proto = caffe_pb2.NetParameter()
        text_format.Merge(open(netFile).read(),net_proto)
        return net_proto


    # In[3]:

    def sitk_to_caffe(img,is_label=False,has_channels=False):
        data = sitk.GetArrayFromImage(img)
        if is_label:
            data_0 = np.zeros_like(data,dtype=float)
            data_0[np.nonzero(data)]=1.0
            data = data_0
        if has_channels:
            data=data.transpose(3,0,1,2)
            data=data[np.newaxis,...]
        elif len(data.shape) < 4:
            data= data[np.newaxis,np.newaxis,...]
        return data
    def sitk_imginfo_dict(img,suffix=''):
        imginfo=dict(info_size=img.GetSize(),
        info_spacing=img.GetSpacing(),
        info_origin=img.GetOrigin(),
        info_direction=img.GetDirection(),
        info_PixelID=img.GetPixelIDValue(),
        info_ndshape=sitk.GetArrayFromImage(img).shape)
        for k in imginfo.keys():
            imginfo[k+suffix] = imginfo.pop(k)
        return imginfo
    def dict_modkeys(d,prefix='',suffix=''):
        for k in d.keys():
            d[prefix+k+suffix] = d.pop(k)
        return d
    def cast_float(img,rescale=True):
        img=sitk.Cast(img,sitk.sitkFloat32)
        if rescale:
            return sitk.RescaleIntensity(img,0,255)
        else:
            return img
    def cast_uint8(img):
        return sitk.Cast(img,sitk.sitkUInt8)
    def cast_int8(img):
        return sitk.Cast(img,sitk.sitkInt8)
    def cast_int16(img):
        return sitk.Cast(img,sitk.sitkInt16)
    def sitk_padflip_image(img,padpre,padpost,rescale=True,swp_idx=0,flip=True):
        img = cast_float(img,rescale=rescale)
        
        img_nd = sitk.GetArrayFromImage(img)
        if flip:
            img_nd=img_nd.swapaxes(2-swp_idx,0)
        img = sitk.GetImageFromArray(img_nd)
        img = sitk.ConstantPad (img,padpre,padpost)
        return img
    def write_h5_set(in_dict,h5_file,txtfile):
        with h5py.File(h5_file, 'w') as f:
            for k,v in in_dict.iteritems():
                try:
                    f[k] = npa(v).astype(float)
                except:
                    f[k] = npa(v)
        with open(txtfile, 'a') as f:
            f.write(h5_file + '\n')
    def basename(fn):
        return os.path.split(os.path.splitext(fn)[0])[1]


    # In[4]:

    #inputfile = '/home/gabriel/projects/OSMSC0087/OSMSC0087-cm.mha'
    #inputfile = '/home/gabriel/projects/weiguang/SU0187_2008_247_33758142.mha'
    #inputfile = '/home/gabriel/projects/tcl_code/models/OSMSC0001/OSMSC0001-cm.mha'
    #netFile='/home/gabriel/projects/caffe-sv/models/I2INet3DMed/I2INet3DMed.prototxt'
    #caffeModel='/home/gabriel/projects/caffe-sv/models/I2INet3DMed/I2INet3DMed.caffemodel'
    #name = "OSMSC001"
    #output_size = npa([96,96,48],dtype=int)[::-1]
    output_size = npa([48,48,48],dtype=int)[::-1]
    overlap = output_size/npa([12,12,12],dtype=int)[::-1]



    outputdir='./EdgeMaps/' + name + '/'
    tempdir='./tempdir/' + name + '/'

    roi=None
    fnbase=basename(inputfile)
    ov = int(((overlap[0]*(output_size[1]*output_size[2]))+
         (overlap[1]*(output_size[0]*output_size[2]))+
          (overlap[2]*(output_size[0]*output_size[1])))/float(np.prod(output_size))*100)
    gpu_id=0
    print("inputfile:",inputfile,"\nbasename",basename(inputfile))
    print("output_size:",output_size,"overlap:",overlap,"roi:",roi if roi is not None else "All")


    # In[5]:

    def CalcuatePre(img,roi,output_size,overlap):

        img_size=np.array(img.GetSize())
        stride = output_size-overlap

        if roi is None:
            roi = npa((0,)*6)
            roi[1::2]=img_size-1

        roi_start = np.maximum(roi[0::2],0)
        roi_end = np.minimum(roi[1::2],img_size)
        roi_size = roi_end-roi_start

        print("Region Start:",roi_start,"Region End:",roi_end,"Size",roi_size)

        num_stride= npa(np.ceil(roi_size/map(float,stride)),dtype=int)
        last_one_start = (num_stride-1)*stride
        last_one_end = last_one_start+output_size
        roi_diff = np.maximum(last_one_end-roi_size,2)
        nums_mod= (roi_diff)/2


        roi_start = roi_start-nums_mod
        roi_end = roi_end+nums_mod

        padpre = [0-min(s,0) for s in roi_start]
        padpost = [max(s,sz)-sz+1 for s,sz in zip (roi_end,img_size)]
        start=roi_start+padpre

        proc_dict=dict(padpre=padpre,
                       padpost=padpost,
                       stride=stride,
                       roi_start=roi_start,
                       roi_end=roi_end,
                       roi_size=roi_size,
                       roi=roi,
                       overlap=overlap,
                       start=start,
                       num_stride=num_stride)
        print("Old Size:",npa(img_size))
        print("New Size:",npa(img_size)+npa(padpre)+npa(padpost))
        
        return proc_dict

    img=sitk.ReadImage(inputfile)
    proc_dict=CalcuatePre(img,roi,output_size,overlap)

    print("Files to write:",np.prod(proc_dict['num_stride']))
    tempinputdir=os.path.join(tempdir,"input")
    outputbasename=os.path.join(tempinputdir,fnbase+"-{:03d}-{:03d}-{:03d}.h5")
    txtfile = os.path.join(tempinputdir,fnbase+".txt")
    if not os.path.exists(tempinputdir): os.makedirs(tempinputdir)
    print(outputbasename.format(0,0,0))
    print(txtfile)


    # In[6]:

    def PreprocessImageAndWrite(img,outputname,txtfile,output_size,
                        proc_dict,imginfo=dict()):
        
        padpre,padpost=proc_dict['padpre'],proc_dict['padpost'],
        num_stride,stride=proc_dict['num_stride'],proc_dict['stride']
        start=proc_dict['start']
        
        imginfo=sitk_imginfo_dict(img,'_org')
       
        imginfo.update(proc_dict)
        
        img = cast_float(img,rescale=False)
        
        temp_padpre,temp_padpost=npa([8, 8, 8]),npa([8, 8, 8])
        
        temp_padpre=np.minimum(temp_padpre,padpre)
        temp_padpost=np.minimum(temp_padpost,padpost)
        img = sitk_padflip_image(img,temp_padpre,temp_padpre,flip=False)
        img = sitk.Normalize(img)*1.0
        
        
        
        ndimg=sitk.GetArrayFromImage(img)
        print("Min",ndimg.min(),"Max:",ndimg.max(),"Mean:",ndimg.mean(),"STD:",ndimg.std())
        padpre-=temp_padpre
        padpost-=temp_padpost
        img = sitk_padflip_image(img,padpre,padpost,rescale=False,flip=False)
        imginfo.update(sitk_imginfo_dict(img,'_pad'))
        num_collect=np.prod(num_stride)
            
        open(txtfile,'w').close()
        dot_on = 5
        for i in range(0,num_stride[0]):
                for j in range(0,num_stride[1]):
                    for k in range(0,num_stride[2]):
                        idx = npa([i,j,k])*stride+start
                        if not all(idx+output_size<=img.GetSize()):
                            print([i,j,k],idx,start,idx+output_size,img.GetSize())
                            raise

                        roi_img = sitk.RegionOfInterest(img, output_size, idx)
                        if not all(r==o for r,o in zip(roi_img.GetSize(),output_size)):
                            print(roi_img.GetSize(),output_size)
                        imginfo_roi=sitk_imginfo_dict(roi_img,'_roi')
                        data = sitk_to_caffe(roi_img)
                        in_dict=dict(image=data,
                                    info_outputsize=output_size,
                                    outputbasename=outputname,
                                    info_idx=idx,
                                    info_inds=[i,j,k])
                        in_dict.update(imginfo)
                        in_dict.update(imginfo_roi)
                        h5_file=outputname.format(i,j,k)
                        write_h5_set(in_dict,h5_file,txtfile)
                        n=np.ravel_multi_index((i,j,k),num_stride)+1
                        if n%dot_on==0 and n > 0: print('.',sep='',end='')
                        if n%(dot_on*20)==0 and n > 0 or n==num_collect: print(' ', n,'/',num_collect,sep='')
        print("Done!",n)
        return imginfo
    imginfo = PreprocessImageAndWrite(img,outputbasename,txtfile,output_size,
                        proc_dict)       


    # In[ ]:




    # In[7]:

    net_proto=load_proto(netFile)
    net_proto.layer[0].hdf5_data_param.source = txtfile
    with open(txtfile,'r') as f:
        img_files= [line.strip() for line in f]


    # In[8]:

    tempoutdir=os.path.join(tempdir,"output")
    outtxtfile = os.path.join(tempoutdir,fnbase+".txt")
    if not os.path.exists(tempoutdir): os.makedirs(tempoutdir)


    # In[9]:

    net=load_net(net_proto,caffeModel)
    with open(net_proto.layer[0].hdf5_data_param.source,'r') as f:
        img_files= [line.strip() for line in f]
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    #caffe.set_mode_cpu()
    imgbname_last=''
    outtxtfiles=dict()
    dot_on=5
    num_collect= len(img_files)
    for n,img_name in enumerate(img_files):
        h5name=img_name
        imgbname=basename('-'.join(h5name.split('-')[:-3]))
        output_subdir=os.path.join(tempoutdir,imgbname)
        
        h5name=img_name.replace(tempinputdir,output_subdir)
        
        if not os.path.exists(output_subdir): os.makedirs(output_subdir)
        if imgbname not in outtxtfiles.keys(): outtxtfiles[imgbname]=outtxtfile

        if(imgbname_last!=imgbname):
            print(imgbname)
        imgbname_last=imgbname
        with h5py.File(img_name,'r') as f:
            write_dict={k:npa(v) for k,v in f.iteritems() if k!='image'}
        net.forward()
        E=net.blobs['score1'].data
        image=net.blobs['image'].data
        in_dict=dict(E=E,**write_dict)
        write_h5_set(in_dict,h5name,outtxtfile)
        if n%dot_on==0 and n > 0: print('.',sep='',end='')
        if n%(dot_on*20)==0 and n > 0 or n==num_collect: print(' ', n,'/',num_collect,sep='')
        sys.stdout.flush()
    print("Done!",n)


    # In[10]:

    net=load_net(net_proto,caffeModel)
    with open(net_proto.layer[0].hdf5_data_param.source,'r') as f:
        img_files= [line.strip() for line in f]

    imgbname_last=''
    outtxtfiles=dict()
    dot_on=5
    num_collect= len(img_files)
    for n,img_name in enumerate(img_files):
        h5name=img_name
        imgbname=basename('-'.join(h5name.split('-')[:-3]))
        output_subdir=os.path.join(tempoutdir,imgbname)
        
        h5name=img_name.replace(tempinputdir,output_subdir)
        
        if not os.path.exists(output_subdir): os.makedirs(output_subdir)
        if imgbname not in outtxtfiles.keys(): outtxtfiles[imgbname]=outtxtfile


    # In[11]:


    for imgbname,outtxtfile in outtxtfiles.iteritems():
        with open(outtxtfile,'r') as f:
            allfiles={os.path.split(line.strip())[1]:line.strip() for line in f}
        
        with h5py.File(allfiles.itervalues().next(), 'r') as f:
            img_size=npa(f['info_ndshape_org'])
            img_size_pad=npa(f['info_ndshape_pad'])
            padpre,padpost=npa(f['padpre']),npa(f['padpost'])
            overlap,stride=npa(f['overlap']),npa(f['stride'])
            regE_=os.path.split(outputbasename)[1].replace('{:03d}','(\d*)').replace(fnbase,"{:s}")
            print(padpre,padpost,overlap,img_size,img_size_pad,)
        
        regE=regE_.format(imgbname)
        print(regE)
        regobj=re.compile(regE)
        fn_nums=dict()
        for fn in allfiles.keys():
            m=regobj.search(fn)
            if m:
                fn_nums[fn]=map(int,m.groups())
            else:
                allfiles.pop(fn)
            
        fnarray=np.empty((npa(fn_nums.values()).max(axis=0)+1),dtype=object)
        for fn,inds in fn_nums.iteritems():
            fnarray[inds[0],inds[1],inds[2]]=fn

       
        


    # In[16]:

    print("overlap: ", overlap)
    print("fnarray: ", fnarray)
    E=np.zeros(img_size_pad)+1
    zz=list()
    for i in range(fnarray.shape[0]):
        Ej=list()
        for j in range(fnarray.shape[1]):
            Ek=list()
            for k in range(fnarray.shape[2]):
                if fnarray[i,j,k] is not None:
                    with h5py.File(os.path.join(allfiles[fnarray[i,j,k]]), 'r') as f:
                        Ei=np.squeeze(np.array(f['E']))
                        idx=npa(f['info_idx'])
                        patch_start=(npa(f['info_idx'])+overlap/2)
                        patch_end=(patch_start+stride)
                        patch_start=patch_start[[2,1,0]]
                        patch_end=patch_end[[2,1,0]]
                        E_shape=E[patch_start[0]:patch_end[0],patch_start[1]:patch_end[1],patch_start[2]:patch_end[2]].shape
                        Ei_shape=Ei[overlap[2]/2:-overlap[2]/2,overlap[1]/2:-overlap[1]/2,overlap[0]/2:-overlap[0]/2].shape
                        if not all(r==o for r,o in zip(E_shape,Ei_shape)):
                            print("start",patch_start,"stop",patch_end,patch_end-patch_start)
                            print(E_shape,Ei_shape)
                            print()
                
                        E[patch_start[0]:patch_end[0],
                          patch_start[1]:patch_end[1],
                          patch_start[2]:patch_end[2]]=sigmoid(Ei)[overlap[2]/2:-overlap[2]/2,
                                                          overlap[1]/2:-overlap[1]/2,
                                                          overlap[0]/2:-overlap[0]/2]
    E=E[padpre[2]:-padpost[2],padpre[1]:-padpost[1],padpre[0]:-padpost[0]]


    # In[14]:

    E=E-E.min()
    E.min()*255,E.max()*255
    itkE=cast_int16(sitk.GetImageFromArray(E*255))
    orgImg=sitk.ReadImage(inputfile)
    itkE.CopyInformation(orgImg)
    imgs=[dict(name='E',**sitk_imginfo_dict(itkE)),dict(name='img',**sitk_imginfo_dict(orgImg))]
    pandas.DataFrame(imgs)


    # In[15]:

    #if not os.path.exists(outputdir): os.makedirs(outputdir)
    #outputname=os.path.join(outputdir,imgbname+'.mha')
    
    outputname = inputfile.replace('.mha','_E.mha')

    print("Writing file:",outputname)
    sitk.WriteImage(itkE,outputname)


# In[ ]:


# ########################################################
# # Actual code to be run 
# ########################################################
inputs = []
#inputs.append('./models/SU0187_2008_247_33758142.mha')
inputs.append('./models/OSMSC0001/OSMSC0001-cm.mha')
#inputs.append('./models/OSMSC0002/OSMSC0002-cm.mha')
#inputs.append('./models/OSMSC0003/OSMSC0003-cm.mha')
#inputs.append('./models/OSMSC0004/OSMSC0004-cm.mha')
#inputs.append('./models/OSMSC0005/OSMSC0005-cm.mha')

names = []
names.append('OSMSC0001')
#names.append('OSMSC0002')
#names.append('OSMSC0003')
#names.append('OSMSC0004')
#names.append('OSMSC0005')


netFile='/home/gabriel/projects/caffe-sv/models/I2INet3DMed/I2INet3DMed.prototxt'
caffeModel='/home/gabriel/projects/caffe-sv/models/I2INet3DMed/I2INet3DMed.caffemodel'

for inputfile,name in zip(inputs,names):
    print("start: ", inputfile, ', ', name)
    makeEdgeMap(name, inputfile, netFile, caffeModel)
    print("end: ", inputfile, ', ', name)
