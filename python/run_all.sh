#!/bin/bash

#options
python train.py FCN options${1}.cfg

#python train.py OBP_FCN options${1}.cfg

#python train.py OBG_FCN options${1}.cfg

python train.py HED options${1}.cfg

python train.py I2INet options${1}.cfg

#python train.py FC_branch options${1}.cfg

#python train.py FCN_finetune options${1}.cfg

#python train.py FCN_multi options${1}.cfg

python train.py ConvFC options${1}.cfg

python evaluate.py options${1}.cfg

python make_models.py options${1}.cfg

#python vtkScreens.py options${1}.cfg
# #options2
# python train.py FCN options2.cfg
#
# python train.py OBP_FCN options2.cfg
#
# python train.py OBG_FCN options2.cfg
#
# python train.py HED options2.cfg
#
# python train.py I2INet options2.cfg
#
# #options3
# python train.py FCN options3.cfg
#
# python train.py OBP_FCN options3.cfg
#
# python train.py OBG_FCN options3.cfg
#
# python train.py HED options3.cfg
#
# python train.py I2INet options3.cfg
#
# #options4
# python train.py FCN options4.cfg
#
# python train.py OBP_FCN options4.cfg
#
# python train.py OBG_FCN options4.cfg
#
# python train.py HED options4.cfg
#
# python train.py I2INet options4.cfg
#
# #options5
# python train.py FCN options5.cfg
#
# python train.py OBP_FCN options5.cfg
#
# python train.py OBG_FCN options5.cfg
#
# python train.py HED options5.cfg
#
# python train.py I2INet options5.cfg
#
# #options6
# python train.py FCN options6.cfg
#
# python train.py OBP_FCN options6.cfg
#
# python train.py OBG_FCN options6.cfg
#
# python train.py HED options6.cfg
#
# python train.py I2INet options6.cfg
