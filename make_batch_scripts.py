import os
import sys

#Input argument 1: folder to walk through for mha files
#Input argument 2: folder to store batch scripts

script_path = '../batch_scripts/'

def make_script(name,path):
	s =  '''#!/bin/bash
	#SBATCH --output="/home/gdmaher/output/'''+name+'''.out"
	#SBATCH --partition=gpu-shared
	#SBATCH --gres=gpu:1
	#SBATCH -t 48:00:00
	#SBATCH --mem=20000
	cd ''' + path + '''
	pwd
	echo "START"
	#rm -rf tempdir
	python /home/gdmaher/tcl_code/I2INet3DMed.py /home/gdmaher/I2INet3DMed/I2INet3DMed.prototxt /home/gdmaher/I2INet3DMed/I2INet3DMed.caffemodel
	#cp -rf /scratch/$USER/$SLURM_JOB_ID/* /home/gdmaher/copy
	echo "DONE"'''

	file = open(sys.argv[2]+name+'.sh','w')
	file.write(s)
	file.close()

print sys.argv

for root,dir,files in os.walk(sys.argv[1]):
	for f in files:
		if ('-cm.mha' in f) and (not '_E' in f) and ('OSMSC' in f):
			make_script(f.replace('.mha',''),root)
