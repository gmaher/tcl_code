import os
import sys

#Input argument 1: folder to walk through for mha files
#Input argument 2: folder to store batch scripts

script_path = '../batch_scripts/'

def make_script(name,path):
	s =  '''#!/bin/bash\n
	#SBATCH --output="/home/gdmaher/output/'''+name+'''.out"\n
	#SBATCH --partition=gpu-shared\n
	#SBATCH --gres=gpu:1\n
	#SBATCH -t 48:00:00\n
	#SBATCH --mem=40000\n
	cd ''' + path + '''\n
	pwd\n
	echo "START"\n
	#rm -rf tempdir\n
	python /home/gdmaher/tcl_code/I2INet3DMed.py\n
	#cp -rf /scratch/$USER/$SLURM_JOB_ID/* /home/gdmaher/copy\n
	echo "DONE"'''

	file = open(sys.argv[2]+name+'.sh','w')
	file.write(s)
	file.close()

print sys.argv

for root,dir,files in os.walk(sys.argv[1]):
	for f in files:
		if ('-cm.mha' in f) and (not '_E' in f) and ('OSMSC' in f):
			make_script(f.replace('.mha',''),root)
