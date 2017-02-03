#!/bin/bash

echo "installing in ${1}"


#########################################
# Copy cmake files
#########################################
cp -r ../cmake/SV3CMakeOptions/ ${1}

for f in $1/SV3CMakeOptions/*; do
  sed -i "s#/home/marsdenlab/projects/SV3/#$1#g" $f
  sed -i "s#/home/marsdenlab#$HOME#g" $f
done

cd $1

#NOTE: Need to add /home/marsdenlab/projects/SV3/sv3_build/Lib to LD_LIBRARY_PATH

#########################################
# Install dependencies
#########################################
sudo apt-get install tcl8.5 tcl8.5-dev libtcl8.5
sudo apt-get install tk8.5 tk8.5-dev libtk8.5
sudo apt-get install libgstreamer-plugins-base0.10-0
sudo apt-get install libwrap0-dev
sudo apt-get install libtiff-tools libtiff5 libtiff5-dev

#########################################
#QT
#########################################
#wget download.qt.io/archive/qt/5.4/5.4.0/qt-opensource-linux-x64-5.4.0.run

# #########################################
# #VTK
# #########################################
git clone https://github.com/SimVascular/VTK.git
cd VTK
git checkout simvascular-patch-6.2b
mkdir build
cd build
cmake -C ../../SV3CMakeOptions/init_vtk.cmake ..
make -j4
cd ../..

# #########################################
# #GDCM
# #########################################
git clone https://github.com/SimVascular/GDCM.git
cd GDCM
git checkout simvascular-patch-2.4.1
cd ..
mkdir GDCM_build
cd GDCM_build
cmake -C ../SV3CMakeOptions/init_gdcm.cmake ../GDCM/
make -j4
cd ..

# #########################################
# #ITK
# #########################################
git clone https://github.com/SimVascular/ITK.git
cd ITK
git checkout simvascular-patch-4.7.1
mkdir build
cd build
cmake -C ../../SV3CMakeOptions/init_itk.cmake ..
make -j4
cd ../..

# #########################################
# #CTK
# #########################################
git clone https://github.com/SimVascular/CTK.git
cd CTK
git checkout simvascular-patch-2016.02.28
mkdir build
cd build
cmake -C ../../SV3CMakeOptions/init_ctk.cmake ..
make -j4
cd ../..

# #########################################
# #SimpleITK
# #########################################
git clone https://github.com/SimVascular/SimpleITK.git
cd SimpleITK
git checkout simvascular-patch-0.8.1
cd SuperBuild
mkdir build
cd build
cmake -C ../../../SV3CMakeOptions/init_simpleitk.cmake ..
make -j4
cd ../../..

# #########################################
# #Simvascular
# #########################################
git clone https://github.com/SimVascular/SimVascular.git
cd SimVascular
#git checkout changes_for_SV3
cd ..

# #########################################
# #Simvascular
# #########################################
git clone https://github.com/lanhzwind/SimVascular.git
cd SimVascular
git checkout workingbranch
cd ..

# #########################################
# #MITK
# #########################################
git clone https://github.com/SimVascular/MITK.git
cd MITK
git checkout simvascular-patch-2016.03.0
mkdir build
cd build
cmake -C ../../SV3CMakeOptions/init_mitk.cmake ..
make -j4
cd ../..

# ##########################################
# # Simvascular Actual build
# ##########################################
mkdir sv3_build
cd sv3_build
cmake -C ../SV3CMakeOptions/init_sv3.cmake ../SimVascular/Code
make -j4
