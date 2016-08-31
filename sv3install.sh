#!/bin/bash

echo "installing in ${1}"

cd $1
#########################################
#QT
#########################################
wget download.qt.io/archive/qt/5.4/5.4.0/qt-opensource-linux-x64-5.4.0.run

#########################################
#VTK
#########################################
git clone https://github.com/SimVascular/VTK.git
cd VTK 
git checkout simvascular-patch-6.2b
cd ..

#########################################
#GDCM
#########################################
git clone https://github.com/SimVascular/GDCM.git
cd GDCM
git checkout simvascular-patch-2.4.1
cd ..

#########################################
#ITK
#########################################
git clone https://github.com/SimVascular/ITK.git
cd ITK
git checkout simvascular-patch-4.7.1
cd ..

#########################################
#CTK
#########################################
git clone https://github.com/SimVascular/CTK.git
cd CTK
git checkout simvascular-patch-2016.02.28
cd ..

#########################################
#MITK
#########################################
git clone https://github.com/SimVascular/MITK.git
cd MITK
git checkout simvascular-patch-2016.03.0
cd ..

#########################################
#SimpleITK
#########################################
git clone https://github.com/SimVascular/SimpleITK.git
cd SimpleITK
git checkout simvascular-patch-0.8.1
cd ..

#########################################
#Simvascular
#########################################
git clone https://github.com/SimVascular/SimVascular.git
cd SimVascular
git checkout changes_for_SV3
cd ..

#########################################
#SV3
#########################################
cd SimVascular/Code
git clone https://github.com/SimVascular/SV3.git
cd ../..