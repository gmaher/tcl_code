#!/bin/bash

#This script will download jameson's custom branch of simvascular
#along with the required dependencies and compile it

#requirements
#using anaconda version of python

#inputs
# -1 : directory in which to install simvascular and all the
#dependencies

echo changing directory to $1

cd $1

# #################################################
# # TCL/TK 8.6.6
# #################################################
#
# #TCL/TK paths
# #TCL_LIB="${1}tcl8.6.6/unix/libtcl8.6.so"
# TCL_LIB="/usr/local/lib/libtcl8.6.so"
# TCL_INCLUDE="${1}tcl8.6.6/generic"
# #TCL_BIN="${1}tcl8.6.6/unix/tclsh"
# TCL_BIN="/usr/local/bin/tclsh8.6"
#
# #TK_LIB="${1}tk8.6.6/unix/libtk8.6.so"
# TK_LIB="/usr/local/lib/libtk8.6.so"
# TK_INCLUDE="${1}tk8.6.6/generic"
# #TK_WISH="${1}tk8.6.6/unix/wish"
# TK_WISH="/usr/local/bin/wish8.6"
#
# ##################################
# #Get TCL
# ##################################
# wget http://heanet.dl.sourceforge.net/project/tcl/Tcl/8.6.6/tcl8.6.6-src.tar.gz
#
# #Tars into folder tcl8.6.6
# tar -xf  tcl8.6.6-src.tar.gz
# cd tcl8.6.6/unix
# ./configure
# make -j4
# sudo make install
# cd ../..
#
# ##################################
# #get TK
# ##################################
# #wget downloads.sourceforge.net/project/tcl/Tcl/8.6.6/tk8.6.6-src.tar.gz
# wget ftp://ftp.tcl.tk/pub/tcl/tcl8_6/tk8.6.6-src.tar.gz
# #Tars into folder tk8.6.6
# tar -xf tk8.6.6-src.tar.gz
# cd tk8.6.6/unix
# ./configure --with-tcl="${1}tcl8.6.6/unix"
# make -j4
# sudo make install
# cd ../..

################################################
# TCL/TK 8.5.0
################################################

# #TCL/TK paths
TCL_LIB="${1}tcl8_6/unix/libtcl8.6.so"
TCL_INCLUDE="${1}tcl8_6/generic"
TCL_BIN="${1}tcl8_6/unix/tclsh"

TK_LIB="${1}tk8_6/unix/libtk8.6.so"
TK_INCLUDE="${1}tk8_6/generic"
TK_WISH="${1}tk8_6/unix/wish"

#TCL
wget https://github.com/tcltk/tcl/archive/core_8_6_5.tar.gz
tar -xf core_8_6_5.tar.gz
rm core_8_6_5.tar.gz
mv tcl-core_8_6_5/ tcl8_6
cd tcl8_6/unix
./configure
make -j4
#sudo make install
cd ../..

#TK
wget https://github.com/tcltk/tk/archive/core_8_6_5.tar.gz
tar -xf core_8_6_5.tar.gz
rm core_8_6_5.tar.gz
mv tk-core_8_6_5 tk8_6
cd tk8_6/unix
./configure
 make -j4
 #sudo make install
 cd ../..

#########################cd /medi
# Install rest
#########################

#VTK args
VTK_CMAKE_ARGS="-DBUILD_SHARED_LIBS=OFF -DBUILD_TESTING=OFF -DVTK_Group_Tk=ON "
VTK_CMAKE_ARGS+="-DVTK_WRAP_TCL:BOOL=ON "
VTK_CMAKE_ARGS+="-DTCL_INCLUDE_PATH=${TCL_INCLUDE} "
VTK_CMAKE_ARGS+="-DTCL_LIBRARY=${TCL_LIB} "
VTK_CMAKE_ARGS+="-DTCL_TCLSH=${TCL_BIN} "
VTK_CMAKE_ARGS+="-DTK_INCLUDE_PATH=${TK_INCLUDE} "
VTK_CMAKE_ARGS+="-DTK_LIBRARY=${TK_LIB} "
VTK_CMAKE_ARGS+="-DCMAKE_C_FLAGS=-DGLX_GLXEXT_LEGACY -DCMAKE_CXX_FLAGS=-DGLX_GLXEXT_LEGACY"

#VTK_CMAKE_ARGS+="-D Module_vtkWrappingTcl=ON -D VTK_USE_TK=ON"

VTK_DIR="${1}VTK6/build"

#ITK Args
ITK_CMAKE_ARGS="-DBUILD_SHARED_LIBS=OFF "
ITK_CMAKE_ARGS+="-DBUILD_TESTING=OFF "
ITK_CMAKE_ARGS+="-DModule_ITKVtkGlue=ON "
ITK_CMAKE_ARGS+="-DITK_Review=ON "
ITK_CMAKE_ARGS+="-DVTK_DIR=${VTK_DIR}"

ITK_DIR="${1}ITK4.9/build"

#SV Args
SV_CMAKE_ARGS="-DSimVascular_SUPERBUILD=OFF "
SV_CMAKE_ARGS+="-DSimVascular_USE_SYSTEM_ITK=ON -DSimVascular_USE_SYSTEM_VTK=ON "
SV_CMAKE_ARGS+="-DITK_DIR=${ITK_DIR} -DVTK_DIR=${VTK_DIR} "
SV_CMAKE_ARGS+="-DTCL_INCLUDE_PATH=${TCL_INCLUDE} -DTCL_LIBRARY=${TCL_LIB} "
SV_CMAKE_ARGS+="-DTCL_TCLSH=${TCL_BIN} -DTK_LIBRARY=${TK_LIB} "
SV_CMAKE_ARGS+="-DTK_INCLUDE_PATH=${TK_INCLUDE} -DTK_WISH=${TK_WISH}"

###################################
#start script
###################################
#echo VT_CMAKE_ARGS ${VTK_CMAKE_ARGS}
#echo ITK_CMAKE_ARGS ${ITK_CMAKE_ARGS}
#echo SV_CMAKE_ARGS ${SV_CMAKE_ARGS}

#################################
# Get some external dependencies
#################################
sudo apt-get install cmake
sudo apt-get install cmake-curses-gui
sudo apt-get install cmake-gui
sudo apt-get install gcc-multilib build-essential g++ gfortran
sudo apt-get install libmpich2-dev
sudo apt-get install dcmtk
sudo apt-get install libgdcm-tools
sudo apt-get install libglu1-mesa-dev libxt-dev libgl1-mesa-dev
sudo apt-get install glib2.0-dev
conda install -c https://conda.anaconda.org/simpleitk SimpleITK

#################################
#get VTK
#################################
#Note assumes libGL.so is set up properly
wget https://github.com/Kitware/VTK/archive/v6.0.0.tar.gz \
&& tar -xf v6.0.0.tar.gz
mv VTK-6.0.0 VTK6
cd VTK6
mkdir build
cd build
cmake ${VTK_CMAKE_ARGS} ..
make -j4
cd ../..

##################################
#get ITK
##################################
wget https://github.com/InsightSoftwareConsortium/ITK/archive/v4.9.1.tar.gz \
&& tar -xf v4.9.1.tar.gz
mv ITK-4.9.1 ITK4.9
cd ITK4.9
mkdir build
cd build
cmake ${ITK_CMAKE_ARGS} ..
make -j4
cd ../..

#################################
# Get simvascular
#################################
git clone https://github.com/jmerkow/SimVascular.git
cd SimVascular
git checkout simvascularday_mods3
cd ..
mkdir sv_build
cd sv_build
cmake ${SV_CMAKE_ARGS} ../SimVascular/Code
make -j4
