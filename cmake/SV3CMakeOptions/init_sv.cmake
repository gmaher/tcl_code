set(CMAKE_CXX_FLAGS "-std=c++11" CACHE STRING "")

#set(SV_USE_THREEDSOLVER ON CACHE BOOL "")
#set(SV_THREEDSOLVER_USE_VTK ON CACHE BOOL "") #default: off

#set(SV_USE_TETGEN ON CACHE BOOL "") #default: on
set(SV_USE_TET_ADAPTOR ON CACHE BOOL "")
#set(SV_USE_VMTK ON CACHE BOOL "") #default: on

set(SV_USE_SYSTEM_ITK ON CACHE BOOL "")
set(SV_USE_SYSTEM_VTK ON CACHE BOOL "")
set(ITK_DIR "/home/hongzhi/ProgramFiles/ITK-simvascular-patch-4.7.1b" CACHE PATH "")
set(VTK_DIR "/home/hongzhi/ProgramFiles/VTK-simvascular-patch-6.2b" CACHE PATH "")
set(TCL_LIBRARY "/usr/lib/x86_64-linux-gnu/libtcl.so" CACHE PATH "")



