set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "")
set(CMAKE_CXX_FLAGS "-fPIC" CACHE BOOL "")

set(SV_USE_QT_GUI ON CACHE BOOL "")
set(CMAKE_PREFIX_PATH "/home/marsdenlab/Qt5.4.0/5.4/gcc_64" CACHE PATH "")

#set(SV_USE_TETGEN ON CACHE BOOL "") #default: on
#set(SV_USE_TET_ADAPTOR ON CACHE BOOL "")
#set(SV_USE_VMTK ON CACHE BOOL "") #default: on

set(SV_USE_SYSTEM_GDCM ON CACHE BOOL "")
set(SV_USE_SYSTEM_VTK ON CACHE BOOL "")
set(SV_USE_SYSTEM_ITK ON CACHE BOOL "")
set(SV_USE_SYSTEM_CTK ON CACHE BOOL "")
set(SV_USE_SYSTEM_SimpleITK ON CACHE BOOL "")
set(SV_USE_SYSTEM_MITK ON CACHE BOOL "")
set(GDCM_DIR "/home/marsdenlab/projects/SV3/GDCM_build" CACHE PATH "")
set(VTK_DIR "/home/marsdenlab/projects/SV3/VTK/build" CACHE PATH "")
set(ITK_DIR "/home/marsdenlab/projects/SV3/ITK/build" CACHE PATH "")
set(CTK_DIR "/home/marsdenlab/projects/SV3/CTK/build" CACHE PATH "")
set(SimpleITK_DIR "/home/marsdenlab/projects/SV3/SimpleITK/SuperBuild/build/SimpleITK-build" CACHE PATH "")
set(MITK_DIR "/home/marsdenlab/projects/SV3/MITK/build/MITK-build" CACHE PATH "")
set(TCL_LIBRARY "/usr/lib/x86_64-linux-gnu/libtcl8.6.so" CACHE FILEPATH "")
set(TCL_LIBRARY "/usr/local/lib/libtcl8.6.so" CACHE PATH "")
#set(TK_LIBRARY "/usr/local/lib/libtk8.6.so" CACHE PATH "")

set(SV_USE_MITK_CONFIG ON CACHE BOOL "")
set(SV_SUPERBUILD OFF CACHE BOOL "")
set(SV_DOWNLOAD_EXTERNALS OFF CACHE BOOL "")
