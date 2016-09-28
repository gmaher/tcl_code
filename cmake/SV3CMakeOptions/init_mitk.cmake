#set(CMAKE_BUILD_TYPE "Release" CACHE STRING "")
set(CMAKE_CXX_FLAGS "-std=c++11" CACHE STRING "")
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "")
set(BUILD_SHARED_LIBS ON CACHE BOOL "")
set(BUILD_TESTING OFF CACHE BOOL "")
set(CMAKE_PREFIX_PATH "/home/marsdenlab/Qt5.4.0/5.4/gcc_64" CACHE PATH "")
set(MITK_USE_Python ON CACHE BOOL "")
set(MITK_USE_SYSTEM_PYTHON ON CACHE BOOL "")

set(EXTERNAL_CTK_DIR "/home/marsdenlab/projects/SV3/CTK/build" CACHE PATH "")
set(EXTERNAL_GDCM_DIR "/home/marsdenlab/projects/SV3/GDCM_build" CACHE PATH "")
set(EXTERNAL_ITK_DIR "/home/marsdenlab/projects/SV3/ITK/build" CACHE PATH "")
set(EXTERNAL_SimpleITK_DIR "/home/marsdenlab/projects/SV3/SimpleITK/SuperBuild/build/SimpleITK-build" CACHE PATH "")
set(EXTERNAL_VTK_DIR "/home/marsdenlab/projects/SV3/VTK/build" CACHE PATH "")

set(MITK_INITIAL_CACHE_FILE "/home/marsdenlab/projects/SV3/SimVascular/Code/SuperBuild/MITK_Init.txt" CACHE FILEPATH "")