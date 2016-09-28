set(CMAKE_CXX_FLAGS "-std=c++11" CACHE STRING "")
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "")
set(CTK_QT_VERSION "5" CACHE STRING "")
set(CMAKE_PREFIX_PATH "/home/marsdenlab/Qt5.4.0/5.4/gcc_64" CACHE PATH "")
set(CTK_BUILD_SHARED_LIBS ON CACHE BOOL "")
set(CTK_BUILD_EXAMPLES ON CACHE BOOL "")
set(BUILD_TESTING OFF CACHE BOOL "")

set(CTK_BUILD_ALL ON CACHE BOOL "")
set(CTK_LIB_Scripting/Python/Core_PYTHONQT_USE_VTK ON CACHE BOOL "")
set(CTK_LIB_Scripting/Python/Core_PYTHONQT_WRAP_QTALL ON CACHE BOOL "")
#set(CTK_LIB_Visualization/VTK/Widgets_USE_TRANSFER_FUNCTION_CHARTS ON CACHE BOOL "")

set(VTK_DIR "/home/marsdenlab/projects/SV3/VTK/build" CACHE PATH "")
set(ITK_DIR "/home/marsdenlab/projects/SV3/ITK/build" CACHE PATH "")
