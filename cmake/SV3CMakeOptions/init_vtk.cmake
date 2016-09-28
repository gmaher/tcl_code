set(CMAKE_CXX_FLAGS "-std=c++11" CACHE STRING "")

set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "")
set(BUILD_SHARED_LIBS ON CACHE BOOL "")
set(BUILD_EXAMPLES OFF CACHE BOOL "")
set(BUILD_TESTING OFF CACHE BOOL "")

# for non0-windows
set(VTK_USE_SYSTEM_FREETYPE ON CACHE BOOL "")

set(VTK_Group_Qt ON CACHE BOOL "")
set(VTK_QT_VERSION "5" CACHE STRING "")
set(CMAKE_PREFIX_PATH "/home/marsdenlab/Qt5.4.0/5.4/gcc_64" CACHE PATH "")
set(TCL_LIBRARY "/usr/local/lib/libtcl8.6.so" CACHE PATH "")
set(TK_LIBRARY "/usr/local/lib/libtk8.6.so" CACHE PATH "")
set(Module_vtkGUISupportQt ON CACHE BOOL "")
set(Module_vtkGUISupportQtWebkit ON CACHE BOOL "")
set(Module_vtkGUISupportQtSQL ON CACHE BOOL "")
set(Module_vtkRenderingQt ON CACHE BOOL "")

set(VTK_WRAP_PYTHON ON CACHE BOOL "")

set(VTK_Group_Tk ON CACHE BOOL "")
set(VTK_WRAP_TCL ON CACHE BOOL "")

set(VTK_Group_Rendering ON CACHE BOOL "")
set(VTK_Group_StandAlone ON CACHE BOOL "")
set(Module_vtkTestingRendering ON CACHE BOOL "")
set(VTK_MAKE_INSTANTIATORS ON CACHE BOOL "")
