cmake_minimum_required(VERSION 2.8.3)
project(uav_target_tracking)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  rospy
  sensor_msgs
  geometry_msgs
  std_msgs
  nav_msgs
  message_filters
)

set(ENABLE_CUDA ON)
find_package(CUDA REQUIRED)
include(FindCUDA)

set(CUDA_NVCC_FLAGS 
  --generate-code arch=compute_50,code=sm_50
  --generate-code arch=compute_52,code=sm_52
  --generate-code arch=compute_53,code=sm_53
  --generate-code arch=compute_30,code=sm_30
  )

if (${CUDA_VERSION} VERSION_GREATER  "7.0" AND ${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS "5.0")
  set(CUDA_NVCC_FLAGS --ptxas-options=-v -std=c++11 --use_fast_math -lcufft -Xcompiler "-w" -lcublas)
elseif(${CUDA_VERSION} MATCHES "8.0")
  set(CUDA_NVCC_FLAGS --ptxas-options=-v --use_fast_math -lcufft -Xcompiler "-w" -lcublas)
endif()

set(CAFFE_INCLUDEDIR 
  $ENV{CAFFE_ROOT}/include 
  $ENV{CAFFE_ROOT}/distribute/include 
  $ENV{CAFFE_ROOT}/build/include)
set(CAFFE_LINK_LIBRARAY 
  $ENV{CAFFE_ROOT}/build/lib)
include_directories(${CAFFE_INCLUDEDIR})

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -Wextra -pedantic -O3 -Wno-long-long -fno-omit-frame-pointer -ftree-vectorize")

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a run_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
add_service_files(
  FILES
  Detector.srv
)

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  sensor_msgs
  geometry_msgs
  nav_msgs
  )

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
 INCLUDE_DIRS include
 LIBRARIES uav_target_tracking
 CATKIN_DEPENDS roscpp rospy
 DEPENDS system_lib
)

###########
## Build ##
###########

set(PATH_GSLICR ${CMAKE_CURRENT_SOURCE_DIR}/src/drift_compensation/gslicr_engine)
add_subdirectory(${PATH_GSLICR}/ORUtils)
set(GSLICR_LIB
  ${PATH_GSLICR}/gSLICr_Lib/engines/gSLICr_core_engine.h
  ${PATH_GSLICR}/gSLICr_Lib/engines/gSLICr_seg_engine.h
  ${PATH_GSLICR}/gSLICr_Lib/engines/gSLICr_seg_engine_GPU.h
  ${PATH_GSLICR}/gSLICr_Lib/engines/gSLICr_seg_engine_shared.h
  ${PATH_GSLICR}/gSLICr_Lib/engines/gSLICr_core_engine.cpp
  ${PATH_GSLICR}/gSLICr_Lib/engines/gSLICr_seg_engine.cpp
  ${PATH_GSLICR}/gSLICr_Lib/engines/gSLICr_seg_engine_GPU.cu
  ${PATH_GSLICR}/gSLICr_Lib/objects/gSLICr_settings.h
  ${PATH_GSLICR}/gSLICr_Lib/objects/gSLICr_spixel_info.h
  ${PATH_GSLICR}/gSLICr_Lib/gSLICr_defines.h
  ${PATH_GSLICR}/gSLICr_Lib/gSLICr.h

  src/drift_compensation/object_proposal_generation.cpp
  src/drift_compensation/loss_target_pursuit.cpp
  src/drift_compensation/crop_feature_space.cu
  src/drift_compensation/superpixel_cluster_weights.cu
  src/drift_compensation/summed_area_table.cu

  src/drift_compensation/tracking_target_detector.cpp
)


## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
  include ${catkin_INCLUDE_DIRS}
)

## Declare a C++ library
# add_library(uav_target_tracking
#   src/${PROJECT_NAME}/uav_target_tracking.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(uav_target_tracking ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
# add_executable(uav_target_tracking_node src/uav_target_tracking_node.cpp)
link_directories(${CAFFE_LINK_LIBRARAY})
cuda_add_executable(uav_target_tracking_node
  src/uav_target_tracking_node.cpp
  src/kernelized_correlation_filters.cpp
  src/deep_feature_extraction.cpp
  src/cuda/cosine_convolution_kernel.cu
  src/cuda/bilinear_interpolation_kernel.cu
  src/cuda/discrete_fourier_transform_kernel.cu
  src/cuda/gaussian_correlation_kernel.cu
  src/cuda/threadFenceReduction_kernel.cu
  src/cuda/fast_maths_kernel.cu
  src/cuda/spatial_feature_pyramid_kernel.cu

  ${GSLICR_LIB}
)
cuda_add_cufft_to_target(uav_target_tracking_node)
cuda_add_cublas_to_target(uav_target_tracking_node)


## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(uav_target_tracking_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries(uav_target_tracking_node
  ${catkin_LIBRARIES} ${OpenCV_LIBS}  caffe glog
  )

# target_link_libraries(uav_target_tracking_node
#   ${catkin_LIBRARIES}
# )

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS uav_target_tracking uav_target_tracking_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_uav_target_tracking.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
