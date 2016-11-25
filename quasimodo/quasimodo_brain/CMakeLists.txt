cmake_minimum_required(VERSION 2.8.3)
project(quasimodo_brain)


find_package(catkin REQUIRED COMPONENTS quasimodo_models roscpp soma_msgs object_manager object_manager_msgs semantic_map_msgs soma_manager local_metric_map_people_rejection quasimodo_msgs quasimodo_conversions quasimodo_models message_runtime pcl_ros metaroom_xml_parser eigen_conversions cv_bridge)


set(CMAKE_CXX_FLAGS "-O4 -g -pg -Wunknown-pragmas -Wno-unknown-pragmas -Wsign-compare -fPIC -std=c++0x -o popcnt -mssse3")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Required for retrieval stuff
include_directories(${catkin_INCLUDE_DIRS})

# This is the root dir of https://github.com/nilsbore/dynamic_object_retrieval
set(retrieval_project_dir ${CMAKE_CURRENT_SOURCE_DIR}/../../dynamic_object_retrieval/)

find_package(OpenGL)

##############################################
##################### INCLUDES ###############
##############################################
#rares library
rosbuild_prepare_qt4(QtCore QtXml)
include_directories(${catkin_INCLUDE_DIRS})
link_directories(${catkin_LIBRARY_DIRS})
link_directories(/home/johane/catkin_ws_dyn/devel/lib)

# PCL
find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

#Opencv stuff
link_directories(/usr/lib/x86_64-linux-gnu)
include_directories(/usr/include/)

#find_package(OpenCV)
#include_directories(${OpenCV_INCLUDE_DIRS})
#target_link_libraries(my_awesome_library ${OpenCV_LIBRARIES})

#QT
find_package(Qt4 REQUIRED)
INCLUDE(${QT_USE_FILE})

#Ceres
FIND_PACKAGE(Ceres REQUIRED)
INCLUDE_DIRECTORIES(${CERES_INCLUDE_DIRS})


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES test_quasimodo
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

#include_directories(/home/johane/catkin_ws/src/g2o/include)
#target_link_libraries(/home/johane/catkin_ws/src/g2o/lib)

include_directories(${catkin_INCLUDE_DIRS})


add_library(quasimodo_brain_util src/Util/Util.cpp)
target_link_libraries(quasimodo_brain_util
					  #${retrieval_libraries}
					  #${benchmark_libraries}
					  ${OpenCV_LIBS}
					  ${QT_QTMAIN_LIBRARY} ${QT_LIBRARIES}
					  ${PCL_LIBRARIES}
					  ${catkin_LIBRARIES})

add_library(quasimodo_ModelDatabase src/ModelDatabase/ModelDatabase.cpp src/ModelDatabase/ModelDatabaseBasic.cpp  src/ModelDatabase/ModelDatabaseRGBHistogram.cpp src/ModelDatabase/ModelDatabaseRetrieval.cpp )
target_link_libraries(quasimodo_ModelDatabase
					  #${retrieval_libraries}
					  #${benchmark_libraries}
                      ${OpenCV_LIBS}
                      ${QT_QTMAIN_LIBRARY} ${QT_LIBRARIES}
                      ${PCL_LIBRARIES}
                      ${catkin_LIBRARIES})

add_executable(			preload_object_data src/preload_object_data.cpp)
target_link_libraries(	preload_object_data ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${QT_LIBRARIES} metaroom_xml_parser)
add_dependencies(		preload_object_data roscpp quasimodo_msgs_generate_messages_cpp)
target_link_libraries(	preload_object_data  image_geometry cpp_common roscpp rosconsole tf_conversions ${QT_QTMAIN_LIBRARY} ${QT_LIBRARIES} ${catkin_LIBRARIES})

add_executable(			robot_listener src/robot_listener.cpp)
target_link_libraries(	robot_listener ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${QT_LIBRARIES} metaroom_xml_parser)
add_dependencies(		robot_listener roscpp quasimodo_msgs_generate_messages_cpp)
target_link_libraries(	robot_listener  image_geometry cpp_common roscpp rosconsole tf_conversions ${QT_QTMAIN_LIBRARY} ${QT_LIBRARIES} ${catkin_LIBRARIES})

add_executable(			modelserver src/modelserver.cpp)
add_dependencies(		modelserver roscpp quasimodo_msgs_generate_messages_cpp)
target_link_libraries(	modelserver quasimodo_superpoint quasimodo_brain_util quasimodo_ModelDatabase  image_geometry cpp_common roscpp rosconsole tf_conversions metaroom_xml_parser ${QT_QTMAIN_LIBRARY} ${QT_LIBRARIES} ${catkin_LIBRARIES})


add_executable(			massregPCD src/massregPCD.cpp)
add_dependencies(		massregPCD roscpp quasimodo_msgs_generate_messages_cpp)
target_link_libraries(	massregPCD quasimodo_ModelDatabase quasimodo_ModelUpdater image_geometry cpp_common roscpp rosconsole tf_conversions metaroom_xml_parser ${QT_QTMAIN_LIBRARY} ${QT_LIBRARIES} ${catkin_LIBRARIES})

add_executable(			voPCD src/voPCD.cpp)
add_dependencies(		voPCD roscpp quasimodo_msgs_generate_messages_cpp)
target_link_libraries(	voPCD quasimodo_brain_util quasimodo_ModelDatabase quasimodo_ModelUpdater image_geometry cpp_common roscpp rosconsole tf_conversions metaroom_xml_parser ${QT_QTMAIN_LIBRARY} ${QT_LIBRARIES} ${catkin_LIBRARIES})


add_executable(			velodyne2 src/velodyne2.cpp)
add_dependencies(		velodyne2 roscpp quasimodo_msgs_generate_messages_cpp)
target_link_libraries(	velodyne2 quasimodo_ModelDatabase quasimodo_ModelUpdater image_geometry cpp_common roscpp rosconsole tf_conversions metaroom_xml_parser ${QT_QTMAIN_LIBRARY} ${QT_LIBRARIES} ${catkin_LIBRARIES})

add_executable(			MeshTest src/MeshTest.cpp)
add_dependencies(		MeshTest roscpp quasimodo_msgs_generate_messages_cpp)
target_link_libraries(	MeshTest quasimodo_ModelDatabase quasimodo_Mesher  image_geometry cpp_common roscpp rosconsole tf_conversions metaroom_xml_parser ${QT_QTMAIN_LIBRARY} ${QT_LIBRARIES} ${catkin_LIBRARIES})

add_executable(			test_align2 src/test_align2.cpp)
add_dependencies(		test_align2 roscpp quasimodo_msgs_generate_messages_cpp)
target_link_libraries(	test_align2 quasimodo_ModelDatabase quasimodo_Mesher  image_geometry cpp_common roscpp rosconsole tf_conversions metaroom_xml_parser ${QT_QTMAIN_LIBRARY} ${QT_LIBRARIES} ${catkin_LIBRARIES})

add_executable(			test_segment src/test_segment.cpp)
add_dependencies(		test_segment roscpp quasimodo_msgs_generate_messages_cpp)
target_link_libraries(	test_segment quasimodo_brain_util quasimodo_ModelDatabase quasimodo_Mesher  image_geometry cpp_common roscpp rosconsole tf_conversions metaroom_xml_parser ${QT_QTMAIN_LIBRARY} ${QT_LIBRARIES} ${catkin_LIBRARIES})

add_executable(			segment_room_pairs src/segment_room_pairs.cpp)
add_dependencies(		segment_room_pairs roscpp quasimodo_msgs_generate_messages_cpp)
target_link_libraries(	segment_room_pairs quasimodo_ModelUpdater quasimodo_brain_util quasimodo_ModelDatabase quasimodo_Mesher  image_geometry cpp_common roscpp rosconsole tf_conversions metaroom_xml_parser ${QT_QTMAIN_LIBRARY} ${QT_LIBRARIES} ${catkin_LIBRARIES})

add_executable(			test_segment_room_pairs src/test_segment_room_pairs.cpp)
add_dependencies(		test_segment_room_pairs roscpp quasimodo_msgs_generate_messages_cpp)
target_link_libraries(	test_segment_room_pairs quasimodo_ModelUpdater quasimodo_brain_util quasimodo_ModelDatabase quasimodo_Mesher  image_geometry cpp_common roscpp rosconsole tf_conversions metaroom_xml_parser ${QT_QTMAIN_LIBRARY} ${QT_LIBRARIES} ${catkin_LIBRARIES})

add_executable(			segmentationserver src/segmentationserver.cpp)
add_dependencies(		segmentationserver roscpp quasimodo_msgs_generate_messages_cpp)
target_link_libraries(	segmentationserver quasimodo_brain_util quasimodo_ModelUpdater quasimodo_ModelDatabase quasimodo_Mesher  image_geometry cpp_common roscpp rosconsole tf_conversions metaroom_xml_parser ${QT_QTMAIN_LIBRARY} ${QT_LIBRARIES} ${catkin_LIBRARIES})

add_executable(			metaroom_additional_view_processing src/metaroom_additional_view_processing.cpp)
add_dependencies(		metaroom_additional_view_processing roscpp quasimodo_msgs_generate_messages_cpp object_manager_msgs_generate_messages_cpp semantic_map_msgs_generate_messages_cpp)
target_link_libraries(	metaroom_additional_view_processing quasimodo_superpoint quasimodo_ModelUpdater quasimodo_brain_util quasimodo_ModelDatabase quasimodo_Mesher image_geometry cpp_common roscpp rosconsole tf_conversions metaroom_xml_parser ${QT_QTMAIN_LIBRARY} ${QT_LIBRARIES} ${catkin_LIBRARIES})

add_executable(			AnnotationTool src/AnnotationTool.cpp)
add_dependencies(		AnnotationTool roscpp quasimodo_msgs_generate_messages_cpp object_manager_msgs_generate_messages_cpp semantic_map_msgs_generate_messages_cpp)
target_link_libraries(	AnnotationTool quasimodo_brain_util quasimodo_ModelDatabase quasimodo_Mesher image_geometry cpp_common roscpp rosconsole tf_conversions metaroom_xml_parser ${QT_QTMAIN_LIBRARY} ${QT_LIBRARIES} ${catkin_LIBRARIES})

add_executable(			pcl_multivisualizer src/pcl_multivisualizer.cpp)
add_dependencies(		pcl_multivisualizer roscpp quasimodo_msgs_generate_messages_cpp object_manager_msgs_generate_messages_cpp semantic_map_msgs_generate_messages_cpp)
target_link_libraries(	pcl_multivisualizer quasimodo_brain_util quasimodo_ModelDatabase quasimodo_Mesher image_geometry cpp_common roscpp rosconsole tf_conversions metaroom_xml_parser ${QT_QTMAIN_LIBRARY} ${QT_LIBRARIES} ${catkin_LIBRARIES})

add_executable(			zoomer src/zoomer.cpp)
add_dependencies(		zoomer roscpp quasimodo_msgs_generate_messages_cpp object_manager_msgs_generate_messages_cpp semantic_map_msgs_generate_messages_cpp)
target_link_libraries(	zoomer quasimodo_brain_util quasimodo_ModelDatabase quasimodo_Mesher image_geometry cpp_common roscpp rosconsole tf_conversions metaroom_xml_parser ${QT_QTMAIN_LIBRARY} ${QT_LIBRARIES} ${catkin_LIBRARIES})


#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

# Mark executables and/or libraries for installation

install(TARGETS zoomer pcl_multivisualizer quasimodo_brain_util quasimodo_ModelDatabase preload_object_data modelserver robot_listener massregPCD voPCD velodyne2 MeshTest test_align2 test_segment segment_room_pairs test_segment_room_pairs segmentationserver metaroom_additional_view_processing AnnotationTool
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

#install(DIRECTORY src/ModelDatabase
#  DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION} # ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#)

#install(DIRECTORY launch
#  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
#)
