cmake_minimum_required(VERSION 2.8.3)
project(object_manager_msgs)

find_package(catkin REQUIRED COMPONENTS std_msgs sensor_msgs geometry_msgs message_generation)

add_message_files(
  FILES
  DynamicObjectTracks.msg
  DynamicObjectTrackingData.msg
)

add_service_files(
   FILES
   DynamicObjectsService.srv
   GetDynamicObjectService.srv
   DynamicObjectComputeMaskService.srv
   ProcessDynamicObjectService.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs  # Or other packages containing msgs
  sensor_msgs
  geometry_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES object_manager_msgs
  CATKIN_DEPENDS message_runtime
#  DEPENDS system_lib
)
