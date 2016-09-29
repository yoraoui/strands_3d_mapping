cmake_minimum_required(VERSION 2.8.3)
project(semantic_map_msgs)

find_package(catkin REQUIRED COMPONENTS std_msgs sensor_msgs geometry_msgs message_generation)

add_message_files(
  FILES
  RoomObservation.msg
)

add_service_files(
   FILES
   ClearMetaroomService.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES semantic_map_msgs
  CATKIN_DEPENDS message_runtime
#  DEPENDS system_lib
)
