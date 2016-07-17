#ifndef brainUtil_H
#define brainUtil_H

#include <thread>
#include <sys/time.h>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <vector>
#include <string>
#include <iostream>

#include "ros/ros.h"
#include "std_msgs/String.h"
#include <sensor_msgs/PointCloud2.h>
#include <string.h>

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include "eigen_conversions/eigen_msg.h"
#include "tf_conversions/tf_eigen.h"


#include "quasimodo_msgs/segment_model.h"

#include <image_geometry/pinhole_camera_model.h>
#include <sensor_msgs/CameraInfo.h>

#include "modelupdater/ModelUpdater.h"
#include "core/RGBDFrame.h"

namespace quasimodo_brain {

double getTime();
reglib::Model * getModelFromMSG(quasimodo_msgs::model & msg);

void addToModelMSG(quasimodo_msgs::model & msg, reglib::Model * model, Eigen::Affine3d rp = Eigen::Affine3d::Identity());
quasimodo_msgs::model getModelMSG(reglib::Model * model);

}

#endif
