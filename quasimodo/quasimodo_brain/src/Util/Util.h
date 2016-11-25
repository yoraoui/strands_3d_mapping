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

#include "metaroom_xml_parser/simple_xml_parser.h"
#include <metaroom_xml_parser/load_utilities.h>

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

#include <soma_llsd/GetScene.h>
#include <soma_llsd/InsertScene.h>
#include <soma_llsd_msgs/Segment.h>
#include <quasimodo_conversions/conversions.h>

namespace quasimodo_brain {

std::string initSegment(ros::NodeHandle& n, reglib::Model * model);

soma_llsd_msgs::Scene getScene(ros::NodeHandle& n, reglib::RGBDFrame * frame, std::string current_waypointid = "", std::string roomRunNumber = "");

reglib::Camera * getCam(sensor_msgs::CameraInfo & info);
reglib::RGBDFrame * getFrame(soma_llsd_msgs::Scene & scene);


std::vector<reglib::superpoint> getSuperPoints(std::string path);

std::vector<reglib::superpoint> getRoomSuperPoints(std::string path, std::string savePath);

void transformSuperPoints(std::vector<reglib::superpoint> & spvec, Eigen::Matrix4d cp);

void saveSuperPoints(std::string path, std::vector<reglib::superpoint> & spvec, Eigen::Matrix4d pose, float ratio_keep = 0.1);

std::vector<Eigen::Matrix4d> readPoseXML(std::string xmlFile);

void savePoses(std::string xmlFile, std::vector<Eigen::Matrix4d> poses, int maxposes = -1);

Eigen::Matrix4d getPose(QXmlStreamReader * xmlReader);

int readNumberOfViews(std::string xmlFile);

void writeXml(std::string xmlFile, std::vector<reglib::RGBDFrame *> & frames, std::vector<Eigen::Matrix4d> & poses);

void writePose(QXmlStreamWriter* xmlWriter, Eigen::Matrix4d pose);

void remove_old_seg(std::string sweep_folder);

std::string replaceAll(std::string str, std::string from, std::string to);

void cleanPath(std::string & path);
void sortXMLs(std::vector<std::string> & sweeps);

std::vector<Eigen::Matrix4f> getRegisteredViewPosesFromFile(std::string poses_file, int no_transforms);
reglib::Model * loadFromRaresFormat(std::string path);

double getTime();
reglib::Model * getModelFromMSG(quasimodo_msgs::model & msg, bool compute_edges = true);

void addToModelMSG(quasimodo_msgs::model & msg, reglib::Model * model, Eigen::Affine3d rp = Eigen::Affine3d::Identity(), bool addClouds = false);
quasimodo_msgs::model getModelMSG(reglib::Model * model, bool addClouds = false);

std::vector<Eigen::Matrix4f> getRegisteredViewPoses(const std::string& poses_file, const int& no_transforms);

Eigen::Matrix4d getMat(tf::StampedTransform tf);
reglib::Model * load_metaroom_model(std::string sweep_xml, std::string savePath = "");

void segment(std::vector< reglib::Model * > bgs, std::vector< reglib::Model * > models, std::vector< std::vector< cv::Mat > > & internal, std::vector< std::vector< cv::Mat > > & external, std::vector< std::vector< cv::Mat > > & dynamic, int debugg = 0, std::string savePath = "");
std::vector<reglib::Model *> loadModelsXML(std::string path);
std::vector<reglib::Model *> loadModelsPCDs(std::string path);

}

#endif
