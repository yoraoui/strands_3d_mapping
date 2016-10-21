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

#include "Util/Util.h"


using namespace std;


bool visualization = false;
boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;

//Assume internally correctly registered...

//Strategy:
//Register frames to background+other frames in the same model DONE
//Determine occlusions inside current model to filter out motion(most likeley people etc) //Done
    //Remove from current model
//Determine occlusion of current model to background
    //These pixels are dynamic objects
//Update background
    //How?

bool segment_model(quasimodo_msgs::segment_model::Request  & req, quasimodo_msgs::segment_model::Response & res){
	printf("segment_model\n");

	std::vector< reglib::Model * > models;
	for(unsigned int i = 0; i < req.models.size(); i++){
		models.push_back(quasimodo_brain::getModelFromMSG(req.models[i]));
	}

	reglib::Model * bg = quasimodo_brain::getModelFromMSG(req.backgroundmodel);

	std::vector< std::vector< cv::Mat > > internal;
	std::vector< std::vector< cv::Mat > > external;
	std::vector< std::vector< cv::Mat > > dynamic;

	std::vector< reglib::Model * > bgs;
	bgs.push_back(bg);

	//quasimodo_brain::segment(bg,models,internal,external,dynamic,visualization);
	quasimodo_brain::segment(bgs,models,internal,external,dynamic,visualization);
	printf("visualization: %i\n",visualization);

	for(unsigned int i = 0; false && visualization && i < models.size(); i++){
		std::vector<cv::Mat> internal_masks = internal[i];
		std::vector<cv::Mat> external_masks = external[i];
		std::vector<cv::Mat> dynamic_masks	= dynamic[i];
		reglib::Model * model = models[i];
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
		for(unsigned int j = 0; j < model->frames.size(); j++){
			reglib::RGBDFrame * frame = model->frames[j];
			unsigned char  * rgbdata		= (unsigned char	*)(frame->rgb.data);
			unsigned short * depthdata		= (unsigned short	*)(frame->depth.data);
			float		   * normalsdata	= (float			*)(frame->normals.data);

			reglib::Camera * camera = frame->camera;

			unsigned char * internalmaskdata = (unsigned char *)(internal_masks[j].data);
			unsigned char * externalmaskdata = (unsigned char *)(external_masks[j].data);
			unsigned char * dynamicmaskdata = (unsigned char *)(dynamic_masks[j].data);

			Eigen::Matrix4d p = model->relativeposes[j];
			float m00 = p(0,0); float m01 = p(0,1); float m02 = p(0,2); float m03 = p(0,3);
			float m10 = p(1,0); float m11 = p(1,1); float m12 = p(1,2); float m13 = p(1,3);
			float m20 = p(2,0); float m21 = p(2,1); float m22 = p(2,2); float m23 = p(2,3);

			const float idepth			= camera->idepth_scale;
			const float cx				= camera->cx;
			const float cy				= camera->cy;
			const float ifx				= 1.0/camera->fx;
			const float ify				= 1.0/camera->fy;
			const unsigned int width	= camera->width;
			const unsigned int height	= camera->height;

			for(unsigned int w = 0; w < width;w++){
				for(unsigned int h = 0; h < height;h++){
					int ind = h*width+w;
					float z = idepth*float(depthdata[ind]);
					if(z > 0){
						float x = (float(w) - cx) * z * ifx;
						float y = (float(h) - cy) * z * ify;
						pcl::PointXYZRGBNormal point;
						point.x = m00*x + m01*y + m02*z + m03;
						point.y = m10*x + m11*y + m12*z + m13;
						point.z = m20*x + m21*y + m22*z + m23;
						point.b = rgbdata[3*ind+0];
						point.g = rgbdata[3*ind+1];
						point.r = rgbdata[3*ind+2];
						if(dynamicmaskdata[ind] != 0){
							point.b = 0;
							point.g = 255;
							point.r = 0;
						}else if(internalmaskdata[ind] == 0){
							point.b = 0;
							point.g = 0;
							point.r = 255;
						}
						cloud->points.push_back(point);
					}
				}
			}
		}
		viewer->removeAllPointClouds();
		viewer->addPointCloud<pcl::PointXYZRGBNormal> (cloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(cloud), "scloud");
		viewer->spin();
	}

//	for(unsigned int i = 0; visualization && i < models.size(); i++){
//		std::vector<cv::Mat> internal_masks = internal[i];
//		std::vector<cv::Mat> external_masks = external[i];
//		std::vector<cv::Mat> dynamic_masks	= dynamic[i];
//		reglib::Model * model = models[i];
//		for(unsigned int j = 0; j < model->frames.size(); j++){
//			cv::namedWindow( "rgb", cv::WINDOW_AUTOSIZE );		cv::imshow( "rgb",		models[i]->frames[j]->rgb);
//			cv::namedWindow( "moving", cv::WINDOW_AUTOSIZE );	cv::imshow( "moving",	255*(internal[i][j] > 0));
//			cv::namedWindow( "dynamic", cv::WINDOW_AUTOSIZE );	cv::imshow( "dynamic",	255*(dynamic[i][j] > 0));
//			cv::waitKey(0);
//		}
//	}
	res.backgroundmodel = req.backgroundmodel;

	res.dynamicmasks.resize(models.size());
	res.movingmasks.resize(models.size());
	for(unsigned int i = 0; i < models.size(); i++){
		for(unsigned int j = 0; j < models[i]->frames.size(); j++){
			cv_bridge::CvImage dynamicmaskBridgeImage;
			dynamicmaskBridgeImage.image			= dynamic[i][j];
			dynamicmaskBridgeImage.encoding			= "mono8";
			res.dynamicmasks[i].images.push_back( *(dynamicmaskBridgeImage.toImageMsg()) );

			cv_bridge::CvImage internalmaskBridgeImage;
			internalmaskBridgeImage.image			= internal[i][j];
			internalmaskBridgeImage.encoding		= "mono8";
			res.movingmasks[i].images.push_back( *(internalmaskBridgeImage.toImageMsg()) );

//			cv::namedWindow( "rgb", cv::WINDOW_AUTOSIZE );		cv::imshow( "rgb",		models[i]->frames[j]->rgb);
//			cv::namedWindow( "moving", cv::WINDOW_AUTOSIZE );	cv::imshow( "moving",	255*(internal[i][j] > 0));
//			cv::namedWindow( "dynamic", cv::WINDOW_AUTOSIZE );	cv::imshow( "dynamic",	255*(dynamic[i][j] > 0));
//			cv::waitKey(0);
		}
		res.models.push_back(quasimodo_brain::getModelMSG(models[i]));
	}


	for(unsigned int i = 0; i < models.size(); i++){
		models[i]->fullDelete();
		delete models[i];
	}
	models.clear();

	bg->fullDelete();
	delete bg;

	return true;
}

int main(int argc, char** argv){
ROS_INFO("starting segmentserver.");
	ros::init(argc, argv, "segmentationserver");
	ros::NodeHandle n;
	for(int i = 1; i < argc;i++){
		printf("input: %s\n",argv[i]);
		if(std::string(argv[i]).compare("-v") == 0){           printf("visualization turned on\n");                visualization = true;}
	}

	if(visualization){
		viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer ("viewer"));
		viewer->addCoordinateSystem(0.01);
		viewer->setBackgroundColor(0.0,0.0,0.0);
	}

	ros::ServiceServer service = n.advertiseService("segment_model", segment_model);
ROS_INFO("Ready to add use segment_model.");

	ros::spin();
}
