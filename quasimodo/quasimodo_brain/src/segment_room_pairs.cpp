#include "ros/ros.h"
#include "std_msgs/String.h"
#include <sensor_msgs/PointCloud2.h>
#include <string.h>

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include "eigen_conversions/eigen_msg.h"
#include "tf_conversions/tf_eigen.h"

#include "metaroom_xml_parser/simple_xml_parser.h"
#include "metaroom_xml_parser/simple_summary_parser.h"

#include <tf_conversions/tf_eigen.h>

#include "ros/ros.h"
#include <metaroom_xml_parser/load_utilities.h>
#include <pcl_ros/point_cloud.h>
#include <cv_bridge/cv_bridge.h>

#include "metaroom_xml_parser/load_utilities.h"
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_ros/transforms.h>

#include <tf_conversions/tf_eigen.h>

#include "quasimodo_msgs/model.h"
#include "quasimodo_msgs/rgbd_frame.h"
#include "quasimodo_msgs/model_from_frame.h"
#include "quasimodo_msgs/index_frame.h"
#include "quasimodo_msgs/fuse_models.h"
#include "quasimodo_msgs/get_model.h"
#include "quasimodo_msgs/segment_model.h"
#include "quasimodo_msgs/metaroom_pair.h"

#include "ros/ros.h"
#include <quasimodo_msgs/query_cloud.h>
#include <quasimodo_msgs/visualize_query.h>
#include <metaroom_xml_parser/load_utilities.h>
#include <pcl_ros/point_cloud.h>
#include <cv_bridge/cv_bridge.h>

#include "metaroom_xml_parser/load_utilities.h"
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_ros/transforms.h>

#include "quasimodo_msgs/model.h"
#include "quasimodo_msgs/rgbd_frame.h"
#include "quasimodo_msgs/model_from_frame.h"
#include "quasimodo_msgs/index_frame.h"
#include "quasimodo_msgs/fuse_models.h"
#include "quasimodo_msgs/get_model.h"

#include "metaroom_xml_parser/simple_xml_parser.h"

#include <image_geometry/pinhole_camera_model.h>
#include <sensor_msgs/CameraInfo.h>

#include "modelupdater/ModelUpdater.h"
#include "core/RGBDFrame.h"
#include "Util/Util.h"

ros::ServiceClient segmentation_client;


using namespace std;

bool segment_metaroom(quasimodo_msgs::metaroom_pair::Request  & req, quasimodo_msgs::metaroom_pair::Response & res){
	printf("segment_metaroom\n");

	printf("background: %s\n",req.background.c_str());
	printf("foreground: %s\n",req.foreground.c_str());

	reglib::Model * bg_model = quasimodo_brain::load_metaroom_model(req.background);
	reglib::Model * fg_model = quasimodo_brain::load_metaroom_model(req.foreground);

	quasimodo_msgs::segment_model sm;
	sm.request.backgroundmodel = quasimodo_brain::getModelMSG(bg_model);
	sm.request.models.push_back(quasimodo_brain::getModelMSG(fg_model));

	bool status;

	if (segmentation_client.call(sm)){
		if(sm.response.dynamicmasks.size() > 0){
			res.dynamicmasks	= sm.response.dynamicmasks.front().images;
			res.movingmasks		= sm.response.movingmasks.front().images;
		}
		status = true;
	}else{
		ROS_ERROR("Failed to call service segment_model");
		status = false;
	}

	bg_model->fullDelete();
	delete bg_model;
	fg_model->fullDelete();
	delete fg_model;
	return status;
}

int main(int argc, char** argv){
	ROS_INFO("starting segment_room_pairs");
	ros::init(argc, argv, "segmentationserver_metaroom");
	ros::NodeHandle n;
	segmentation_client = n.serviceClient<quasimodo_msgs::segment_model>("segment_model");
	ros::ServiceServer service = n.advertiseService("segment_metaroom", segment_metaroom);

	ROS_INFO("running segment_room_pairs");
	ros::spin();
	/*
    reglib::RegistrationRandom *	reg	= new reglib::RegistrationRandom();

	viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer ("3D Viewer"));
	viewer->setBackgroundColor (0.5, 0, 0.5);
	viewer->addCoordinateSystem (1.0);
	viewer->initCameraParameters ();

    for(int ar = 1; ar < argc; ar++){
        string overall_folder = std::string(argv[ar]);
        vector<string> sweep_xmls = semantic_map_load_utilties::getSweepXmls<PointType>(overall_folder);
        printf("sweep_xmls\n");
        for (auto sweep_xml : sweep_xmls) {
            printf("sweep_xml: %s\n",sweep_xml.c_str());
            load2(sweep_xml);
        }
    }

	//Not needed if metaroom well calibrated
    reglib::MassRegistrationPPR2 * bgmassreg = new reglib::MassRegistrationPPR2(0.01);
    bgmassreg->timeout = 20;
    bgmassreg->viewer = viewer;
    bgmassreg->use_surface = true;
    bgmassreg->use_depthedge = false;
    bgmassreg->visualizationLvl = 0;
    bgmassreg->maskstep = 10;
    bgmassreg->nomaskstep = 10;
    bgmassreg->nomask = true;
    bgmassreg->stopval = 0.0005;
    bgmassreg->setData(models.front()->frames,models.front()->modelmasks);
    reglib::MassFusionResults bgmfr = bgmassreg->getTransforms(models.front()->relativeposes);
    delete bgmassreg;


	ros::init(argc, argv, "test_segment");
	ros::NodeHandle n;
	ros::ServiceClient segmentation_client = n.serviceClient<quasimodo_msgs::segment_model>("segment_model");

	for(unsigned int i = 1; i < models.size(); i++){
		quasimodo_msgs::segment_model sm;
		sm.request.models.push_back(quasimodo_brain::getModelMSG(models[i]));

		if(i > 0){
			sm.request.backgroundmodel = quasimodo_brain::getModelMSG(models[i-1]);
		}

		if (segmentation_client.call(sm)){//Build model from frame
			//int model_id = mff.response.model_id;
			printf("segmented: %i\n",i);
		}else{ROS_ERROR("Failed to call service segment_model");}
	}
	//ros::spin();

    delete reg;
    for(size_t j = 0; j < models.size(); j++){
        models[j]->fullDelete();
        delete models[j];
    }
	printf("done\n");
	*/
	return 0;
}
