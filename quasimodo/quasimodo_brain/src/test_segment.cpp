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



using namespace std;

typedef pcl::PointXYZRGB PointType;
typedef pcl::PointCloud<PointType> Cloud;
typedef typename Cloud::Ptr CloudPtr;
typedef pcl::search::KdTree<PointType> Tree;
typedef semantic_map_load_utilties::DynamicObjectData<PointType> ObjectData;

using pcl::visualization::PointCloudColorHandlerCustom;

std::vector< std::vector<cv::Mat> > rgbs;
std::vector< std::vector<cv::Mat> > depths;
std::vector< std::vector<cv::Mat> > masks;
std::vector< std::vector<tf::StampedTransform > >tfs;
std::vector< std::vector<Eigen::Matrix4f> > initposes;
std::vector< std::vector< image_geometry::PinholeCameraModel > > cams;

pcl::visualization::PCLVisualizer* pg;
boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;

std::vector<reglib::Model * > models;

reglib::Model * load2(std::string sweep_xml){
	int slash_pos = sweep_xml.find_last_of("/");
	std::string sweep_folder = sweep_xml.substr(0, slash_pos) + "/";
	printf("folder: %s\n",sweep_folder.c_str());

	SimpleXMLParser<PointType> parser;
	SimpleXMLParser<PointType>::RoomData roomData  = parser.loadRoomFromXML(sweep_folder+"/room.xml");

//	projection
//	532.158936 0.000000 310.514310 0.000000
//	0.000000 533.819214 236.842039 0.000000
//	0.000000 0.000000 1.000000 0.000000

	reglib::Model * sweepmodel = 0;

	std::vector<reglib::RGBDFrame * > current_room_frames;
    for (size_t i=0; i < 1 && i<roomData.vIntermediateRoomClouds.size(); i++)
	{

		cv::Mat fullmask;
		fullmask.create(480,640,CV_8UC1);
		unsigned char * maskdata = (unsigned char *)fullmask.data;
		for(int j = 0; j < 480*640; j++){maskdata[j] = 255;}

		reglib::Camera * cam		= new reglib::Camera();//TODO:: ADD TO CAMERAS
		cam->fx = 532.158936;
		cam->fy = 533.819214;
		cam->cx = 310.514310;
		cam->cy = 236.842039;


		cout<<"Intermediate cloud size "<<roomData.vIntermediateRoomClouds[i]->points.size()<<endl;

		printf("%i / %i\n",i,roomData.vIntermediateRoomClouds.size());

		//Transform
		tf::StampedTransform tf	= roomData.vIntermediateRoomCloudTransformsRegistered[i];
		geometry_msgs::TransformStamped tfstmsg;
		tf::transformStampedTFToMsg (tf, tfstmsg);
		geometry_msgs::Transform tfmsg = tfstmsg.transform;
		geometry_msgs::Pose		pose;
		pose.orientation		= tfmsg.rotation;
		pose.position.x		= tfmsg.translation.x;
		pose.position.y		= tfmsg.translation.y;
		pose.position.z		= tfmsg.translation.z;
		Eigen::Affine3d epose;
		tf::poseMsgToEigen(pose, epose);

		reglib::RGBDFrame * frame = new reglib::RGBDFrame(cam,roomData.vIntermediateRGBImages[i],5.0*roomData.vIntermediateDepthImages[i],0, epose.matrix());

        current_room_frames.push_back(frame);
		if(i == 0){
			sweepmodel = new reglib::Model(frame,fullmask);
		}else{
			sweepmodel->frames.push_back(frame);
			sweepmodel->relativeposes.push_back(current_room_frames.front()->pose.inverse() * frame->pose);
			sweepmodel->modelmasks.push_back(new reglib::ModelMask(fullmask));
		}
	}

	//sweepmodel->recomputeModelPoints();
	//printf("nr points: %i\n",sweepmodel->points.size());

    models.push_back(sweepmodel);
	return sweepmodel;
}

//segment(reglib::Model * bg, std::vector< reglib::Model * > models, bool debugg)

int main(int argc, char** argv){

//    reglib::RegistrationRandom *	reg	= new reglib::RegistrationRandom();

//    viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer ("3D Viewer"));
//    viewer->setBackgroundColor (0.5, 0, 0.5);
//    viewer->addCoordinateSystem (1.0);
//    viewer->initCameraParameters ();

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
    bgmassreg->maskstep = 4;
    bgmassreg->nomaskstep = 4;
    bgmassreg->nomask = true;
    bgmassreg->stopval = 0.0005;
    bgmassreg->setData(models.front()->frames,models.front()->modelmasks);
    reglib::MassFusionResults bgmfr = bgmassreg->getTransforms(models.front()->relativeposes);
    delete bgmassreg;

    for(unsigned int i = 0; i < models.size(); i++){
        models[i]->relativeposes = bgmfr.poses;
    }


/*
	ros::init(argc, argv, "test_segment");
	ros::NodeHandle n;
	ros::ServiceClient segmentation_client = n.serviceClient<quasimodo_msgs::segment_model>("segment_model");


	for(unsigned int i = 1; i < models.size(); i++){
		quasimodo_msgs::segment_model sm;
		sm.request.models.push_back(quasimodo_brain::getModelMSG(models[i]));
		if(i > 0){sm.request.backgroundmodel = quasimodo_brain::getModelMSG(models[i-1]);}

		if (segmentation_client.call(sm)){//Build model from frame
			printf("segmented: %i\n",i);
		}else{ROS_ERROR("Failed to call service segment_model");}
	}
*/

	for(unsigned int i = 1; i < models.size(); i++){
		std::vector< reglib::Model * > foreground;
		foreground.push_back(models[i]);
		std::vector< std::vector< cv::Mat > > internal;
		std::vector< std::vector< cv::Mat > > external;
		std::vector< std::vector< cv::Mat > > dynamic;

		std::vector< reglib::Model * > background;
		background.push_back(models[i-1]);

		quasimodo_brain::segment(background,foreground,internal,external,dynamic,false);

//		quasimodo_msgs::segment_model sm;
//		sm.request.models.push_back(quasimodo_brain::getModelMSG(models[i]));
//		if(i > 0){sm.request.backgroundmodel = quasimodo_brain::getModelMSG(models[i-1]);}

//		if (segmentation_client.call(sm)){//Build model from frame
//			printf("segmented: %i\n",i);
//		}else{ROS_ERROR("Failed to call service segment_model");}
	}

    for(size_t j = 0; j < models.size(); j++){
        models[j]->fullDelete();
        delete models[j];
    }

	//delete reg;
	printf("done\n");
	return 0;
}
