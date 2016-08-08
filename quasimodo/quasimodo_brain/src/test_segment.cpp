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



std::vector<Eigen::Matrix4f> getRegisteredViewPoses(const std::string& poses_file, const int& no_transforms){
	std::vector<Eigen::Matrix4f> toRet;
	ifstream in(poses_file);
	if (!in.is_open()){
		cout<<"ERROR: cannot find poses file "<<poses_file<<endl;
		return toRet;
	}
	cout<<"Loading additional view registered poses from "<<poses_file<<endl;

	for (int i=0; i<no_transforms+1; i++){
		Eigen::Matrix4f transform;
		float temp;
		for (size_t j=0; j<4; j++){
			for (size_t k=0; k<4; k++){
				in >> temp;
				transform(j,k) = temp;
			}
		}
		toRet.push_back(transform);
	}
	return toRet;
}

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
    for (size_t i=0; i < 2 && i<roomData.vIntermediateRoomClouds.size(); i++)
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
//        delete frame;
//        delete cam;

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
    printf("nr points: %i\n",sweepmodel->points.size());
//    sweepmodel->fullDelete();
//    delete sweepmodel;


    models.push_back(sweepmodel);
	return sweepmodel;
}

void runExitBehaviour(){
//	for(int j = 0; j < models.size(); j++){
//		models[j]->fullDelete();
//	}
	printf("done\n");
	viewer.reset();
}

int main(int argc, char** argv){
    reglib::RegistrationRandom *	reg	= new reglib::RegistrationRandom();

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
    bgmassreg->maskstep = 10;
    bgmassreg->nomaskstep = 10;
    bgmassreg->nomask = true;
    bgmassreg->stopval = 0.0005;
    bgmassreg->setData(models.front()->frames,models.front()->modelmasks);
    reglib::MassFusionResults bgmfr = bgmassreg->getTransforms(models.front()->relativeposes);
    delete bgmassreg;


//    reglib::ModelUpdaterBasicFuse * mu	= new reglib::ModelUpdaterBasicFuse( models.front(), reg);
//    mu->occlusion_penalty               = 15;
//    mu->massreg_timeout                 = 60*4;
//    mu->viewer							= viewer;
    for(size_t j = 0; j < models.size(); j++){
//        //models[j]->relativeposes	= bgmfr.poses;
//        models[j]->points			= mu->getSuperPoints(models[j]->relativeposes,models[j]->frames,models[j]->modelmasks,1,false);
    }







ros::init(argc, argv, "test_segment");
ros::NodeHandle n;
ros::ServiceClient segmentation_client = n.serviceClient<quasimodo_msgs::segment_model>("segment_model");








//


//	for(int i = 0; i < models.front()->relativeposes.size(); i++){
//		models.front()->frames[i]->camera->print();
//		std::cout << models.front()->relativeposes[i] << std::endl << std::endl;
//	}


	for(unsigned int i = 1; i < models.size(); i++){
		quasimodo_msgs::segment_model sm;
		sm.request.models.push_back(quasimodo_brain::getModelMSG(models[i]));
                /*
		if(i > 0){
			sm.request.backgroundmodel = quasimodo_brain::getModelMSG(models[i-1]);
		}

		if (segmentation_client.call(sm)){//Build model from frame
			//int model_id = mff.response.model_id;
			printf("segmented: %i\n",i);
		}else{ROS_ERROR("Failed to call service segment_model");}
        */
	}

    delete reg;
    //viewer.reset();
    for(size_t j = 0; j < models.size(); j++){
        models[j]->fullDelete();
        delete models[j];
    }
    return 0;

//	mff.request.mask		= *(maskBridgeImage.toImageMsg());
//	mff.request.isnewmodel	= (j == (fadded.size()-1));
//	mff.request.frame_id	= fid[j];

//	if (model_from_frame_client.call(mff)){//Build model from frame
//		int model_id = mff.response.model_id;
//		if(model_id > 0){
//			ROS_INFO("model_id%i", model_id );
//		}
//	}else{ROS_ERROR("Failed to call service index_frame");}
/*
	ros::NodeHandle pn("~");


	for(unsigned int i = 0; i < models.size(); i++){
		printf("%i -> %i\n",i,models[i]->frames.size());

		vector<Eigen::Matrix4d> cp;
		vector<reglib::RGBDFrame*> cf;
		vector<reglib::ModelMask*> mm;

		vector<Eigen::Matrix4d> cp_front;
		vector<reglib::RGBDFrame*> cf_front;
		vector<reglib::ModelMask*> mm_front;
		for(int j = 0; j <= i; j++){
			cp_front.push_back(models.front()->relativeposes.front().inverse() * models[j]->relativeposes.front());
			cf_front.push_back(models[j]->frames.front());
			mm_front.push_back(models[j]->modelmasks.front());
		}

		if(i > 0){
			reglib::MassRegistrationPPR2 * massreg = new reglib::MassRegistrationPPR2(0.05);
			massreg->timeout = 1200;
			massreg->viewer = viewer;
			massreg->visualizationLvl = 0;

			massreg->maskstep = 5;//std::max(1,int(0.4*double(models[i]->frames.size())));
			massreg->nomaskstep = 5;//std::max(3,int(0.5+0.*double(models[i]->frames.size())));//std::max(1,int(0.5+1.0*double(model->frames.size())));
			massreg->nomask = true;
			massreg->stopval = 0.0005;

			//		massreg->setData(models[i]->frames,models[i]->modelmasks);
			//		reglib::MassFusionResults mfr = massreg->getTransforms(models[i]->relativeposes);

			massreg->setData(cf_front,mm_front);


			reglib::MassFusionResults mfr_front = massreg->getTransforms(cp_front);

			for(int j = i; j >= 0; j--){
				Eigen::Matrix4d change = mfr_front.poses[j] * cp_front[j].inverse();
				for(unsigned int k = 0; k < models[j]->relativeposes.size(); k++){
					cp.push_back(change * models.front()->relativeposes.front().inverse() * models[j]->relativeposes[k]);
					cf.push_back(models[j]->frames[k]);
					mm.push_back(models[j]->modelmasks[k]);
				}
			}
		}else{
			for(int j = i; j >= 0; j--){
				Eigen::Matrix4d change = Eigen::Matrix4d::Identity();//mfr_front.poses[j] * cp_front[j].inverse();
				for(unsigned int k = 0; k < models[j]->relativeposes.size(); k++){
					cp.push_back(change * models.front()->relativeposes.front().inverse() * models[j]->relativeposes[k]);
					cf.push_back(models[j]->frames[k]);
					mm.push_back(models[j]->modelmasks[k]);
				}
			}
		}




		reglib::MassRegistrationPPR2 * massreg2 = new reglib::MassRegistrationPPR2(0.0);
		massreg2->timeout = 1200;
		massreg2->viewer = viewer;
		massreg2->visualizationLvl = 1;

		massreg2->maskstep = 10;//std::max(1,int(0.4*double(models[i]->frames.size())));
		massreg2->nomaskstep = 10;//std::max(3,int(0.5+0.*double(models[i]->frames.size())));//std::max(1,int(0.5+1.0*double(model->frames.size())));
		massreg2->nomask = true;
		massreg2->stopval = 0.0005;

		massreg2->setData(cf,mm);
		reglib::MassFusionResults mfr2 = massreg2->getTransforms(cp);
		cp = mfr2.poses;

		//massreg->setData(models[i]->frames,models[i]->modelmasks);
		//reglib::MassFusionResults mfr = massreg->getTransforms(models[i]->relativeposes);

//		massreg->setData(cf,mm);
//		reglib::MassFusionResults mfr = massreg->getTransforms(cp);
//		cp = mfr.poses;



		reglib::RegistrationRandom *	reg	= new reglib::RegistrationRandom();
		reglib::ModelUpdaterBasicFuse * mu	= new reglib::ModelUpdaterBasicFuse( models[i], reg);
		mu->occlusion_penalty               = 15;
		mu->massreg_timeout                 = 60*4;
		mu->viewer							= viewer;

		vector<cv::Mat> mats;
		mu->computeDynamicObject(cp,cf,mats);//models[i]->relativeposes, models[i]->frames, mats );

//		models[i]->print();
//		mu->show_init_lvl = 2;
//		mu->makeInitialSetup();
//		models[i]->print();

		//delete mu;
	}
*/
	ros::spin();
}
