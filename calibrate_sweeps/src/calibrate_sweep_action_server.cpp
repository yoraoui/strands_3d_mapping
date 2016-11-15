#include <calibrate_sweeps/CalibrateSweepsAction.h>
#include <actionlib/server/simple_action_server.h>
#include <iostream>
#include <string>
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>
#include <boost/filesystem.hpp>
#include <tf_conversions/tf_eigen.h>

#include <semantic_map/room_xml_parser.h>
#include <semantic_map/reg_features.h>
#include <semantic_map/room_utilities.h>
#include <semantic_map/reg_transforms.h>
#include <semantic_map/sweep_parameters.h>
#include <metaroom_xml_parser/load_utilities.h>
#include <strands_sweep_registration/RobotContainer.h>

#include "ros/ros.h"
#include "std_msgs/String.h"

typedef pcl::PointXYZRGB PointType;
typedef pcl::PointCloud<pcl::PointXYZRGB> Cloud;
typedef typename Cloud::Ptr CloudPtr;
using namespace std;

typedef actionlib::SimpleActionServer<calibrate_sweeps::CalibrateSweepsAction> Server;

std::string runCalibration(int min_num_sweeps = 1, int max_num_sweeps = 100000, std::string sweep_location = "", std::string save_location = "",
						   bool newParams = true, float fx = 534.191590, float fy = 534.016892, float cx = 315.622746, float cy = 238.568515){
	if (sweep_location=="")
	{
		passwd* pw = getpwuid(getuid());
		std::string path(pw->pw_dir);

		path+="/.semanticMap/";
		sweep_location = path;
	} else {
		sweep_location+="/";
	}
	if ( ! boost::filesystem::exists( sweep_location ) )
	{
		ROS_ERROR_STREAM("Could not find folder where to load sweeps from "+sweep_location);
		//as->setAborted(res,"Could not find folder where to load sweeps from "+sweep_location);
		return "";
	}

	//SweepParameters complete_sweep_parameters (-160, 20, 160, -30, 30, 30);
	SweepParameters complete_sweep_parameters	(-160, 20, 160, -30, 30,  30);

	SweepParameters medium_sweep_parameters		(-160, 20, 160, -30, -30, -30);
	SweepParameters short_sweep_parameters		(-160, 40, 160, -30, -30, -30);
	SweepParameters shortest_sweep_parameters	(-160, 60, 140, -30, -30, -30);

	SweepParameters medium_sweep_parameters2	(-160, 20, 160, -30, 30, -30);
	SweepParameters short_sweep_parameters2		(-160, 40, 160, -30, 30, -30);
	SweepParameters shortest_sweep_parameters2	(-160, 60, 140, -30, 30, -30);

	ROS_INFO_STREAM("complete_sweep_parameters paramters " << complete_sweep_parameters);
	ROS_INFO_STREAM("medium_sweep_parameters   paramters " << medium_sweep_parameters);
	ROS_INFO_STREAM("short_sweep_parameters    paramters " << short_sweep_parameters);
	ROS_INFO_STREAM("shortest_sweep_parameters paramters " << shortest_sweep_parameters);


	ROS_INFO_STREAM("Calibrating using sweeps with paramters "<<complete_sweep_parameters);

	ROS_INFO_STREAM("Sweeps will be read from "<<sweep_location);

	std::string save_folder;
	// default location
	passwd* pw = getpwuid(getuid());
	std::string path(pw->pw_dir);

	path+="/.ros/semanticMap/";
	save_folder = path;
	if ( ! boost::filesystem::exists( save_folder ) )
	{
		if (!boost::filesystem::create_directory(save_folder))
		{
			ROS_ERROR_STREAM("Could not create folder where to save calibration data "+save_folder);
			 //as->setAborted(res,"Could not create folder where to save calibration data "+save_folder);
			 return "";
		}
	}
	ROS_INFO_STREAM("Calibration data will be saved at: "<<save_folder);

	// Load sweeps
	vector<string> matchingObservations = semantic_map_load_utilties::getSweepXmls<PointType>(sweep_location);

	if (matchingObservations.size() < min_num_sweeps)
	{
		ROS_ERROR_STREAM("Not enough sweeps to perform calibration "<<matchingObservations.size());
		 //as->setAborted(res,"Not enough sweeps to perform calibration "+matchingObservations.size());
		 return "";
	}

	std::string saveLocation = save_location;
	if (saveLocation == "")
	{
		saveLocation = sweep_location; // save in the same folder
	} else {
		saveLocation+="/";
		if ( ! boost::filesystem::exists( saveLocation ) )
		{
			if (!boost::filesystem::create_directory(saveLocation))
			{
				ROS_ERROR_STREAM("Could not create folder where to save calibration data "+saveLocation);
				 //as->setAborted(res,"Could not create folder where to save calibration data "+saveLocation);
				 return "";
			}
		}
	}
	ROS_INFO_STREAM("The registered sweeps will be saved at: "<<saveLocation);

	sort(matchingObservations.begin(), matchingObservations.end());
	reverse(matchingObservations.begin(), matchingObservations.end());

	// Initialize calibration class
	unsigned int gx = 17;
	unsigned int todox = 17;
	unsigned int gy = 3;
	unsigned int todoy = 3;
	RobotContainer * rc = new RobotContainer(gx,todox,gy,todoy);


//    	534.191590 0.000000 315.622746
//		0.000000 534.016892 238.568515
//0.000000 0.000000 1.000000
//printf("%f %f %f %f\n",534.191590, 534.016892,315.622746, 238.568515);
//	rc->initializeCamera(534.191590, 534.016892,315.622746, 238.568515, 640, 480);



	if(newParams){
		//printf("%f %f %f %f\n",fx,fy,cx,cy);
		rc->initializeCamera(fx,fy,cx,cy, 640, 480);
	}else{
		// initialize camera parameters from the sweep
		if (matchingObservations.size()){
			SemanticRoom<PointType> aRoom = SemanticRoomXMLParser<PointType>::loadRoomFromXML(matchingObservations.front(),true);
			if (aRoom.getIntermediateCloudCameraParameters().size()){
				image_geometry::PinholeCameraModel aCameraModel = aRoom.getIntermediateCloudCameraParameters()[0];
				rc->initializeCamera(aCameraModel.fx(), aCameraModel.fy(), aCameraModel.cx(), aCameraModel.cy(), aCameraModel.fullResolution().width, aCameraModel.fullResolution().height);
			} else {
				// no camera parameters saved with the sweep -> initialize optimizer with default parameters
				rc->initializeCamera(540.0, 540.0,319.5, 219.5, 640, 480);
			}
		}else{
			exit(0);
		}
	}

	CloudPtr dummy_cloud (new Cloud());
	dummy_cloud->width = 640;
	dummy_cloud->height = 480;
	dummy_cloud->points.resize(dummy_cloud->width*dummy_cloud->height);

//exit(0);

//	SweepParameters complete_sweep_parameters	(-160, 20, 160, -30, 30,  30);
//	SweepParameters medium_sweep_parameters		(-160, 20, 160, -30, 30, -30);
//	SweepParameters short_sweep_parameters		(-160, 40, 160, -30, 30, -30);
//	SweepParameters shortest_sweep_parameters	(-160, 60, 140, -30, 30, -30);

	for (size_t i=0; i<max_num_sweeps && i<matchingObservations.size(); i++)
	{
		printf("sweep: %i\n",i);
		// check if sweep parameters correspond

		unsigned found = matchingObservations[i].find_last_of("/");
		std::string base_path = matchingObservations[i].substr(0,found+1);
		RegistrationFeatures reg(false);
		//reg.saveOrbFeatures<PointType>(aRoom,base_path);

		std::vector<semantic_map_registration_features::RegistrationFeatures> features = semantic_map_registration_features::loadRegistrationFeaturesFromSingleSweep(matchingObservations[i], false);
		if (features.size() == 0){
			SemanticRoom<PointType> aRoom = SemanticRoomXMLParser<PointType>::loadRoomFromXML(matchingObservations[i],true);
			reg.saveOrbFeatures<PointType>(aRoom,base_path);
		}
		rc->addToTrainingORBFeatures(matchingObservations[i]);

/*
		if		 (aRoom.m_SweepParameters == complete_sweep_parameters){
			printf("complete_sweep_parameters\n");

			if (features.size() == 0){reg.saveOrbFeatures<PointType>(aRoom,base_path);}
			rc->addToTrainingORBFeatures(matchingObservations[i]);

		}else if (aRoom.m_SweepParameters == medium_sweep_parameters   || aRoom.m_SweepParameters == medium_sweep_parameters2 ){
			printf("medium_sweep_parameters\n");

			for(unsigned int j = 0; j < before.size(); j++){
				after.push_back(before[j]);
			}

			while(after.size() < 51){after.push_back(dummy_cloud);}
			aRoom.setIntermediateClouds(after);

			if (features.size() == 0){reg.saveOrbFeatures<PointType>(aRoom,base_path);}
			rc->addToTrainingORBFeatures(matchingObservations[i]);

		}else if (aRoom.m_SweepParameters == short_sweep_parameters    || aRoom.m_SweepParameters == short_sweep_parameters2 ){
			printf("short_sweep_parameters\n");
		}else if (aRoom.m_SweepParameters == shortest_sweep_parameters || aRoom.m_SweepParameters == shortest_sweep_parameters2 ){
			printf("shortest_sweep_parameters\n");
		}else{
			printf("sweep has no known type\n");
			std::cout <<"sweep parameters not correct: " << aRoom.m_SweepParameters << std::endl << std::endl << std::endl << std::endl;
		}
*/

//		if (aRoom.m_SweepParameters != complete_sweep_parameters){
//			ROS_INFO_STREAM("Skipping "<<matchingObservations[i]<<" sweep parameters not correct: "<<aRoom.m_SweepParameters<<" Required parameters "<<complete_sweep_parameters);
//			continue; // not a match
//		}

//		// check if the orb features have already been computed

		//
	}
	// perform calibration
	std::vector<Eigen::Matrix4f> cameraPoses = rc->train();

	std::vector<tf::StampedTransform> registeredPoses;

	for (auto eigenPose : cameraPoses)
	{
		tf::StampedTransform tfStamped;
		tfStamped.frame_id_ = "temp";
		tfStamped.child_frame_id_ = "temp";
		tf::Transform tfTr;
		const Eigen::Affine3d eigenTr(eigenPose.cast<double>());
		tf::transformEigenToTF(eigenTr, tfTr);
		tfStamped.setOrigin(tfTr.getOrigin());
		tfStamped.setBasis(tfTr.getBasis());
		registeredPoses.push_back(tfStamped);
	}
	std::string registeredPosesFile = semantic_map_registration_transforms::saveRegistrationTransforms(registeredPoses);
	registeredPoses.clear();
	registeredPoses = semantic_map_registration_transforms::loadRegistrationTransforms(registeredPosesFile);
	ROS_INFO_STREAM("Calibration poses saved at: "<<registeredPosesFile);

	std::vector<tf::StampedTransform> complete_registeredPoses = registeredPoses;

	std::vector<tf::StampedTransform> medium_registeredPoses;
	for(unsigned int i = 0; i < 17; i+=1){medium_registeredPoses.push_back(registeredPoses[i]);}

	std::vector<tf::StampedTransform> short_registeredPoses;
	for(unsigned int i = 0; i < 17; i+=2){short_registeredPoses.push_back(registeredPoses[i]);}

	std::vector<tf::StampedTransform> shortest_registeredPoses;
	for(unsigned int i = 0; i < 17; i+=3){shortest_registeredPoses.push_back(registeredPoses[i]);}

	std::string sweepParametersFile = semantic_map_registration_transforms::saveSweepParameters(complete_sweep_parameters);
	ROS_INFO_STREAM("Calibration sweep parameters saved at: "<<sweepParametersFile);

	double*** rawPoses = rc->poses;
	unsigned int x,y;
	std::string rawPosesFile = semantic_map_registration_transforms::saveRegistrationTransforms(rawPoses, rc->todox,rc->todoy);
	ROS_INFO_STREAM("Raw calibration data saved at: "<<rawPosesFile);


	// correct used sweeps with the new transforms and camera parameters
	// create corrected cam params
	sensor_msgs::CameraInfo camInfo;
	camInfo.P = {rc->camera->fx, 0.0, rc->camera->cx, 0.0, 0.0, rc->camera->fy, rc->camera->cy, 0.0,0.0, 0.0, 1.0,0.0};
	camInfo.D = {0,0,0,0,0};
	image_geometry::PinholeCameraModel aCameraModel;
	aCameraModel.fromCameraInfo(camInfo);

	std::string camParamsFile = semantic_map_registration_transforms::saveCameraParameters(aCameraModel);
	ROS_INFO_STREAM("Camera parameters saved at: "<<camParamsFile);



	// update sweeps with new poses and new camera parameters

	SemanticRoomXMLParser<PointType> reg_parser(saveLocation);

	for (auto usedObs : matchingObservations){
		SemanticRoom<PointType> aRoom = SemanticRoomXMLParser<PointType>::loadRoomFromXML(usedObs,true);
		auto origTransforms = aRoom.getIntermediateCloudTransforms();
		aRoom.clearIntermediateCloudRegisteredTransforms();
		aRoom.clearIntermediateCloudCameraParametersCorrected();

		std::vector<tf::StampedTransform> corresponding_registeredPoses;
		corresponding_registeredPoses = semantic_map_registration_transforms::loadCorrespondingRegistrationTransforms(aRoom.m_SweepParameters);
		ROS_INFO_STREAM("Corresponing registered poses "<<corresponding_registeredPoses.size()<<" original transforms "<<origTransforms.size());
		if (corresponding_registeredPoses.size() != origTransforms.size()){
			ROS_ERROR_STREAM("Cannot use registered poses to correct sweep: "<<usedObs);
			continue;
		}

		for (size_t i=0; i<origTransforms.size(); i++)
		{
			tf::StampedTransform transform = origTransforms[i];
//            transform.setOrigin(registeredPoses[i].getOrigin());
//            transform.setBasis(registeredPoses[i].getBasis());
			transform.setOrigin(corresponding_registeredPoses[i].getOrigin());
			transform.setBasis(corresponding_registeredPoses[i].getBasis());
			aRoom.addIntermediateCloudCameraParametersCorrected(aCameraModel);
			aRoom.addIntermediateRoomCloudRegisteredTransform(transform);
		}
		semantic_map_room_utilities::reprojectIntermediateCloudsUsingCorrectedParams<PointType>(aRoom);
		semantic_map_room_utilities::rebuildRegisteredCloud<PointType>(aRoom);
		// transform to global frame of reference
		tf::StampedTransform origin = origTransforms[0];
//		CloudPtr completeCloud = aRoom.getCompleteRoomCloud();
//		pcl_ros::transformPointCloud(*completeCloud, *completeCloud,origin);
//		aRoom.setCompleteRoomCloud(completeCloud);
		string room_path = reg_parser.saveRoomAsXML(aRoom,"room.xml",true);
		ROS_INFO_STREAM("..done");
		// recompute ORB features
//		unsigned found = room_path.find_last_of("/");
//		std::string base_path = room_path.substr(0,found+1);
//		RegistrationFeatures reg(false);
//		reg.saveOrbFeatures<PointType>(aRoom,base_path);
	}
	delete rc;
	return saveLocation;
}

void execute(const calibrate_sweeps::CalibrateSweepsGoalConstPtr& goal, Server* as)
{
    ROS_INFO_STREAM("Received calibrate message. Min/max sweeps: "<<goal->min_num_sweeps<<" "<<goal->max_num_sweeps);
    calibrate_sweeps::CalibrateSweepsResult res;
	res.calibration_file = runCalibration(goal->min_num_sweeps, goal->max_num_sweeps, goal->sweep_location, goal->save_location);//registeredPosesFile;

	if (res.calibration_file.compare("") != 0){
		as->setAborted(res,"");
	}else{
		as->setSucceeded(res,"Done");
	}
/*
	ROS_INFO_STREAM("Received calibrate message. Min/max sweeps: "<<goal->min_num_sweeps<<" "<<goal->max_num_sweeps);
	calibrate_sweeps::CalibrateSweepsResult res;
    SweepParameters complete_sweep_parameters (-160, 20, 160, -30, 30, 30);
    ROS_INFO_STREAM("Calibrating using sweeps with paramters "<<complete_sweep_parameters);

    std::string sweep_location;
    if (goal->sweep_location=="")
    {
        passwd* pw = getpwuid(getuid());
        std::string path(pw->pw_dir);

        path+="/.semanticMap/";
        sweep_location = path;
    } else {
        sweep_location = goal->sweep_location;
        sweep_location+="/";
    }
    if ( ! boost::filesystem::exists( sweep_location ) )
    {
        ROS_ERROR_STREAM("Could not find folder where to load sweeps from "+sweep_location);
        as->setAborted(res,"Could not find folder where to load sweeps from "+sweep_location);
        return;
    }

    ROS_INFO_STREAM("Sweeps will be read from "<<sweep_location);

    std::string save_folder;
    // default location
    passwd* pw = getpwuid(getuid());
    std::string path(pw->pw_dir);

    path+="/.ros/semanticMap/";
    save_folder = path;
    if ( ! boost::filesystem::exists( save_folder ) )
    {
        if (!boost::filesystem::create_directory(save_folder))
        {
            ROS_ERROR_STREAM("Could not create folder where to save calibration data "+save_folder);
             as->setAborted(res,"Could not create folder where to save calibration data "+save_folder);
             return;
        }
    }
    ROS_INFO_STREAM("Calibration data will be saved at: "<<save_folder);

    // Load sweeps
    vector<string> matchingObservations = semantic_map_load_utilties::getSweepXmls<PointType>(sweep_location);

    if (matchingObservations.size() < goal->min_num_sweeps)
    {
        ROS_ERROR_STREAM("Not enough sweeps to perform calibration "<<matchingObservations.size());
         as->setAborted(res,"Not enough sweeps to perform calibration "+matchingObservations.size());
         return;
    }

    std::string saveLocation = goal->save_location;
    if (saveLocation == "")
    {
        saveLocation = sweep_location; // save in the same folder
    } else {
        saveLocation+="/";
        if ( ! boost::filesystem::exists( saveLocation ) )
        {
            if (!boost::filesystem::create_directory(saveLocation))
            {
                ROS_ERROR_STREAM("Could not create folder where to save calibration data "+saveLocation);
				 as->setAborted(res,"Could not create folder where to /media/johane/SSDstorage/final_tsc_semantic_maps/semantic_maps//20160808/patrol_run_110/room_0/save calibration data "+saveLocation);
                 return;
            }
        }
    }
    ROS_INFO_STREAM("The registered sweeps will be saved at: "<<saveLocation);

    sort(matchingObservations.begin(), matchingObservations.end());
    reverse(matchingObservations.begin(), matchingObservations.end());

    // Initialize calibration class
    unsigned int gx = 17;
    unsigned int todox = 17;
    unsigned int gy = 3;
    unsigned int todoy = 3;
    RobotContainer * rc = new RobotContainer(gx,todox,gy,todoy);
    
    
//    	534.191590 0.000000 315.622746
//		0.000000 534.016892 238.568515
//0.000000 0.000000 1.000000
printf("%f %f %f %f\n",534.191590, 534.016892,315.622746, 238.568515);
	rc->initializeCamera(534.191590, 534.016892,315.622746, 238.568515, 640, 480);
//exit(0);
    // initialize camera parameters from the sweep

//    if (matchingObservations.size()){
//        SemanticRoom<PointType> aRoom = SemanticRoomXMLParser<PointType>::loadRoomFromXML(matchingObservations[0],true);
//        if (aRoom.getIntermediateCloudCameraParameters().size()){
//            image_geometry::PinholeCameraModel aCameraModel = aRoom.getIntermediateCloudCameraParameters()[0];
//            rc->initializeCamera(aCameraModel.fx(), aCameraModel.fy(), aCameraModel.cx(), aCameraModel.cy(), aCameraModel.fullResolution().width, aCameraModel.fullResolution().height);
//        } else {
//            // no camera parameters saved with the sweep -> initialize optimizer with default parameters
//            rc->initializeCamera(540.0, 540.0,319.5, 219.5, 640, 480);
//        }
//    }


    for (size_t i=0; i<goal->max_num_sweeps && i<matchingObservations.size(); i++)
    {
        // check if sweep parameters correspond
        SemanticRoom<PointType> aRoom = SemanticRoomXMLParser<PointType>::loadRoomFromXML(matchingObservations[i],true);
        if (aRoom.m_SweepParameters != complete_sweep_parameters){
            ROS_INFO_STREAM("Skipping "<<matchingObservations[i]<<" sweep parameters not correct: "<<aRoom.m_SweepParameters<<" Required parameters "<<complete_sweep_parameters);
            continue; // not a match
        }

        // check if the orb features have already been computed
        std::vector<semantic_map_registration_features::RegistrationFeatures> features = semantic_map_registration_features::loadRegistrationFeaturesFromSingleSweep(matchingObservations[i], false);
        if (features.size() == 0)
        {
            // recompute orb

            unsigned found = matchingObservations[i].find_last_of("/");
            std::string base_path = matchingObservations[i].substr(0,found+1);
            RegistrationFeatures reg(false);
            reg.saveOrbFeatures<PointType>(aRoom,base_path);
        }
        rc->addToTrainingORBFeatures(matchingObservations[i]);
    }

    // perform calibration
    std::vector<Eigen::Matrix4f> cameraPoses = rc->train();
    std::vector<tf::StampedTransform> registeredPoses;

    for (auto eigenPose : cameraPoses)
    {
        tf::StampedTransform tfStamped;
        tfStamped.frame_id_ = "temp";
        tfStamped.child_frame_id_ = "temp";
        tf::Transform tfTr;
        const Eigen::Affine3d eigenTr(eigenPose.cast<double>());
        tf::transformEigenToTF(eigenTr, tfTr);
        tfStamped.setOrigin(tfTr.getOrigin());
        tfStamped.setBasis(tfTr.getBasis());
        registeredPoses.push_back(tfStamped);
    }
    std::string registeredPosesFile = semantic_map_registration_transforms::saveRegistrationTransforms(registeredPoses);
    registeredPoses.clear();
    registeredPoses = semantic_map_registration_transforms::loadRegistrationTransforms(registeredPosesFile);
    ROS_INFO_STREAM("Calibration poses saved at: "<<registeredPosesFile);

    std::string sweepParametersFile = semantic_map_registration_transforms::saveSweepParameters(complete_sweep_parameters);
    ROS_INFO_STREAM("Calibration sweep parameters saved at: "<<sweepParametersFile);

    double*** rawPoses = rc->poses;
    unsigned int x,y;
    std::string rawPosesFile = semantic_map_registration_transforms::saveRegistrationTransforms(rawPoses, rc->todox,rc->todoy);
    ROS_INFO_STREAM("Raw calibration data saved at: "<<rawPosesFile);


    // correct used sweeps with the new transforms and camera parameters
    // create corrected cam params
    sensor_msgs::CameraInfo camInfo;
    camInfo.P = {rc->camera->fx, 0.0, rc->camera->cx, 0.0, 0.0, rc->camera->fy, rc->camera->cy, 0.0,0.0, 0.0, 1.0,0.0};
    camInfo.D = {0,0,0,0,0};
    image_geometry::PinholeCameraModel aCameraModel;
    aCameraModel.fromCameraInfo(camInfo);

    std::string camParamsFile = semantic_map_registration_transforms::saveCameraParameters(aCameraModel);
    ROS_INFO_STREAM("Camera parameters saved at: "<<camParamsFile);

    // update sweeps with new poses and new camera parameters

    SemanticRoomXMLParser<PointType> reg_parser(saveLocation);

    for (auto usedObs : matchingObservations)
    {
        SemanticRoom<PointType> aRoom = SemanticRoomXMLParser<PointType>::loadRoomFromXML(usedObs,true);
        auto origTransforms = aRoom.getIntermediateCloudTransforms();
        aRoom.clearIntermediateCloudRegisteredTransforms();
        aRoom.clearIntermediateCloudCameraParametersCorrected();

        std::vector<tf::StampedTransform> corresponding_registeredPoses;
        corresponding_registeredPoses = semantic_map_registration_transforms::loadCorrespondingRegistrationTransforms(aRoom.m_SweepParameters);
        ROS_INFO_STREAM("Corresponing registered poses "<<corresponding_registeredPoses.size()<<" original transforms "<<origTransforms.size());
        if (corresponding_registeredPoses.size() != origTransforms.size()){
            ROS_ERROR_STREAM("Cannot use registered poses to correct sweep: "<<usedObs);
            continue;
        }

        for (size_t i=0; i<origTransforms.size(); i++)
        {
            tf::StampedTransform transform = origTransforms[i];
//            transform.setOrigin(registeredPoses[i].getOrigin());
//            transform.setBasis(registeredPoses[i].getBasis());
            transform.setOrigin(corresponding_registeredPoses[i].getOrigin());
            transform.setBasis(corresponding_registeredPoses[i].getBasis());
            aRoom.addIntermediateCloudCameraParametersCorrected(aCameraModel);
            aRoom.addIntermediateRoomCloudRegisteredTransform(transform);
        }
        semantic_map_room_utilities::reprojectIntermediateCloudsUsingCorrectedParams<PointType>(aRoom);
        semantic_map_room_utilities::rebuildRegisteredCloud<PointType>(aRoom);
        // transform to global frame of reference
        tf::StampedTransform origin = origTransforms[0];
        CloudPtr completeCloud = aRoom.getCompleteRoomCloud();
        pcl_ros::transformPointCloud(*completeCloud, *completeCloud,origin);
        aRoom.setCompleteRoomCloud(completeCloud);
        string room_path = reg_parser.saveRoomAsXML(aRoom);
        ROS_INFO_STREAM("..done");
        // recompute ORB features
        unsigned found = room_path.find_last_of("/");
        std::string base_path = room_path.substr(0,found+1);
        RegistrationFeatures reg(false);
        reg.saveOrbFeatures<PointType>(aRoom,base_path);
    }

    delete rc;

	res.calibration_file = registeredPosesFile;

    as->setSucceeded(res,"Done");
	*/
}

void runCallback(const std_msgs::String::ConstPtr& msg){
	runCalibration();
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "calibrate_sweeps_action_server");
  ros::NodeHandle n;
  Server server(n, "calibrate_sweeps", boost::bind(&execute, _1, &server), false);
  ROS_INFO_STREAM("Calibrate sweep action server initialized");
  server.start();
  ros::Subscriber sub = n.subscribe("calibrate_sweeps_action_server/run", 1000, runCallback);

  bool runCalib = false;
  std::vector<std::string> recalibPaths;
  //std::string recalibPath = std::string(getenv ("HOME"))+"/.semanticMap/";
  bool loadParams = false;
  float fx = 535;
  float fy = 535;
  float cx = 0.5*(640.0-1);
  float cy = 0.5*(480.0-1);

  int max_sweeps = 100;

  int inputstate = -1;
  for(unsigned int i = 1; i < argc; i++){
	  if(std::string(argv[i]).compare("-run") == 0){
		  runCalib = true;
		  inputstate = 0;
	  }else if(std::string(argv[i]).compare("-fx") == 0){
		  inputstate = 1;
	  }else if(std::string(argv[i]).compare("-fy") == 0){
		  inputstate = 2;
	  }else if(std::string(argv[i]).compare("-cx") == 0){
		  inputstate = 3;
	  }else if(std::string(argv[i]).compare("-cy") == 0){
		  inputstate = 4;
	  }else if(std::string(argv[i]).compare("-max_sweeps") == 0){
		  inputstate = 5;
	  }else if(std::string(argv[i]).compare("-loadParams") == 0){
		  loadParams = true;
	  }else if(inputstate == 0){
		  recalibPaths.push_back(std::string(argv[i]));
	  }else if(inputstate == 1){
		  loadParams = true;
		  fx = atof(argv[i]);
	  }else if(inputstate == 2){
		  loadParams = true;
		  fy = atof(argv[i]);
	  }else if(inputstate == 3){
		  loadParams = true;
		  cx = atof(argv[i]);
	  }else if(inputstate == 4){
		  loadParams = true;
		  cy = atof(argv[i]);
	  }else if(inputstate == 5){
		  max_sweeps = atoi(argv[i]);
	  }
  }

  if(runCalib){
	if(recalibPaths.size() == 0){
		recalibPaths.push_back(std::string(getenv ("HOME"))+"/.semanticMap/");
	}
	for(unsigned int i = 0; i < recalibPaths.size(); i++){
		if(loadParams){
			runCalibration(1,max_sweeps,recalibPaths[i], "",loadParams,fx,fy,cx,cy);
		}else{
			runCalibration(1,max_sweeps,recalibPaths[i], "");
		}
		//bool newParams = true, float fx = 534.191590, float fy = 534.016892, float cx = 315.622746, float cy = 238.568515
	}
	exit(0);
  }

  ros::spin();
  return 0;
}
