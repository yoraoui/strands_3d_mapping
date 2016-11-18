#include "Util.h"

namespace quasimodo_brain {

void cleanPath(std::string & path){
	std::size_t found = path.find("//");
	while (found!=std::string::npos){
		path.replace(found,2,"/");
		found = path.find("//");
	}
}

bool xmlSortFunction (std::string & a, std::string & b) {
	return true;
}

void sortXMLs(std::vector<std::string> & sweeps){

}

reglib::Model * loadFromRaresFormat(std::string path){
	reglib::Model * model = new reglib::Model();
	reglib::Model * sweep = load_metaroom_model(path);
	if(sweep->frames.size() == 0){
		printf("no frames in sweep, returning\n");
		return sweep;
	}

	model->submodels.push_back(sweep);
	model->submodels_relativeposes.push_back(Eigen::Matrix4d::Identity());


	int slash_pos = path.find_last_of("/");
	std::string sweep_folder = path.substr(0, slash_pos) + "/";
	printf("folder: %s\n",sweep_folder.c_str());


	std::vector<semantic_map_load_utilties::DynamicObjectData<pcl::PointXYZRGB>> objects = semantic_map_load_utilties::loadAllDynamicObjectsFromSingleSweep<pcl::PointXYZRGB>(sweep_folder+"room.xml");

	for (auto object : objects){
		if (!object.objectScanIndices.size()){continue;}

		printf("%i %i %i\n",int(object.vAdditionalViews.size()),int(object.vAdditionalViewsTransformsRegistered.size()),int(object.vAdditionalViewsTransforms.size()));
		for (unsigned int i = 0; i < object.vAdditionalViews.size(); i ++){
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = object.vAdditionalViews[i];

			cv::Mat rgb;
			rgb.create(cloud->height,cloud->width,CV_8UC3);
			unsigned char * rgbdata = (unsigned char *)rgb.data;

			cv::Mat depth;
			depth.create(cloud->height,cloud->width,CV_16UC1);
			unsigned short * depthdata = (unsigned short *)depth.data;

			unsigned int nr_data = cloud->height * cloud->width;
			for(unsigned int j = 0; j < nr_data; j++){
				pcl::PointXYZRGB p = cloud->points[j];
				rgbdata[3*j+0]	= p.b;
				rgbdata[3*j+1]	= p.g;
				rgbdata[3*j+2]	= p.r;
				depthdata[j]	= short(5000.0 * p.z);
			}

			cv::Mat fullmask;
			fullmask.create(cloud->height,cloud->width,CV_8UC1);
			unsigned char * maskdata = (unsigned char *)fullmask.data;
			for(int j = 0; j < nr_data; j++){maskdata[j] = 255;}

			reglib::Camera * cam		= sweep->frames.front()->camera->clone();

			reglib::RGBDFrame * frame = new reglib::RGBDFrame(cam,rgb,depth,0, getMat(object.vAdditionalViewsTransforms[i]));
			model->frames.push_back(frame);
			if(model->relativeposes.size() == 0){
				model->relativeposes.push_back(Eigen::Matrix4d::Identity());//current_room_frames.front()->pose.inverse() * frame->pose);
			}else{
				model->relativeposes.push_back(model->frames.front()->pose.inverse() * frame->pose);
			}
			model->modelmasks.push_back(new reglib::ModelMask(fullmask));
		}
	}


	//	if(model->relativeposes.size() == 0){
	//		model->submodels_relativeposes.push_back(Eigen::Matrix4d::Identity());
	//	}else{
	//		model->submodels_relativeposes.push_back(model->relativeposes.front().inverse() * sweep->frames.front()->pose);
	//	}


	return model;
}

std::vector<Eigen::Matrix4f> getRegisteredViewPoses(std::string poses_file, int no_transforms){
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

double getTime(){
	struct timeval start1;
	gettimeofday(&start1, NULL);
	return double(start1.tv_sec+(start1.tv_usec/1000000.0));
}

reglib::Model * getModelFromMSG(quasimodo_msgs::model & msg, bool compute_edges){
	reglib::Model * model = new reglib::Model();

	for(unsigned int i = 0; i < msg.local_poses.size(); i++){
		sensor_msgs::CameraInfo		camera			= msg.frames[i].camera;
		ros::Time					capture_time	= msg.frames[i].capture_time;
		geometry_msgs::Pose			pose			= msg.frames[i].pose;

		cv_bridge::CvImagePtr			rgb_ptr;
		try{							rgb_ptr = cv_bridge::toCvCopy(msg.frames[i].rgb, sensor_msgs::image_encodings::BGR8);}
		catch (cv_bridge::Exception& e){ROS_ERROR("cv_bridge exception: %s", e.what());}
		cv::Mat rgb = rgb_ptr->image;

		cv_bridge::CvImagePtr			depth_ptr;
		try{							depth_ptr = cv_bridge::toCvCopy(msg.frames[i].depth, sensor_msgs::image_encodings::MONO16);}
		catch (cv_bridge::Exception& e){ROS_ERROR("cv_bridge exception: %s", e.what());}
		cv::Mat depth = depth_ptr->image;

		Eigen::Affine3d epose;
		tf::poseMsgToEigen(pose, epose);

		reglib::Camera * cam		= new reglib::Camera();
		if(camera.K[0] > 0){
			cam->fx = camera.K[0];
			cam->fy = camera.K[4];
			cam->cx = camera.K[2];
			cam->cy = camera.K[5];
		}

		reglib::RGBDFrame * frame = new reglib::RGBDFrame(cam,rgb, depth, double(capture_time.sec)+double(capture_time.nsec)/1000000000.0, epose.matrix(),true,"",compute_edges);
		model->frames.push_back(frame);

		geometry_msgs::Pose	pose1 = msg.local_poses[i];
		Eigen::Affine3d epose1;
		tf::poseMsgToEigen(pose1, epose1);
		model->relativeposes.push_back(epose1.matrix());

		cv_bridge::CvImagePtr			mask_ptr;
		try{							mask_ptr = cv_bridge::toCvCopy(msg.masks[i], sensor_msgs::image_encodings::MONO8);}
		catch (cv_bridge::Exception& e){ROS_ERROR("cv_bridge exception: %s", e.what());}
		cv::Mat mask = mask_ptr->image;

		model->modelmasks.push_back(new reglib::ModelMask(mask));
	}
	model->recomputeModelPoints();
	return model;
}

void addToModelMSG(quasimodo_msgs::model & msg, reglib::Model * model, Eigen::Affine3d rp, bool addClouds){
	int startsize = msg.local_poses.size();
	msg.local_poses.resize(startsize+model->relativeposes.size());
	msg.frames.resize(startsize+model->frames.size());
	msg.masks.resize(startsize+model->modelmasks.size());
	for(unsigned int i = 0; i < model->relativeposes.size(); i++){
		geometry_msgs::Pose		pose1;
		tf::poseEigenToMsg (Eigen::Affine3d(model->relativeposes[i])*rp, pose1);
		geometry_msgs::Pose		pose2;
		tf::poseEigenToMsg (Eigen::Affine3d(model->frames[i]->pose)*rp, pose2);
		cv_bridge::CvImage rgbBridgeImage;
		rgbBridgeImage.image = model->frames[i]->rgb;
		rgbBridgeImage.encoding = "bgr8";
		cv_bridge::CvImage depthBridgeImage;
		depthBridgeImage.image = model->frames[i]->depth;
		depthBridgeImage.encoding = "mono16";
		cv_bridge::CvImage maskBridgeImage;
		maskBridgeImage.image			= model->modelmasks[i]->getMask();
		maskBridgeImage.encoding		= "mono8";
		msg.local_poses[startsize+i]			= pose1;
		msg.frames[startsize+i].capture_time	= ros::Time();
		msg.frames[startsize+i].pose			= pose2;
		msg.frames[startsize+i].frame_id		= model->frames[i]->id;
		msg.frames[startsize+i].rgb				= *(rgbBridgeImage.toImageMsg());
		msg.frames[startsize+i].depth			= *(depthBridgeImage.toImageMsg());
		msg.masks[startsize+i]					= *(maskBridgeImage.toImageMsg());

		msg.frames[startsize+i].camera.K[0] = model->frames[i]->camera->fx;
		msg.frames[startsize+i].camera.K[4] = model->frames[i]->camera->fy;
		msg.frames[startsize+i].camera.K[2] = model->frames[i]->camera->cx;
		msg.frames[startsize+i].camera.K[5] = model->frames[i]->camera->cy;

		sensor_msgs::PointCloud2 output;
		if(addClouds){pcl::toROSMsg(*(model->frames[i]->getPCLcloud()), output);}
		msg.clouds.push_back(output);
	}
	for(unsigned int i = 0; i < model->submodels_relativeposes.size(); i++){
		addToModelMSG(msg,model->submodels[i],Eigen::Affine3d(model->submodels_relativeposes[i])*rp,addClouds);
	}
}

quasimodo_msgs::model getModelMSG(reglib::Model * model, bool addClouds){
	quasimodo_msgs::model msg;
	msg.model_id = model->id;
	addToModelMSG(msg,model,Eigen::Affine3d::Identity(),addClouds);
	return msg;
}

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

Eigen::Matrix4d getMat(tf::StampedTransform tf){

	//printf("getMat\n");
	//Transform
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

	//std::cout << epose.matrix() << std::endl << std::endl;
	return epose.matrix();
}

reglib::Model * load_metaroom_model(std::string sweep_xml, std::string savePath){
	int slash_pos = sweep_xml.find_last_of("/");
	std::string sweep_folder = sweep_xml.substr(0, slash_pos) + "/";
//	printf("load_metaroom_model(%s)\n",sweep_folder.c_str());

	SimpleXMLParser<pcl::PointXYZRGB> parser;
	std::vector<std::string> dummy;
	dummy.push_back("RoomIntermediateCloud");
	dummy.push_back("IntermediatePosition");
	SimpleXMLParser<pcl::PointXYZRGB>::RoomData roomData  = parser.loadRoomFromXML(sweep_folder+"/room.xml",dummy);
	printf("loaded room XML: %s\n",(sweep_folder+"/room.xml").c_str());

	if(roomData.vIntermediateRoomClouds.size() == 0 ){
		printf("WARNING:: no clouds in room\n");
		return new reglib::Model();
	}

	reglib::Model * sweepmodel = 0;
	printf("%i\n",roomData.vIntermediateRoomCloudTransforms.size());
	Eigen::Matrix4d m2 = getMat(roomData.vIntermediateRoomCloudTransforms[0]);
	std::vector<reglib::RGBDFrame * > current_room_frames;


//	printf("roomData.vIntermediateRoomClouds.size() = %i\n",roomData.vIntermediateRoomClouds.size());
//	printf("roomData.vIntermediateRoomCloudCamParamsCorrected.size() = %i\n",roomData.vIntermediateRoomCloudCamParamsCorrected.size());
//	printf("roomData.vIntermediateRoomCloudTransformsRegistered.size() = %i\n",roomData.vIntermediateRoomCloudTransformsRegistered.size());

	for (size_t i=0; i<roomData.vIntermediateRoomClouds.size(); i++){
		//printf("loading intermedite %i\n",i);
		cv::Mat fullmask;
		fullmask.create(480,640,CV_8UC1);
		unsigned char * maskdata = (unsigned char *)fullmask.data;
		for(int j = 0; j < 480*640; j++){maskdata[j] = 255;}

		image_geometry::PinholeCameraModel aCameraModel = roomData.vIntermediateRoomCloudCamParamsCorrected[i];
		//image_geometry::PinholeCameraModel aCameraModel = roomData.vIntermediateRoomCloudCamParams[i];
		reglib::Camera * cam		= new reglib::Camera();
		cam->fx = aCameraModel.fx();
		cam->fy = aCameraModel.fy();
		cam->cx = aCameraModel.cx();
		cam->cy = aCameraModel.cy();
		//cam->print();

		Eigen::Matrix4d m = m2*getMat(roomData.vIntermediateRoomCloudTransformsRegistered[i]);
		//Eigen::Matrix4d m =      getMat(roomData.vIntermediateRoomCloudTransforms[i]);

		reglib::RGBDFrame * frame = new reglib::RGBDFrame(cam,roomData.vIntermediateRGBImages[i],5.0*roomData.vIntermediateDepthImages[i],0, m,true,savePath);

		current_room_frames.push_back(frame);
		if(sweepmodel == 0){
			sweepmodel = new reglib::Model(frame,fullmask);
		}else{
			sweepmodel->frames.push_back(frame);
			sweepmodel->relativeposes.push_back(current_room_frames.front()->pose.inverse() * frame->pose);
			sweepmodel->modelmasks.push_back(new reglib::ModelMask(fullmask));
		}
	}


	sweepmodel->recomputeModelPoints();

	return sweepmodel;
}

void segment(std::vector< reglib::Model * > bgs, std::vector< reglib::Model * > models, std::vector< std::vector< cv::Mat > > & internal, std::vector< std::vector< cv::Mat > > & external, std::vector< std::vector< cv::Mat > > & dynamic, int debugg, std::string savePath){
	double startTime = getTime();
	printf("running segment method\n");
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
	if(debugg){
		viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer ("viewer"));
		viewer->addCoordinateSystem(0.01);
		viewer->setBackgroundColor(1.0,1.0,1.0);
	}

	reglib::MassRegistrationPPR2 * massregmod = new reglib::MassRegistrationPPR2(0.15);
	if(savePath.size() != 0){
		massregmod->savePath = savePath+"/segment_"+std::to_string(models.front()->id);
	}
	massregmod->timeout = 1200;
	massregmod->viewer = viewer;
	massregmod->visualizationLvl = debugg > 1;
	massregmod->maskstep = 11;//std::max(1,int(0.4*double(models[i]->frames.size())));
	massregmod->nomaskstep = 11;//std::max(3,int(0.5+0.*double(models[i]->frames.size())));//std::max(1,int(0.5+1.0*double(model->frames.size())));
	massregmod->nomask = true;
	massregmod->stopval = 0.0001;

	reglib::RegistrationRandom *	reg	= new reglib::RegistrationRandom();
	reglib::ModelUpdaterBasicFuse * mu	= new reglib::ModelUpdaterBasicFuse( models.front(), reg);
	mu->occlusion_penalty               = 15;
	mu->massreg_timeout                 = 60*4;
	mu->viewer							= viewer;
	printf("total segment part1 time: %5.5fs\n",getTime()-startTime);

	std::vector<Eigen::Matrix4d> cpmod;
	if(models.size() > 0 && bgs.size() > 0){
		for(unsigned int i = 0; i < bgs.size(); i++){
			if(bgs[i]->points.size() == 0){bgs[i]->recomputeModelPoints();}
			cpmod.push_back(bgs.front()->frames.front()->pose.inverse() * bgs[i]->frames.front()->pose);
			massregmod->addModel(bgs[i]);
		}

		for(int j = 0; j < models.size(); j++){
			if(models[j]->points.size() == 0){models[j]->recomputeModelPoints();}
			cpmod.push_back(bgs.front()->frames.front()->pose.inverse() * models[j]->frames.front()->pose);//bg->relativeposes.front().inverse() * models[j]->relativeposes.front());
			massregmod->addModel(models[j]);
		}
		printf("total segment part1b time: %5.5fs\n",getTime()-startTime);
		reglib::MassFusionResults mfrmod = massregmod->getTransforms(cpmod);

		for(unsigned int i = 0; i < bgs.size(); i++){cpmod[i] = mfrmod.poses[i];}

		for(int j = 0; j < models.size(); j++){
			Eigen::Matrix4d change = mfrmod.poses[j+bgs.size()];
			for(unsigned int k = 0; k < models[j]->relativeposes.size(); k++){
				models[j]->relativeposes[k] = change*models[j]->relativeposes[k];
			}
			for(unsigned int k = 0; k < models[j]->submodels_relativeposes.size(); k++){
				models[j]->submodels_relativeposes[k] = change*models[j]->submodels_relativeposes[k];
			}
		}
	}else if(models.size() > 1){
		for(int j = 0; j < models.size(); j++){
			if(models[j]->points.size() == 0){models[j]->recomputeModelPoints();}
			cpmod.push_back(models.front()->relativeposes.front().inverse() * models[j]->relativeposes.front());
			massregmod->addModel(models[j]);
		}
		reglib::MassFusionResults mfrmod = massregmod->getTransforms(cpmod);
		for(int j = 0; j < models.size(); j++){
			Eigen::Matrix4d change = mfrmod.poses[j] * cpmod[j].inverse();

			for(unsigned int k = 0; k < models[j]->relativeposes.size(); k++){
				models[j]->relativeposes[k] = change*models[j]->relativeposes[k];
			}

			for(unsigned int k = 0; k < models[j]->submodels_relativeposes.size(); k++){
				models[j]->submodels_relativeposes[k] = change*models[j]->submodels_relativeposes[k];
			}
		}
	}
	delete massregmod;

	printf("total segment part2 time: %5.5fs\n",getTime()-startTime);

	std::vector<Eigen::Matrix4d> bgcp;
	std::vector<reglib::RGBDFrame*> bgcf;
	std::vector<cv::Mat> bgmask;
	for(unsigned int i = 0; i < bgs.size(); i++){
		Eigen::Matrix4d cp = cpmod[i];
		std::cout << cp << std::endl;
		for(unsigned int k = 0; k < bgs[i]->relativeposes.size(); k++){
			bgcp.push_back(cp * bgs[i]->relativeposes[k]);
			bgcf.push_back(bgs[i]->frames[k]);
			bgmask.push_back(bgs[i]->modelmasks[k]->getMask());
		}
	}
	for(int j = 0; j < models.size(); j++){
		reglib::Model * model = models[j];

		std::vector<cv::Mat> masks;
		for(unsigned int i = 0; i < model->frames.size(); i++){
			reglib::RGBDFrame * frame = model->frames[i];
			reglib::Camera * cam = frame->camera;
			cv::Mat mask;
			mask.create(cam->height,cam->width,CV_8UC1);
			unsigned char * maskdata = (unsigned char *)(mask.data);
			for(unsigned int k = 0; k < cam->height*cam->width;k++){maskdata[k] = 255;}
			masks.push_back(mask);
		}

		std::vector<cv::Mat> movemask;
		std::vector<cv::Mat> dynmask;
		printf("computeMovingDynamicStatic\n");
		mu->computeMovingDynamicStatic(movemask,dynmask,bgcp,bgcf,model->relativeposes,model->frames,debugg,savePath);//Determine self occlusions
		external.push_back(movemask);
		internal.push_back(masks);
		dynamic.push_back(dynmask);
	}

	delete reg;
	delete mu;
	printf("total segment time: %5.5fs\n",getTime()-startTime);
}

/*
void segment(std::vector< reglib::Model * > bgs, std::vector< reglib::Model * > models, std::vector< std::vector< cv::Mat > > & internal, std::vector< std::vector< cv::Mat > > & external, std::vector< std::vector< cv::Mat > > & dynamic, bool debugg){
	double startTime = getTime();
	printf("running segment method\n");
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
	if(debugg){
		viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer ("viewer"));
		viewer->addCoordinateSystem(0.01);
		viewer->setBackgroundColor(1.0,1.0,1.0);
	}

	reglib::MassRegistrationPPR2 * massregmod = new reglib::MassRegistrationPPR2(0.05);
	massregmod->timeout = 1200;
	massregmod->viewer = viewer;
	massregmod->visualizationLvl = 1;
	massregmod->maskstep = 10;//std::max(1,int(0.4*double(models[i]->frames.size())));
	massregmod->nomaskstep = 10;//std::max(3,int(0.5+0.*double(models[i]->frames.size())));//std::max(1,int(0.5+1.0*double(model->frames.size())));
	massregmod->nomask = true;
	massregmod->stopval = 0.0001;

	reglib::RegistrationRandom *	reg	= new reglib::RegistrationRandom();
	reglib::ModelUpdaterBasicFuse * mu	= new reglib::ModelUpdaterBasicFuse( models.front(), reg);
	mu->occlusion_penalty               = 15;
	mu->massreg_timeout                 = 60*4;
	mu->viewer							= viewer;
	printf("total segment part1 time: %5.5fs\n",getTime()-startTime);

	if(models.size() > 0 && bg->frames.size() > 0){
		std::vector<Eigen::Matrix4d> cpmod;
		if(bg->points.size() == 0){bg->recomputeModelPoints();}

		cpmod.push_back(Eigen::Matrix4d::Identity());
		massregmod->addModel(bg);

		for(int j = 0; j < models.size(); j++){
			if(models[j]->points.size() == 0){models[j]->recomputeModelPoints();}
			cpmod.push_back(bg->frames.front()->pose.inverse() * models[j]->frames.front()->pose);//bg->relativeposes.front().inverse() * models[j]->relativeposes.front());
			massregmod->addModel(models[j]);
		}
		printf("total segment part1b time: %5.5fs\n",getTime()-startTime);
		reglib::MassFusionResults mfrmod = massregmod->getTransforms(cpmod);
		for(int j = 0; j < models.size(); j++){
			Eigen::Matrix4d change = mfrmod.poses[j+1];
			for(unsigned int k = 0; k < models[j]->relativeposes.size(); k++){
				models[j]->relativeposes[k] = change*models[j]->relativeposes[k];
			}
			for(unsigned int k = 0; k < models[j]->submodels_relativeposes.size(); k++){
				models[j]->submodels_relativeposes[k] = change*models[j]->submodels_relativeposes[k];
			}
		}
	}else if(models.size() > 1){
		std::vector<Eigen::Matrix4d> cpmod;
		for(int j = 0; j < models.size(); j++){
			if(models[j]->points.size() == 0){models[j]->recomputeModelPoints();}
			cpmod.push_back(models.front()->relativeposes.front().inverse() * models[j]->relativeposes.front());
			massregmod->addModel(models[j]);
		}
		reglib::MassFusionResults mfrmod = massregmod->getTransforms(cpmod);
		for(int j = 0; j < models.size(); j++){
			Eigen::Matrix4d change = mfrmod.poses[j] * cpmod[j].inverse();

			for(unsigned int k = 0; k < models[j]->relativeposes.size(); k++){
				models[j]->relativeposes[k] = change*models[j]->relativeposes[k];
			}

			for(unsigned int k = 0; k < models[j]->submodels_relativeposes.size(); k++){
				models[j]->submodels_relativeposes[k] = change*models[j]->submodels_relativeposes[k];
			}
		}
	}
	delete massregmod;

	printf("total segment part2 time: %5.5fs\n",getTime()-startTime);

	std::vector<Eigen::Matrix4d> bgcp;
	std::vector<reglib::RGBDFrame*> bgcf;
	std::vector<cv::Mat> bgmask;
	for(unsigned int k = 0; k < bg->relativeposes.size(); k++){
		bgcp.push_back(bg->relativeposes[k]);
		bgcf.push_back(bg->frames[k]);
		bgmask.push_back(bg->modelmasks[k]->getMask());
	}
	for(int j = 0; j < models.size(); j++){
		reglib::Model * model = models[j];

		std::vector<cv::Mat> masks;
		for(unsigned int i = 0; i < model->frames.size(); i++){
			reglib::RGBDFrame * frame = model->frames[i];
			reglib::Camera * cam = frame->camera;
			cv::Mat mask;
			mask.create(cam->height,cam->width,CV_8UC1);
			unsigned char * maskdata = (unsigned char *)(mask.data);
			for(unsigned int k = 0; k < cam->height*cam->width;k++){maskdata[k] = 255;}
			masks.push_back(mask);
		}

		std::vector<cv::Mat> movemask;
		std::vector<cv::Mat> dynmask;
		printf("computeMovingDynamicStatic\n");
		mu->computeMovingDynamicStatic(movemask,dynmask,bgcp,bgcf,model->relativeposes,model->frames,debugg);//Determine self occlusions
		external.push_back(movemask);
		internal.push_back(masks);
		dynamic.push_back(dynmask);
	}

	delete reg;
	delete mu;
	printf("total segment time: %5.5fs\n",getTime()-startTime);
}
*/

std::vector<std::string> getFileList(std::string path){
	std::vector<std::string> files;

	DIR *dp;
	struct dirent *ep;
	dp = opendir (path.c_str());
	if (dp != NULL) {
		while (ep = readdir (dp)){
			if (ep->d_type != DT_DIR) {
				files.push_back(std::string(ep->d_name));
			}
		}
		closedir (dp);
	}else{
		printf("Couldn't open %s\n",path.c_str());
	}

	return files;
}

std::vector<std::string> getFolderList(std::string path){
	std::vector<std::string> files;

	DIR *dp;
	struct dirent *ep;
	dp = opendir (path.c_str());
	if (dp != NULL) {
		while (ep = readdir (dp)){
			std::string data = std::string(ep->d_name);
			if (data.compare(".") != 0 && data.compare("..") != 0 && ep->d_type == DT_DIR) {
				files.push_back(data);
			}
		}
		closedir (dp);
	}else{
		printf("Couldn't open %s\n",path.c_str());
	}
	return files;
}

void recursiveGetFolderList(std::vector<std::string> & folders, std::string path){
	std::vector<std::string> list = getFolderList(path);
	for(unsigned int i = 0; i < list.size(); i++){
		folders.push_back(path+"/"+list[i]);
		recursiveGetFolderList(folders, path+"/"+list[i]);
	}
}

std::vector<std::string> recursiveGetFolderList(std::string path){
	std::vector<std::string> folders;
	recursiveGetFolderList(folders, path);
	return folders;
}

Eigen::Matrix4f getRegisteredViewPoses(const std::string& poses_file){
	Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
	std::ifstream in(poses_file);
	if (!in.is_open()){
		cout<<"ERROR: cannot find poses file "<<poses_file<<endl;
		return transform;
	}
	cout<<"Loading view pose from "<<poses_file<<endl;


	float temp;
	for (size_t j=0; j<4; j++){
		for (size_t k=0; k<4; k++){
			in >> temp;
			transform(j,k) = temp;
		}
	}
	in.close();
	return transform;
}

std::vector<int> getIndsFromFile(const std::string& poses_file){
	std::vector<int> indexes;
	std::ifstream in(poses_file);
	if (!in.is_open()){
		cout<<"ERROR: cannot find idnex file "<<poses_file<<endl;
		return indexes;
	}
	cout<<"Loading indexes from "<<poses_file<<endl;


	std::string line;
	while ( getline (in,line) )
	{
		indexes.push_back(std::stoi(line));

	}
	in.close();

	return indexes;
}




std::vector<reglib::Model *> loadModelsPCDs(std::string path){
	std::vector<reglib::Model *> models;

	std::vector<std::string> folders = getFolderList(path);//recursiveGetFolderList(path);//getFolderList(path);//
	for(unsigned int i = 0; i < folders.size(); i++){
		printf("folders: %s\n",folders[i].c_str());
		std::string views = path+folders[i]+"/views";
		std::vector<std::string> files = getFileList(views);

		std::vector<std::string> cloudspaths;
		std::vector<std::string> indexpaths;
		std::vector<std::string> posepaths;

		for(unsigned int j = 0; j < files.size(); j++){
			std::string file = views+"/"+files[j];
			printf("Files: %s\n",file.c_str());
			if ((files[j].find("cloud") != std::string::npos) && (files[j].find(".pcd") !=std::string::npos)){cloudspaths.push_back(file);}
			if (files[j].find("indices") != std::string::npos){indexpaths.push_back(file);}
			if (files[j].find("pose") != std::string::npos){posepaths.push_back(file);}
		}

		printf("%i %i %i\n",cloudspaths.size(),indexpaths.size(),posepaths.size());
		if((cloudspaths.size() == indexpaths.size()) && (cloudspaths.size() == posepaths.size())){
			std::sort (cloudspaths.begin(), cloudspaths.end());
			std::sort (indexpaths.begin(), indexpaths.end());
			std::sort (posepaths.begin(), posepaths.end());


			reglib::Model * model = new reglib::Model();

//			for(unsigned int i = 0; i < msg.local_poses.size(); i++){
//				sensor_msgs::CameraInfo		camera			= msg.frames[i].camera;
//				ros::Time					capture_time	= msg.frames[i].capture_time;
//				geometry_msgs::Pose			pose			= msg.frames[i].pose;

//				cv_bridge::CvImagePtr			rgb_ptr;
//				try{							rgb_ptr = cv_bridge::toCvCopy(msg.frames[i].rgb, sensor_msgs::image_encodings::BGR8);}
//				catch (cv_bridge::Exception& e){ROS_ERROR("cv_bridge exception: %s", e.what());}
//				cv::Mat rgb = rgb_ptr->image;

//				cv_bridge::CvImagePtr			depth_ptr;
//				try{							depth_ptr = cv_bridge::toCvCopy(msg.frames[i].depth, sensor_msgs::image_encodings::MONO16);}
//				catch (cv_bridge::Exception& e){ROS_ERROR("cv_bridge exception: %s", e.what());}
//				cv::Mat depth = depth_ptr->image;

//				Eigen::Affine3d epose;
//				tf::poseMsgToEigen(pose, epose);

//				reglib::Camera * cam		= new reglib::Camera();
//				if(camera.K[0] > 0){
//					cam->fx = camera.K[0];
//					cam->fy = camera.K[4];
//					cam->cx = camera.K[2];
//					cam->cy = camera.K[5];
//				}

//			}
//

			for(unsigned int j =  0; j  < cloudspaths.size(); j++){
				std::string cloudpath = cloudspaths[j];
				std::string indexpath = indexpaths[j];
				std::string posepath = posepaths[j];
				printf("loading: %s %s %s\n", cloudpath.c_str(),indexpath.c_str(),posepath.c_str());

				pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
				if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (cloudpath, *cloud) != -1){
					Eigen::Matrix4f pose = getRegisteredViewPoses(posepath);
					std::vector<int> indexes = getIndsFromFile(indexpath);
					std::cout << pose << std::endl << std::endl;
					std::cout << indexes.size() << std::endl;

					unsigned int width = cloud->width;
					unsigned int height = cloud->height;
					cv::Mat rgb;
					rgb.create(height,width,CV_8UC3);
					unsigned char   * rgbdata   = rgb.data;

					cv::Mat depth;
					depth.create(height,width,CV_16UC1);
					unsigned short  * depthdata = (unsigned short *)(depth.data);

					reglib::Camera * cam = new reglib::Camera();//figure out cam params

					//TODO figure out optical centre and focal lengths from cloud...
//					for(unsigned int w = 0; w < width; w++){
//						for(unsigned int w = 0; w < width; w++){
//						pcl::PointXYZRGB p = cloud->points[k];
//						float x = p.x;
//						float y = p.y;
//						float z = p.z;
//						}
//					}

					for(unsigned int k = 0; k < width*height; k++){
						pcl::PointXYZRGB p = cloud->points[k];
						rgbdata[3*k+0] = p.b;
						rgbdata[3*k+1] = p.g;
						rgbdata[3*k+2] = p.r;
						if(!std::isnan(p.z)){
							depthdata[k] = 5000*p.z;
						}else{
							depthdata[k] = 0;
						}
					}


					cv::Mat mask;
					mask.create(height,width,CV_8UC1);
					unsigned char   * maskdata   = mask.data;
					for(unsigned int k = 0; k < indexes.size(); k++){maskdata[indexes[k]] = 255;}

					reglib::RGBDFrame * frame = new reglib::RGBDFrame(cam,rgb, depth, 0 , pose.cast<double>() , true,"",false);
					model->frames.push_back(frame);
					model->relativeposes.push_back(model->frames.front()->pose.inverse() * pose.cast<double>());
					model->modelmasks.push_back(new reglib::ModelMask(mask));
				}else{
					printf ("Couldn't read pcd file\n");
				}
			}

			model->recomputeModelPoints();
			models.push_back(model);
		}else{
			printf("number of clouds, indices and poses dont match... ignoring\n");
		}
	}

	return models;
}

}
