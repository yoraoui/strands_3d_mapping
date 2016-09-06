#include "Util.h"

namespace quasimodo_brain {

double getTime(){
	struct timeval start1;
	gettimeofday(&start1, NULL);
	return double(start1.tv_sec+(start1.tv_usec/1000000.0));
}

reglib::Model * getModelFromMSG(quasimodo_msgs::model & msg){
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

		reglib::RGBDFrame * frame = new reglib::RGBDFrame(cam,rgb, depth, double(capture_time.sec)+double(capture_time.nsec)/1000000000.0, epose.matrix());
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

		if(addClouds){}
		sensor_msgs::PointCloud2 output;
		pcl::toROSMsg(*(model->frames[i]->getPCLcloud()), output);
		printf("outputcloud: %i %i\n",output.point_step,output.row_step);

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
	return epose.matrix();
}

reglib::Model * load_metaroom_model(std::string sweep_xml){
	int slash_pos = sweep_xml.find_last_of("/");
	std::string sweep_folder = sweep_xml.substr(0, slash_pos) + "/";
	printf("folder: %s\n",sweep_folder.c_str());

	SimpleXMLParser<pcl::PointXYZRGB> parser;
	SimpleXMLParser<pcl::PointXYZRGB>::RoomData roomData  = parser.loadRoomFromXML(sweep_folder+"/room.xml");

	reglib::Model * sweepmodel = 0;

	Eigen::Matrix4d m2 = getMat(roomData.vIntermediateRoomCloudTransforms[0]);
	cout << m2 << endl << endl;

	std::vector<reglib::RGBDFrame * > current_room_frames;
	for (size_t i=0; i<roomData.vIntermediateRoomClouds.size(); i++)
	{
		//if(i < 7 || i > 8){continue;}

		cv::Mat fullmask;
		fullmask.create(480,640,CV_8UC1);
		unsigned char * maskdata = (unsigned char *)fullmask.data;
		for(int j = 0; j < 480*640; j++){maskdata[j] = 255;}

		reglib::Camera * cam		= new reglib::Camera();//TODO:: ADD TO CAMERAS
//		cam->fx = 532.158936;
//		cam->fy = 533.819214;
//		cam->cx = 310.514310;
//		cam->cy = 236.842039;

		image_geometry::PinholeCameraModel aCameraModel = roomData.vIntermediateRoomCloudCamParamsCorrected[i];
		//rc->initializeCamera(aCameraModel.fx(), aCameraModel.fy(), aCameraModel.cx(), aCameraModel.cy(), aCameraModel.fullResolution().width, aCameraModel.fullResolution().height);

		cam->fx = aCameraModel.fx();
		cam->fy = aCameraModel.fy();
		cam->cx = aCameraModel.cx();
		cam->cy = aCameraModel.cy();
		cam->print();
		//aCameraModel.fullResolution().width, aCameraModel.fullResolution().height);


		//		cout<<"Intermediate cloud size "<<roomData.vIntermediateRoomClouds[i]->points.size()<<endl;

		//		printf("%i / %i\n",i,roomData.vIntermediateRoomClouds.size());






		//		//Transform
		//		tf::StampedTransform tf	= roomData.vIntermediateRoomCloudTransformsRegistered[i];
		//		geometry_msgs::TransformStamped tfstmsg;
		//		tf::transformStampedTFToMsg (tf, tfstmsg);
		//		geometry_msgs::Transform tfmsg = tfstmsg.transform;
		//		geometry_msgs::Pose		pose;
		//		pose.orientation		= tfmsg.rotation;
		//		pose.position.x		= tfmsg.translation.x;
		//		pose.position.y		= tfmsg.translation.y;
		//		pose.position.z		= tfmsg.translation.z;
		//		Eigen::Affine3d epose;
		//		tf::poseMsgToEigen(pose, epose);

		Eigen::Matrix4d m = m2*getMat(roomData.vIntermediateRoomCloudTransformsRegistered[i]);

		//		cout << m << endl << endl;

		reglib::RGBDFrame * frame = new reglib::RGBDFrame(cam,roomData.vIntermediateRGBImages[i],5.0*roomData.vIntermediateDepthImages[i],0, m);

		current_room_frames.push_back(frame);
		if(sweepmodel == 0){
			sweepmodel = new reglib::Model(frame,fullmask);
		}else{
			sweepmodel->frames.push_back(frame);
			sweepmodel->relativeposes.push_back(current_room_frames.front()->pose.inverse() * frame->pose);
			sweepmodel->modelmasks.push_back(new reglib::ModelMask(fullmask));
		}
	}

	//sweepmodel->recomputeModelPoints();
	printf("nr points: %i\n",sweepmodel->points.size());

	return sweepmodel;
}

void segment(reglib::Model * bg, std::vector< reglib::Model * > models, std::vector< std::vector< cv::Mat > > & internal, std::vector< std::vector< cv::Mat > > & external, std::vector< std::vector< cv::Mat > > & dynamic, bool debugg){
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
	if(debugg){
		viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer ("viewer"));
		viewer->addCoordinateSystem(0.01);
		viewer->setBackgroundColor(1.0,1.0,1.0);
	}

	reglib::MassRegistrationPPR2 * massregmod = new reglib::MassRegistrationPPR2(0.05);
	massregmod->timeout = 1200;
	massregmod->viewer = viewer;
	massregmod->visualizationLvl = 0;
	massregmod->maskstep = 10;//std::max(1,int(0.4*double(models[i]->frames.size())));
	massregmod->nomaskstep = 10;//std::max(3,int(0.5+0.*double(models[i]->frames.size())));//std::max(1,int(0.5+1.0*double(model->frames.size())));
	massregmod->nomask = true;
	massregmod->stopval = 0.0001;


	reglib::RegistrationRandom *	reg	= new reglib::RegistrationRandom();
	reglib::ModelUpdaterBasicFuse * mu	= new reglib::ModelUpdaterBasicFuse( models.front(), reg);
	mu->occlusion_penalty               = 15;
	mu->massreg_timeout                 = 60*4;
	mu->viewer							= viewer;

	if(models.size() > 0 && bg->frames.size() > 0){
		std::vector<Eigen::Matrix4d> cpmod;

		bg->points = mu->getSuperPoints(bg->relativeposes,bg->frames,bg->modelmasks,1,false);

		cpmod.push_back(Eigen::Matrix4d::Identity());//,models.front()->relativeposes.front().inverse() * bg->relativeposes.front());
		massregmod->addModel(bg);
		printf("bg->points = %i\n",bg->points.size());

		for(int j = 0; j < models.size(); j++){
			models[j]->points			= mu->getSuperPoints(models[j]->relativeposes,models[j]->frames,models[j]->modelmasks,1,false);
			cpmod.push_back(bg->frames.front()->pose.inverse() * models[j]->frames.front()->pose);//bg->relativeposes.front().inverse() * models[j]->relativeposes.front());
			massregmod->addModel(models[j]);
		}


		reglib::MassFusionResults mfrmod = massregmod->getTransforms(cpmod);
		for(int j = 0; j < models.size(); j++){
			Eigen::Matrix4d change = mfrmod.poses[j+1];// * cpmod[j+1].inverse();

			for(unsigned int k = 0; k < models[j]->relativeposes.size(); k++){
				models[j]->relativeposes[k] = change*models[j]->relativeposes[k];
			}

			for(unsigned int k = 0; k < models[j]->submodels_relativeposes.size(); k++){
				models[j]->submodels_relativeposes[k] = change*models[j]->submodels_relativeposes[k];
			}
		}
	}else if(models.size() > 1){
		std::vector<Eigen::Matrix4d> cpmod;

		Eigen::Matrix4d first = Eigen::Matrix4d::Identity();
		//Eigen::Matrix4d first = mod_po_vec.front().front();
		for(int j = 0; j < models.size(); j++){
			models[j]->points	= mu->getSuperPoints(models[j]->relativeposes,models[j]->frames,models[j]->modelmasks,1,false);
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
		mu->computeMovingDynamicStatic(movemask,dynmask,bgcp,bgcf,model->relativeposes,model->frames,debugg);//Determine self occlusions
		external.push_back(movemask);
		internal.push_back(masks);
		dynamic.push_back(dynmask);

//		for(unsigned int i = 0; i < model->frames.size(); i++){
//			cv::imshow( "rgb",		model->frames[i]->rgb );
//			cv::imshow( "movemask",	movemask[i] );
//			cv::imshow( "dynmask",	dynmask[i] );
//			cv::waitKey(0);
//		}

//		std::vector<cv::Mat> internal_masks = mu->computeDynamicObject(bgcp,bgcf,bgmask,model->relativeposes,model->frames,masks,model->relativeposes,model->frames,masks,debugg);//Determine self occlusions
//		std::vector<cv::Mat> external_masks = mu->computeDynamicObject(model->relativeposes,model->frames,masks,bgcp,bgcf,bgmask,model->relativeposes,model->frames,masks,debugg);//Determine external occlusions
//		std::vector<cv::Mat> dynamic_masks;
//		for(unsigned int i = 0; i < model->frames.size(); i++){
//			reglib::RGBDFrame * frame = model->frames[i];
//			reglib::Camera * cam = frame->camera;
//			cv::Mat mask;
//			mask.create(cam->height,cam->width,CV_8UC1);
//			unsigned char * maskdata = (unsigned char *)(mask.data);
//			for(unsigned int k = 0; k < cam->height*cam->width;k++){maskdata[k] = 255;}

//			unsigned char * internalmaskdata = (unsigned char *)(internal_masks[i].data);
//			unsigned char * externalmaskdata = (unsigned char *)(external_masks[i].data);
//			for(unsigned int k = 0; k < cam->height * cam->width;k++){
//				if(externalmaskdata[k] == 0 && internalmaskdata[k] != 0 ){
//					maskdata[k] = 255;
//				}else{
//					maskdata[k] = 0;
//				}
//			}

//			dynamic_masks.push_back(mask);

//			//            cv::imshow( "rgb", frame->rgb );
//			//            cv::imshow( "internal_masks",	internal_masks[i] );
//			//            cv::imshow( "externalmask",		external_masks[i] );
//			//            cv::imshow( "dynamic_mask",		dynamic_masks[i] );
//			//            cv::waitKey(0);
//		}

//		internal.push_back(internal_masks);
//		external.push_back(external_masks);
//		dynamic.push_back(dynamic_masks);
	}

	delete reg;
	delete mu;
}

}
