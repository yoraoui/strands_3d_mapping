#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/String.h"

#include "eigen_conversions/eigen_msg.h"
#include "tf_conversions/tf_eigen.h"
#include <tf_conversions/tf_eigen.h>

#include "quasimodo_msgs/model.h"
#include "quasimodo_msgs/rgbd_frame.h"
#include "quasimodo_msgs/model_from_frame.h"
#include "quasimodo_msgs/index_frame.h"
#include "quasimodo_msgs/fuse_models.h"
#include "quasimodo_msgs/get_model.h"
#include "quasimodo_msgs/retrieval_query_result.h"
#include "quasimodo_msgs/retrieval_query.h"
#include "quasimodo_msgs/retrieval_result.h"
#include "soma_msgs/SOMAObject.h"

#include "soma_manager/SOMAInsertObjs.h"

//#include "modelupdater/ModelUpdater.h"
//#include "/home/johane/catkin_ws_dyn/src/quasimodo_models/include/modelupdater/ModelUpdater.h"
#include "modelupdater/ModelUpdater.h"
#include "core/RGBDFrame.h"
#include <sensor_msgs/PointCloud2.h>
#include <string.h>

#include "metaroom_xml_parser/simple_xml_parser.h"
#include "metaroom_xml_parser/simple_summary_parser.h"

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <map>

#include "ModelDatabase/ModelDatabase.h"

#include <thread>

#include <sys/time.h>

#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <vector>
#include <string>
#include <iostream>

#include "Util/Util.h"

using namespace quasimodo_brain;

bool visualization = false;
bool show_db = false;//Full database show
bool save_db = false;//Full database save
int save_db_counter = 0;

int show_init_lvl = 0;//init show
int show_refine_lvl = 0;//refine show
int show_reg_lvl = 0;//registration show
bool show_scoring = false;//fuse scoring show
bool show_search = false;
bool show_modelbuild = false;


std::map<int , reglib::Camera *>		cameras;
std::map<int , reglib::RGBDFrame *>		frames;

std::map<int , reglib::Model *>			models;
std::map<int , reglib::ModelUpdater *>	updaters;

std::set<std::string>					framekeys;
reglib::RegistrationRandom *			registration;
ModelDatabase * 						modeldatabase;

std::string								savepath = ".";

boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;

ros::Publisher models_new_pub;
ros::Publisher models_updated_pub;
ros::Publisher models_deleted_pub;

ros::Publisher model_pcd_pub;
ros::Publisher database_pcd_pub;
ros::Publisher model_history_pub;
ros::Publisher model_places_pub;
ros::Publisher model_last_pub;
ros::Publisher chatter_pub;

//<<<<<<< HEAD
ros::ServiceClient retrieval_client;
ros::ServiceClient conversion_client;
ros::ServiceClient insert_client;

//ros::ServiceClient soma2add;
////=======
//ros::ServiceClient soma_add;
//>>>>>>> integrate_object_search

double occlusion_penalty = 10;
double massreg_timeout = 120;

bool run_search = false;
double search_timeout = 30;

int sweepid_counter = 0;

int current_model_update = 0;

bool myfunction (reglib::Model * i,reglib::Model * j) { return (i->frames.size() + i->submodels.size())  > (j->frames.size() + j->submodels.size()); }

quasimodo_msgs::retrieval_result sresult;
bool new_search_result = false;
double last_search_result_time = 0;

void publishDatabasePCD(bool original_colors = false){
	std::vector<reglib::Model *> results;
	for(unsigned int i = 0; i < modeldatabase->models.size(); i++){results.push_back(modeldatabase->models[i]);}
	std::sort (results.begin(), results.end(), myfunction);

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr	conccloud	(new pcl::PointCloud<pcl::PointXYZRGB>);

	float maxx = 0;
	for(unsigned int i = 0; i < results.size(); i++){
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = results[i]->getPCLcloud(1, false);
		float meanx = 0;
		float meany = 0;
		float meanz = 0;
		for(unsigned int j = 0; j < cloud->points.size(); j++){
			meanx += cloud->points[j].x;
			meany += cloud->points[j].y;
			meanz += cloud->points[j].z;
		}
		meanx /= float(cloud->points.size());
		meany /= float(cloud->points.size());
		meanz /= float(cloud->points.size());

		for(unsigned int j = 0; j < cloud->points.size(); j++){
			cloud->points[j].x -= meanx;
			cloud->points[j].y -= meany;
			cloud->points[j].z -= meanz;
		}

		float minx = 100000000000;

		for(unsigned int j = 0; j < cloud->points.size(); j++){minx = std::min(cloud->points[j].x , minx);}
		for(unsigned int j = 0; j < cloud->points.size(); j++){cloud->points[j].x += maxx-minx + 0.15;}
		for(unsigned int j = 0; j < cloud->points.size(); j++){maxx = std::max(cloud->points[j].x,maxx);}

		for(unsigned int j = 0; j < cloud->points.size(); j++){conccloud->points.push_back(cloud->points[j]);}
	}

	sensor_msgs::PointCloud2 input;
	pcl::toROSMsg (*conccloud,input);//, *transformed_cloud);
	input.header.frame_id = "/map";
	database_pcd_pub.publish(input);
}

void publish_history(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> history){
	for(unsigned int i = 0; i < history.size(); i++){
		sensor_msgs::PointCloud2 input;
		pcl::toROSMsg (*history[i],input);//, *transformed_cloud);
		input.header.frame_id = "/map";
		model_history_pub.publish(input);
	}
}

void dumpDatabase(std::string path = "."){
	char command [1024];
	sprintf(command,"rm -r -f %s/database_tmp",path.c_str());
	int r = system(command);

	sprintf(command,"mkdir %s/database_tmp",path.c_str());
	r = system(command);

	for(unsigned int m = 0; m < modeldatabase->models.size(); m++){
		char buf [1024];
		sprintf(buf,"%s/database_tmp/model_%08i",path.c_str(),m);
		sprintf(command,"mkdir -p %s",buf);
		r = system(command);
		modeldatabase->models[m]->save(std::string(buf));
	}
}

void retrievalCallback(const quasimodo_msgs::retrieval_query_result & qr){
	printf("retrievalCallback\n");
	sresult = qr.result;
	new_search_result = true;
	last_search_result_time = getTime();
}

void showModels(std::vector<reglib::Model *> mods){
	//viewer->removeAllPointClouds();
	float maxx = 0;

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr	conccloud	(new pcl::PointCloud<pcl::PointXYZRGB>);
	for(unsigned int i = 0; i < mods.size(); i++){
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = mods[i]->getPCLcloud(1, false);
		float meanx = 0;
		float meany = 0;
		float meanz = 0;
		for(unsigned int j = 0; j < cloud->points.size(); j++){
			meanx += cloud->points[j].x;
			meany += cloud->points[j].y;
			meanz += cloud->points[j].z;
		}
		meanx /= float(cloud->points.size());
		meany /= float(cloud->points.size());
		meanz /= float(cloud->points.size());

		for(unsigned int j = 0; j < cloud->points.size(); j++){
			cloud->points[j].x -= meanx;
			cloud->points[j].y -= meany;
			cloud->points[j].z -= meanz;
		}

		float minx = 100000000000;

		for(unsigned int j = 0; j < cloud->points.size(); j++){minx = std::min(cloud->points[j].x , minx);}
		for(unsigned int j = 0; j < cloud->points.size(); j++){cloud->points[j].x += maxx-minx + 0.15;}
		for(unsigned int j = 0; j < cloud->points.size(); j++){maxx = std::max(cloud->points[j].x,maxx);}

//		char buf [1024];
//		sprintf(buf,"cloud%i",i);
//		viewer->addPointCloud<pcl::PointXYZRGB> (cloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(cloud), buf);

		*conccloud += *cloud;
	}

	if( save_db ){
		printf("save_db\n");
		char buf [1024];
		sprintf(buf,"quasimodoDB_%i.pcd",save_db_counter);
		pcl::io::savePCDFileBinaryCompressed(buf, *conccloud);
		save_db_counter++;
	}

	if(show_db && visualization){
		viewer->removeAllPointClouds();
		viewer->addPointCloud<pcl::PointXYZRGB> (conccloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(conccloud), "conccloud");
		viewer->spin();
	}

//	printf("show_sorted()\n");
//	exit(0);
}

int savecounter = 0;
void show_sorted(){
	if(!show_db && !save_db ){return;}
	//if(!visualization){return;}
	std::vector<reglib::Model *> results;
	for(unsigned int i = 0; i < modeldatabase->models.size(); i++){results.push_back(modeldatabase->models[i]);}
	std::sort (results.begin(), results.end(), myfunction);
	showModels(results);
}

std::vector<soma_msgs::SOMAObject> getSOMA2ObjectMSGs(reglib::Model * model){
	std::vector<soma_msgs::SOMAObject> msgs;
	for(unsigned int i = 0; i < model->frames.size(); i++){
		soma_msgs::SOMAObject msg;
		char buf [1024];
		sprintf(buf,"id_%i_%i",int(model->id),int(model->frames[i]->id));
		msg.id				= std::string(buf);
		msg.map_name		= "whatevermapname";				//	#### the global map that the object belongs. Automatically filled by SOMA2 insert service
		msg.map_unique_id	= "";								// 	#### the unique db id of the map. Automatically filled by SOMA2 insert service
		msg.config			= "configid";						//	#### the corresponding world configuration. This could be incremented as config1, config2 at each meta-room observation
		msg.mesh			= "";								//	#### mesh model of the object. Could be left blank
		sprintf(buf,"type_%i",int(model->id));
		msg.type			= std::string(buf);					//	#### type of the object. For higher level objects, this could chair1, chair2. For middle or lower level segments it could be segment1101, segment1102, etc.
		//		msg.waypoint		= "";								//	#### the waypoint id. Could be left blank
		msg.timestep 		= 0;								//	#### this is the discrete observation instance. This could be incremented at each meta-room observation as 1,2,3,etc...
		msg.logtimestamp	= 1000000.0 * double(model->frames[i]->capturetime);	//	#### this is the unix timestamp information that holds the time when the object is logged. SOMA2 uses UTC time values.

		Eigen::Affine3f transform = Eigen::Affine3f(model->frames[i]->pose.cast<float>());//Eigen::Affine3f::Identity();

		bool * maskdata = model->modelmasks[i]->maskvec;
		unsigned char * rgbdata = (unsigned char *) model->frames[i]->rgb.data;
		unsigned short * depthdata = (unsigned short *) model->frames[i]->depth.data;

		const unsigned int width	= model->frames[i]->camera->width; const unsigned int height	= model->frames[i]->camera->height;
		const double idepth			= model->frames[i]->camera->idepth_scale;
		const double cx				= model->frames[i]->camera->cx;		const double cy				= model->frames[i]->camera->cy;
		const double ifx			= 1.0/model->frames[i]->camera->fx;	const double ify			= 1.0/model->frames[i]->camera->fy;

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr	cloud	(new pcl::PointCloud<pcl::PointXYZRGB>);
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr	transformed_cloud	(new pcl::PointCloud<pcl::PointXYZRGB>);
		for(unsigned int w = 0; w < width; w++){
			for(unsigned int h = 0; h < height;h++){
				int ind = h*width+w;
				double z = idepth*double(depthdata[ind]);
				if(z > 0 && maskdata[ind]){
					pcl::PointXYZRGB p;
					p.x = (double(w) - cx) * z * ifx;
					p.y = (double(h) - cy) * z * ify;
					p.z = z;
					p.b = rgbdata[3*ind+0];
					p.g = rgbdata[3*ind+1];
					p.r = rgbdata[3*ind+2];
					cloud->points.push_back(p);
				}
			}
		}

		pcl::transformPointCloud (*cloud, *transformed_cloud, transform);
		sensor_msgs::PointCloud2ConstPtr input;
		pcl::fromROSMsg (*input, *transformed_cloud);


		msg.cloud = *input; //#### The 3D cloud of the object		(with respect to /map frame)

		Eigen::Quaterniond q = Eigen::Quaterniond(Eigen::Affine3d(model->frames[i]->pose).rotation());

		geometry_msgs::Pose		pose;
		pose.orientation.x	= q.x();
		pose.orientation.y	= q.y();
		pose.orientation.z	= q.z();
		pose.orientation.w	= q.w();
		pose.position.x		= model->frames[i]->pose(0,3);
		pose.position.y		= model->frames[i]->pose(1,3);
		pose.position.z		= model->frames[i]->pose(2,3);
		msg.pose = pose;		//	#### Object pose in 3D			(with respect to /map frame)
		//		geometry_msgs/Pose sweepCenter 	#### The pose of the robot during sweep (with respect to /map frame)
		msgs.push_back(msg);
	}

	//	msg.model_id = model->id;
	//	msg.local_poses.resize(model->relativeposes.size());
	//	msg.frames.resize(model->frames.size());
	//	msg.masks.resize(model->modelmasks.size());
	//	for(unsigned int i = 0; i < model->relativeposes.size(); i++){
	//		geometry_msgs::Pose		pose1;
	//		tf::poseEigenToMsg (Eigen::Affine3d(model->relativeposes[i]), pose1);
	//		geometry_msgs::Pose		pose2;
	//		tf::poseEigenToMsg (Eigen::Affine3d(model->frames[i]->pose), pose2);
	//		cv_bridge::CvImage rgbBridgeImage;
	//		rgbBridgeImage.image = model->frames[i]->rgb;
	//		rgbBridgeImage.encoding = "bgr8";
	//		cv_bridge::CvImage depthBridgeImage;
	//		depthBridgeImage.image = model->frames[i]->depth;
	//		depthBridgeImage.encoding = "mono16";
	//		cv_bridge::CvImage maskBridgeImage;
	//		maskBridgeImage.image			= model->modelmasks[i]->getMask();
	//		maskBridgeImage.encoding		= "mono8";
	//		msg.local_poses[i]			= pose1;
	//		msg.frames[i].capture_time	= ros::Time();
	//		msg.frames[i].pose			= pose2;
	//		msg.frames[i].frame_id		= model->frames[i]->id;
	//		msg.frames[i].rgb			= *(rgbBridgeImage.toImageMsg());
	//		msg.frames[i].depth			= *(depthBridgeImage.toImageMsg());
	//		msg.masks[i]				= *(maskBridgeImage.toImageMsg());//getMask()
	//	}
	return msgs;
}

void publishModel(reglib::Model * model){
//<<<<<<< HEAD
//	soma_manager::SOMA2InsertObjs objs;
//	objs.request.objects = getSOMA2ObjectMSGs(model);
//	if (soma2add.call(objs)){
//	}else{ROS_ERROR("Failed to call service soma2add");}
//=======
//	//	std::vector<soma_msgs::SOMAObject> somamsgs = getSOMA2ObjectMSGs(model);
//	soma_manager::SOMAInsertObjs objs;
//	objs.request.objects = getSOMA2ObjectMSGs(model);
//	if (soma_add.call(objs)){//Add frame to model server
//		////			int frame_id = ifsrv.response.frame_id;
//	}else{ROS_ERROR("Failed to call service soma_add");}

//	//	for(int i = 0; i < somamsgs.size(); i++){
//	//		if (soma_add.call(somamsgs[i])){//Add frame to model server
//	////			int frame_id = ifsrv.response.frame_id;
//	//		 }else{ROS_ERROR("Failed to call service soma_add");}
//	//	 }
//>>>>>>> integrate_object_search
}


bool getModel(quasimodo_msgs::get_model::Request  & req, quasimodo_msgs::get_model::Response & res){
	int model_id			= req.model_id;
	reglib::Model * model	= models[model_id];
	res.model = getModelMSG(model);
	return true;
}

bool indexFrame(quasimodo_msgs::index_frame::Request  & req, quasimodo_msgs::index_frame::Response & res){
	sensor_msgs::CameraInfo		camera			= req.frame.camera;
	ros::Time					capture_time	= req.frame.capture_time;
	geometry_msgs::Pose			pose			= req.frame.pose;

	cv_bridge::CvImagePtr			rgb_ptr;
	try{							rgb_ptr = cv_bridge::toCvCopy(req.frame.rgb, sensor_msgs::image_encodings::BGR8);}
	catch (cv_bridge::Exception& e){ROS_ERROR("cv_bridge exception: %s", e.what());return false;}
	cv::Mat rgb = rgb_ptr->image;

	cv_bridge::CvImagePtr			depth_ptr;
	try{							depth_ptr = cv_bridge::toCvCopy(req.frame.depth, sensor_msgs::image_encodings::MONO16);}
	catch (cv_bridge::Exception& e){ROS_ERROR("cv_bridge exception: %s", e.what());return false;}
	cv::Mat depth = depth_ptr->image;

	Eigen::Affine3d epose;
	tf::poseMsgToEigen(pose, epose);

	reglib::Camera * cam		= new reglib::Camera();//TODO:: ADD TO CAMERAS
	cam->fx = camera.K[0];
	cam->fy = camera.K[4];
	cam->cx = camera.K[2];
	cam->cy = camera.K[5];

	reglib::RGBDFrame * frame = new reglib::RGBDFrame(cam,rgb, depth, double(capture_time.sec)+double(capture_time.nsec)/1000000000.0, epose.matrix());
	frames[frame->id] = frame;
	res.frame_id = frame->id;
	return true;
}

bool addToDB(ModelDatabase * database, reglib::Model * model, bool add);//, bool deleteIfFail = false);
bool addIfPossible(ModelDatabase * database, reglib::Model * model, reglib::Model * model2){

	printf("addIfPossible\n");

	reglib::RegistrationRandom *	reg	= new reglib::RegistrationRandom();
	reglib::ModelUpdaterBasicFuse * mu	= new reglib::ModelUpdaterBasicFuse( model2, reg);
	mu->occlusion_penalty               = occlusion_penalty;
	mu->massreg_timeout                 = massreg_timeout;
	mu->viewer							= viewer;
	mu->show_init_lvl					= show_init_lvl;//init show
	mu->show_refine_lvl					= show_refine_lvl;//refine show
	mu->show_scoring					= show_scoring;//fuse scoring show
	reg->visualizationLvl				= show_reg_lvl;

	reglib::FusionResults fr = mu->registerModel(model);

	if(fr.score > 100){
		reglib::UpdatedModels ud = mu->fuseData(&fr, model2, model);
		delete mu;
		delete reg;

		if(ud.deleted_models.size() > 0 || ud.updated_models.size() > 0 || ud.new_models.size() > 0){
			for(unsigned int j = 0; j < ud.deleted_models.size();	j++){
				database->remove(ud.deleted_models[j]);
				delete ud.deleted_models[j];
			}

			for(unsigned int j = 0; j < ud.updated_models.size();	j++){
				database->remove(ud.updated_models[j]);
				models_deleted_pub.publish(getModelMSG(ud.updated_models[j]));
			}


			for(unsigned int j = 0; j < ud.updated_models.size();	j++){
				addToDB(database, ud.updated_models[j], true);
			}

			for(unsigned int j = 0; j < ud.new_models.size();	j++){
				addToDB(database, ud.new_models[j], true);
			}
			return true;
		}
	}else{
		delete mu;
		delete reg;
	}
	return false;
}

bool addToDB(ModelDatabase * database, reglib::Model * model, bool add){// = true){, bool deleteIfFail = false){
	if(add){
		if(model->submodels.size() > 2){
			reglib::RegistrationRandom *	reg	= new reglib::RegistrationRandom();
			reglib::ModelUpdaterBasicFuse * mu	= new reglib::ModelUpdaterBasicFuse( model, 0);
			mu->occlusion_penalty               = occlusion_penalty;
			mu->massreg_timeout                 = massreg_timeout;
			mu->viewer							= viewer;
			reg->visualizationLvl				= 0;
			mu->show_init_lvl = show_init_lvl;//init show
			mu->show_refine_lvl = show_refine_lvl;//refine show
			mu->show_scoring = show_scoring;//fuse scoring show
			mu->refine(0.001,false,0);
			delete mu;
			delete reg;
		}
		database->add(model);
		model->last_changed = ++current_model_update;
	}

	std::vector<reglib::Model * > res = modeldatabase->search(model,1);
	if(show_search){showModels(res);}

	for(unsigned int i = 0; i < res.size(); i++){
		if(addIfPossible(database,model,res[i])){return true;}
	}
	return false;
}

bool runSearch(ModelDatabase * database, reglib::Model * model, int number_of_searches = 5){
	quasimodo_msgs::model_to_retrieval_query m2r;
	m2r.request.model = quasimodo_brain::getModelMSG(model,true);
	if (conversion_client.call(m2r)){
		quasimodo_msgs::query_cloud qc;
		qc.request.query = m2r.response.query;
		qc.request.query.query_kind = qc.request.query.METAROOM_QUERY;
		qc.request.query.number_query = number_of_searches+10;

		if (retrieval_client.call(qc)){
			quasimodo_msgs::retrieval_result result = qc.response.result;

			for(unsigned int i = 0; i < result.retrieved_images.size(); i++){
				for(unsigned int j = 0; j < result.retrieved_images[i].images.size(); j++){
					cv_bridge::CvImagePtr ret_image_ptr;
					try {ret_image_ptr = cv_bridge::toCvCopy(result.retrieved_images[i].images[j], sensor_msgs::image_encodings::BGR8);}
					catch (cv_bridge::Exception& e) {ROS_ERROR("cv_bridge exception: %s", e.what());exit(-1);}

					cv_bridge::CvImagePtr ret_mask_ptr;
					try {ret_mask_ptr = cv_bridge::toCvCopy(result.retrieved_masks[i].images[j], sensor_msgs::image_encodings::MONO8);}
					catch (cv_bridge::Exception& e) {ROS_ERROR("cv_bridge exception: %s", e.what());exit(-1);}

					cv_bridge::CvImagePtr ret_depth_ptr;
					try {ret_depth_ptr = cv_bridge::toCvCopy(result.retrieved_depths[i].images[j], sensor_msgs::image_encodings::MONO16);}
					catch (cv_bridge::Exception& e) {ROS_ERROR("cv_bridge exception: %s", e.what());exit(-1);}

					cv::Mat rgbimage	= ret_image_ptr->image;
					cv::Mat maskimage	= ret_mask_ptr->image;
					cv::Mat depthimage	= ret_depth_ptr->image;

					cv::namedWindow( "rgbimage", cv::WINDOW_AUTOSIZE );			cv::imshow( "rgbimage", rgbimage );
					cv::namedWindow( "maskimage", cv::WINDOW_AUTOSIZE );		cv::imshow( "maskimage", maskimage );
					cv::namedWindow( "depthimage", cv::WINDOW_AUTOSIZE );		cv::imshow( "depthimage", depthimage );
					cv::waitKey( 0 );
				}
			}

				//						int maxval = 0;
				//						int maxind = 0;

				//						int dilation_size = 0;
				//						cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
				//																	cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
				//																	cv::Point( dilation_size, dilation_size ) );
				//						for(unsigned int j = 0; j < sresult.retrieved_images[i].images.size(); j++){
				//							//framekeys

				//							cv_bridge::CvImagePtr ret_image_ptr;
				//							try {ret_image_ptr = cv_bridge::toCvCopy(sresult.retrieved_images[i].images[j], sensor_msgs::image_encodings::BGR8);}
				//							catch (cv_bridge::Exception& e) {ROS_ERROR("cv_bridge exception: %s", e.what());exit(-1);}

				//							cv_bridge::CvImagePtr ret_mask_ptr;
				//							try {ret_mask_ptr = cv_bridge::toCvCopy(sresult.retrieved_masks[i].images[j], sensor_msgs::image_encodings::MONO8);}
				//							catch (cv_bridge::Exception& e) {ROS_ERROR("cv_bridge exception: %s", e.what());exit(-1);}

				//							cv_bridge::CvImagePtr ret_depth_ptr;
				//							try {ret_depth_ptr = cv_bridge::toCvCopy(sresult.retrieved_depths[i].images[j], sensor_msgs::image_encodings::MONO16);}
				//							catch (cv_bridge::Exception& e) {ROS_ERROR("cv_bridge exception: %s", e.what());exit(-1);}

				//							cv::Mat rgbimage	= ret_image_ptr->image;
				//							cv::Mat maskimage	= ret_mask_ptr->image;
				//							cv::Mat depthimage	= ret_depth_ptr->image;

				//							cv::Mat erosion_dst;
				//							cv::erode( maskimage, erosion_dst, element );

				//							unsigned short * depthdata = (unsigned short * )depthimage.data;
				//							unsigned char * rgbdata = (unsigned char * )rgbimage.data;
				//							unsigned char * maskdata = (unsigned char * )erosion_dst.data;
				//							int count = 0;
				//							for(int pixel = 0; pixel < depthimage.rows*depthimage.cols;pixel++){
				//								if(depthdata[pixel] > 0 && maskdata[pixel] > 0){
				//									count++;
				//								}
				//							}
				//							if(count > maxval){
				//								maxval = count;
				//								maxind = j;
				//							}
				//						}

				//						//for(unsigned int j = 0; j < 1 && j < sresult.retrieved_images[i].images.size(); j++){
				//						if(maxval > 100){//Atleast some pixels has to be in the mask...
				//							std::string key = sresult.retrieved_image_paths[i].strings[maxind];
				//							printf("searchresult key:%s\n",key.c_str());
				//							if(framekeys.count(key) == 0){

				//								int j = maxind;

				//								cv_bridge::CvImagePtr ret_image_ptr;
				//								try {ret_image_ptr = cv_bridge::toCvCopy(sresult.retrieved_images[i].images[j], sensor_msgs::image_encodings::BGR8);}
				//								catch (cv_bridge::Exception& e) {ROS_ERROR("cv_bridge exception: %s", e.what());exit(-1);}

				//								cv_bridge::CvImagePtr ret_mask_ptr;
				//								try {ret_mask_ptr = cv_bridge::toCvCopy(sresult.retrieved_masks[i].images[j], sensor_msgs::image_encodings::MONO8);}
				//								catch (cv_bridge::Exception& e) {ROS_ERROR("cv_bridge exception: %s", e.what());exit(-1);}

				//								cv_bridge::CvImagePtr ret_depth_ptr;
				//								try {ret_depth_ptr = cv_bridge::toCvCopy(sresult.retrieved_depths[i].images[j], sensor_msgs::image_encodings::MONO16);}
				//								catch (cv_bridge::Exception& e) {ROS_ERROR("cv_bridge exception: %s", e.what());exit(-1);}

				//								cv::Mat rgbimage	= ret_image_ptr->image;
				//								cv::Mat maskimage	= ret_mask_ptr->image;
				//								cv::Mat depthimage	= ret_depth_ptr->image;
				//								for (int ii = 0; ii < depthimage.rows; ++ii) {
				//									for (int jj = 0; jj < depthimage.cols; ++jj) {
				//										depthimage.at<uint16_t>(ii, jj) *= 5;
				//									}
				//								}


				//								cv::Mat erosion_dst;
				//								cv::erode( maskimage, erosion_dst, element );

				//								//								cv::Mat overlap	= rgbimage.clone();
				//								//								unsigned short * depthdata = (unsigned short * )depthimage.data;
				//								//								unsigned char * rgbdata = (unsigned char * )rgbimage.data;
				//								//								unsigned char * maskdata = (unsigned char * )maskimage.data;
				//								//								unsigned char * overlapdata = (unsigned char * )overlap.data;
				//								//								for(int pixel = 0; pixel < depthimage.rows*depthimage.cols;pixel++){
				//								//									if(depthdata[pixel] > 0 && maskdata[pixel] > 0){
				//								//										overlapdata[3*pixel+0] = rgbdata[3*pixel+0];
				//								//										overlapdata[3*pixel+1] = rgbdata[3*pixel+1];
				//								//										overlapdata[3*pixel+2] = rgbdata[3*pixel+2];
				//								//									}else{
				//								//										overlapdata[3*pixel+0] = 0;
				//								//										overlapdata[3*pixel+1] = 0;
				//								//										overlapdata[3*pixel+2] = 0;
				//								//									}
				//								//								}
				//								//																	cv::namedWindow( "rgbimage", cv::WINDOW_AUTOSIZE );			cv::imshow( "rgbimage", rgbimage );
				//								//																	cv::namedWindow( "maskimage", cv::WINDOW_AUTOSIZE );		cv::imshow( "maskimage", maskimage );
				//								//																	cv::namedWindow( "depthimage", cv::WINDOW_AUTOSIZE );		cv::imshow( "depthimage", 100*depthimage );
				//								//																	cv::namedWindow( "overlap", cv::WINDOW_AUTOSIZE );			cv::imshow( "overlap", overlap );

				//								Eigen::Affine3d epose = Eigen::Affine3d::Identity();
				//								reglib::RGBDFrame * frame = new reglib::RGBDFrame(cameras[0],rgbimage, depthimage, 0, epose.matrix());
				//								frame->keyval = key;

				//								//                                    cv::namedWindow( "maskimage", cv::WINDOW_AUTOSIZE );			cv::imshow( "maskimage", maskimage );
				//								//                                    frame->show(true);
				//								//reglib::Model * searchmodel = new reglib::Model(frame,maskimage);
				//								reglib::Model * searchmodel = new reglib::Model(frame,erosion_dst);
				//								bool res = searchmodel->testFrame(0);

				//								reglib::Model * searchmodelHolder = new reglib::Model();
				//								searchmodelHolder->submodels.push_back(searchmodel);
				//								searchmodelHolder->submodels_relativeposes.push_back(Eigen::Matrix4d::Identity());
				//								searchmodelHolder->last_changed = ++current_model_update;
				//								searchmodelHolder->recomputeModelPoints();

				//								//searchmodelHolder->showHistory(viewer);


				//								//								if(cv::waitKey(0) != 'n'){
				//								//Todo:: check new model is not flat or L shape

				//								printf("--- trying to add serach results, if more then one addToDB: results added-----\n");
				//								//addToDB(modeldatabase, searchmodel,false,true);
				//								addToDB(modeldatabase, searchmodelHolder,false,true);
				//								show_sorted();
				//							}
				//						}

//			printf("---------------ids in searchresult: %i----------------\n",result.vocabulary_ids.size());
//			for(unsigned int i = 0; i < result.vocabulary_ids.size(); i++){
//				int ind = result.vocabulary_ids[i];
//				printf("found object with ind: %i\n",ind);
//				if(vocabulary_ids.count(ind) == 0){	printf("does not exist in db, continue\n"); continue; }
//				reglib::Model * search_model = vocabulary_ids[ind];
//				if(search_model != model){printf("adding to search results\n"); ret.push_back(search_model);}
//				if(ret.size() == number_of_matches){printf("found enough search results\n"); return ret;}
//			}
		}else{
			ROS_ERROR("retrieval_client service FAIL in %s :: %i !",__FILE__,__LINE__);
		}
	}else{
		ROS_ERROR("model_to_retrieval_query service FAIL in %s :: %i !",__FILE__,__LINE__);
	}
	return true;
}

std::set<int> searchset;

void addNewModel(reglib::Model * model){
	reglib::RegistrationRandom *	reg	= new reglib::RegistrationRandom();
	reglib::ModelUpdaterBasicFuse * mu	= new reglib::ModelUpdaterBasicFuse( model, reg);
	mu->occlusion_penalty               = occlusion_penalty;
	mu->massreg_timeout                 = massreg_timeout;
	mu->viewer							= viewer;

	mu->show_init_lvl					= show_init_lvl;//init show
	mu->show_refine_lvl					= show_refine_lvl;//refine show
	mu->show_scoring					= show_scoring;//fuse scoring show
	reg->visualizationLvl				= show_reg_lvl;

	mu->makeInitialSetup();


	delete mu;
	delete reg;


	reglib::Model * newmodelHolder = new reglib::Model();
	newmodelHolder->submodels.push_back(model);
	newmodelHolder->submodels_relativeposes.push_back(Eigen::Matrix4d::Identity());
	newmodelHolder->last_changed = ++current_model_update;
	if(show_modelbuild){
		newmodelHolder->recomputeModelPoints(Eigen::Matrix4d::Identity(),viewer);
	}else{
		newmodelHolder->recomputeModelPoints();
	}

	modeldatabase->add(newmodelHolder);
	addToDB(modeldatabase, newmodelHolder,false);
	show_sorted();

	bool do_next = true;
	while(do_next && run_search){
		printf("running search loop\n");
		do_next = false;
		for(unsigned int i = 0; i < modeldatabase->models.size(); i++){
			reglib::Model * current = modeldatabase->models[i];
			if(searchset.count(current->id)==0){
				searchset.insert(current->id);
				printf("new search %i\n",current->id);


				if(runSearch(modeldatabase, current)){
					do_next = true;
					break;
				}
			}else{
				printf("already searched %i\n",current->id);
			}
		}
	}

	for(unsigned int i = 0; i < modeldatabase->models.size(); i++){publish_history(modeldatabase->models[i]->getHistory());}
	publishDatabasePCD();
	dumpDatabase(savepath);
}


void somaCallback(const std_msgs::String & m){
	printf("somaCallback(%s)\n",m.data.c_str());
//	quasimodo_msgs::model mod = m;
//	addNewModel(quasimodo_brain::getModelFromMSG(mod,false));
}

void modelCallback(const quasimodo_msgs::model & m){
	quasimodo_msgs::model mod = m;
	addNewModel(quasimodo_brain::getModelFromMSG(mod,false));
}

bool nextNew = true;
reglib::Model * newmodel = 0;

//std::vector<cv::Mat> newmasks;
std::vector<cv::Mat> allmasks;

int modaddcount = 0;
bool modelFromFrame(quasimodo_msgs::model_from_frame::Request  & req, quasimodo_msgs::model_from_frame::Response & res){
	//printf("======================================\nmodelFromFrame\n======================================\n");
	uint64 frame_id = req.frame_id;
	uint64 isnewmodel = req.isnewmodel;

	printf("modelFromFrame %i and %i\n",int(frame_id),int(isnewmodel));

	cv_bridge::CvImagePtr			mask_ptr;
	try{							mask_ptr = cv_bridge::toCvCopy(req.mask, sensor_msgs::image_encodings::MONO8);}
	catch (cv_bridge::Exception& e){ROS_ERROR("cv_bridge exception: %s", e.what());return false;}

	cv::Mat mask					= mask_ptr->image;
	allmasks.push_back(mask);
	//std::cout << frames[frame_id]->pose << std::endl << std::endl;

	if(newmodel == 0){
		newmodel					= new reglib::Model(frames[frame_id],mask);
		//modeldatabase->add(newmodel);
	}else{
		newmodel->frames.push_back(frames[frame_id]);
		newmodel->relativeposes.push_back(newmodel->frames.front()->pose.inverse() * frames[frame_id]->pose);
		newmodel->modelmasks.push_back(new reglib::ModelMask(mask));
		newmodel->recomputeModelPoints();
	}
	newmodel->modelmasks.back()->sweepid = sweepid_counter;

	res.model_id					= newmodel->id;
	if(isnewmodel != 0){
		reglib::RGBDFrame * frontframe = newmodel->frames.front();
		int current_model_update_before = current_model_update;
		newmodel->recomputeModelPoints();


		reglib::RegistrationRandom *	reg	= new reglib::RegistrationRandom();
		reglib::ModelUpdaterBasicFuse * mu	= new reglib::ModelUpdaterBasicFuse( newmodel, reg);
		mu->occlusion_penalty               = occlusion_penalty;
		mu->massreg_timeout                 = massreg_timeout;
		mu->viewer							= viewer;

		mu->show_init_lvl = show_init_lvl;//init show
		mu->show_refine_lvl = show_refine_lvl;//refine show
		mu->show_scoring = show_scoring;//fuse scoring show
		reg->visualizationLvl				= show_reg_lvl;

		//newmodel->print();
		mu->makeInitialSetup();
		//newmodel->print();
		delete mu;
		delete reg;

		reglib::Model * newmodelHolder = new reglib::Model();
		newmodelHolder->submodels.push_back(newmodel);
		newmodelHolder->submodels_relativeposes.push_back(Eigen::Matrix4d::Identity());
		newmodelHolder->last_changed = ++current_model_update;
		newmodelHolder->recomputeModelPoints();

		modeldatabase->add(newmodelHolder);
		addToDB(modeldatabase, newmodelHolder,false);
		for(unsigned int i = 0; i < modeldatabase->models.size(); i++){
			publish_history(modeldatabase->models[i]->getHistory());
		}
		/*
		for(unsigned int i = 0; i < modeldatabase->models.size(); i++){
			publish_history(modeldatabase->models[i]->getHistory());
		}*/
		show_sorted();
		publishDatabasePCD();
		dumpDatabase(savepath);

		//exit(0);


		/*
//SIMPLE TEST
reglib::Model * testmodel = new reglib::Model();
for(int i = 0; i < newmodel->frames.size();i++){
	printf("%i / %i\n",i+1,newmodel->frames.size());
	reglib::Model * tmp = new reglib::Model();
	tmp->frames.push_back(newmodel->frames[i]);
	tmp->modelmasks.push_back(newmodel->modelmasks[i]);
	tmp->relativeposes.push_back(Eigen::Matrix4d::Identity());
	tmp->recomputeModelPoints();

	std::vector<Eigen::Matrix4d> cp;
	std::vector<reglib::RGBDFrame*> cf;
	std::vector<reglib::ModelMask*> cm;
	for(unsigned int i = 0; i < tmp->relativeposes.size(); i++){
		cp.push_back(tmp->relativeposes[i]);
		cf.push_back(tmp->frames[i]);
		cm.push_back(tmp->modelmasks[i]);
	}
	mu->getGoodCompareFrames(cp,cf,cm);
	tmp->rep_relativeposes = cp;
	tmp->rep_frames = cf;
	tmp->rep_modelmasks = cm;

//	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud = tmp->getPCLnormalcloud(1,false);
//	viewer->removeAllPointClouds();
//	viewer->addPointCloud<pcl::PointXYZRGBNormal> (cloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(cloud), "cloud");
//	viewer->spin();

	testmodel->submodels.push_back(tmp);
	testmodel->submodels_relativeposes.push_back(newmodel->relativeposes[i]);
}

testmodel->recomputeModelPoints();
{
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud = testmodel->getPCLnormalcloud(1,false);
	viewer->removeAllPointClouds();
	viewer->addPointCloud<pcl::PointXYZRGBNormal> (cloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(cloud), "cloud");
	viewer->spin();
}

std::vector<reglib::Model *> testmodels;
std::vector<Eigen::Matrix4d> testrps;
mu->addModelsToVector(testmodels,testrps,testmodel,Eigen::Matrix4d::Identity());

printf("testmodels: %i\n",testmodels.size());
std::vector<std::vector < reglib::OcclusionScore > > testocs = mu->computeOcclusionScore(testmodels,testrps);
std::vector<std::vector < float > > scores = mu->getScores(testocs);
std::vector<int> partition = mu->getPartition(scores,2,5,2);

for(unsigned int i = 0; i < scores.size(); i++){
	for(unsigned int j = 0; j < scores.size(); j++){
		if(scores[i][j] > 0){printf(" ");}
		printf("%5.5f ",0.0001*scores[i][j]);
	}
	printf("\n");
}

printf("partition");
for(unsigned int i = 0; i < partition.size(); i++){printf("%i ", partition[i]);}
printf("\n");
*/

		//exit(0);
		/*
vector<Model *> models;
vector<Matrix4d> rps;

//			models.push_back(model);
//			models.push_back(model2);
//			rps.push_back(Eigen::Matrix4d::Identity());
//			rps.push_back(pose);

addModelsToVector(models,rps,model,Eigen::Matrix4d::Identity());
unsigned int nr_models1 = models.size();
addModelsToVector(models,rps,model2,pose);
vector<vector < OcclusionScore > > ocs = computeOcclusionScore(models,rps);
*/
		/*

		//newmodel->recomputeModelPoints();


		newmodel->last_changed = ++current_model_update;
		newmodel->print();
//exit(0);
		addToDB(modeldatabase, newmodel,false);
		//if(modaddcount % 1 == 0){show_sorted();}
		//show_sorted();
		publishDatabasePCD();
		modaddcount++;
		//dumpDatabase(savepath);
*/

		for(unsigned int m = 0; run_search && m < modeldatabase->models.size(); m++){
			printf("looking at: %i\n",int(modeldatabase->models[m]->last_changed));
			reglib::Model * currentTest = modeldatabase->models[m];
			if(currentTest->last_changed > current_model_update_before){
				printf("changed: %i\n",int(m));

				double start = getTime();

				new_search_result = false;
				models_new_pub.publish(getModelMSG(currentTest));

				while(getTime()-start < search_timeout){
					ros::spinOnce();
					if(new_search_result){

						for(unsigned int i = 0; i < sresult.retrieved_images.size(); i++){

							int maxval = 0;
							int maxind = 0;

							int dilation_size = 0;
							cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
																		cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
																		cv::Point( dilation_size, dilation_size ) );
							for(unsigned int j = 0; j < sresult.retrieved_images[i].images.size(); j++){
								//framekeys

								cv_bridge::CvImagePtr ret_image_ptr;
								try {ret_image_ptr = cv_bridge::toCvCopy(sresult.retrieved_images[i].images[j], sensor_msgs::image_encodings::BGR8);}
								catch (cv_bridge::Exception& e) {ROS_ERROR("cv_bridge exception: %s", e.what());exit(-1);}

								cv_bridge::CvImagePtr ret_mask_ptr;
								try {ret_mask_ptr = cv_bridge::toCvCopy(sresult.retrieved_masks[i].images[j], sensor_msgs::image_encodings::MONO8);}
								catch (cv_bridge::Exception& e) {ROS_ERROR("cv_bridge exception: %s", e.what());exit(-1);}

								cv_bridge::CvImagePtr ret_depth_ptr;
								try {ret_depth_ptr = cv_bridge::toCvCopy(sresult.retrieved_depths[i].images[j], sensor_msgs::image_encodings::MONO16);}
								catch (cv_bridge::Exception& e) {ROS_ERROR("cv_bridge exception: %s", e.what());exit(-1);}

								cv::Mat rgbimage	= ret_image_ptr->image;
								cv::Mat maskimage	= ret_mask_ptr->image;
								cv::Mat depthimage	= ret_depth_ptr->image;

								cv::Mat erosion_dst;
								cv::erode( maskimage, erosion_dst, element );

								unsigned short * depthdata = (unsigned short * )depthimage.data;
								unsigned char * rgbdata = (unsigned char * )rgbimage.data;
								unsigned char * maskdata = (unsigned char * )erosion_dst.data;
								int count = 0;
								for(int pixel = 0; pixel < depthimage.rows*depthimage.cols;pixel++){
									if(depthdata[pixel] > 0 && maskdata[pixel] > 0){
										count++;
									}
								}
								if(count > maxval){
									maxval = count;
									maxind = j;
								}
							}

							//for(unsigned int j = 0; j < 1 && j < sresult.retrieved_images[i].images.size(); j++){
							if(maxval > 100){//Atleast some pixels has to be in the mask...
								std::string key = sresult.retrieved_image_paths[i].strings[maxind];
								printf("searchresult key:%s\n",key.c_str());
								if(framekeys.count(key) == 0){

									int j = maxind;

									cv_bridge::CvImagePtr ret_image_ptr;
									try {ret_image_ptr = cv_bridge::toCvCopy(sresult.retrieved_images[i].images[j], sensor_msgs::image_encodings::BGR8);}
									catch (cv_bridge::Exception& e) {ROS_ERROR("cv_bridge exception: %s", e.what());exit(-1);}

									cv_bridge::CvImagePtr ret_mask_ptr;
									try {ret_mask_ptr = cv_bridge::toCvCopy(sresult.retrieved_masks[i].images[j], sensor_msgs::image_encodings::MONO8);}
									catch (cv_bridge::Exception& e) {ROS_ERROR("cv_bridge exception: %s", e.what());exit(-1);}

									cv_bridge::CvImagePtr ret_depth_ptr;
									try {ret_depth_ptr = cv_bridge::toCvCopy(sresult.retrieved_depths[i].images[j], sensor_msgs::image_encodings::MONO16);}
									catch (cv_bridge::Exception& e) {ROS_ERROR("cv_bridge exception: %s", e.what());exit(-1);}

									cv::Mat rgbimage	= ret_image_ptr->image;
									cv::Mat maskimage	= ret_mask_ptr->image;
									cv::Mat depthimage	= ret_depth_ptr->image;
									for (int ii = 0; ii < depthimage.rows; ++ii) {
										for (int jj = 0; jj < depthimage.cols; ++jj) {
											depthimage.at<uint16_t>(ii, jj) *= 5;
										}
									}


									cv::Mat erosion_dst;
									cv::erode( maskimage, erosion_dst, element );

									//								cv::Mat overlap	= rgbimage.clone();
									//								unsigned short * depthdata = (unsigned short * )depthimage.data;
									//								unsigned char * rgbdata = (unsigned char * )rgbimage.data;
									//								unsigned char * maskdata = (unsigned char * )maskimage.data;
									//								unsigned char * overlapdata = (unsigned char * )overlap.data;
									//								for(int pixel = 0; pixel < depthimage.rows*depthimage.cols;pixel++){
									//									if(depthdata[pixel] > 0 && maskdata[pixel] > 0){
									//										overlapdata[3*pixel+0] = rgbdata[3*pixel+0];
									//										overlapdata[3*pixel+1] = rgbdata[3*pixel+1];
									//										overlapdata[3*pixel+2] = rgbdata[3*pixel+2];
									//									}else{
									//										overlapdata[3*pixel+0] = 0;
									//										overlapdata[3*pixel+1] = 0;
									//										overlapdata[3*pixel+2] = 0;
									//									}
									//								}
									//																	cv::namedWindow( "rgbimage", cv::WINDOW_AUTOSIZE );			cv::imshow( "rgbimage", rgbimage );
									//																	cv::namedWindow( "maskimage", cv::WINDOW_AUTOSIZE );		cv::imshow( "maskimage", maskimage );
									//																	cv::namedWindow( "depthimage", cv::WINDOW_AUTOSIZE );		cv::imshow( "depthimage", 100*depthimage );
									//																	cv::namedWindow( "overlap", cv::WINDOW_AUTOSIZE );			cv::imshow( "overlap", overlap );

									Eigen::Affine3d epose = Eigen::Affine3d::Identity();
									reglib::RGBDFrame * frame = new reglib::RGBDFrame(cameras[0],rgbimage, depthimage, 0, epose.matrix());
									frame->keyval = key;

									//                                    cv::namedWindow( "maskimage", cv::WINDOW_AUTOSIZE );			cv::imshow( "maskimage", maskimage );
									//                                    frame->show(true);
									//reglib::Model * searchmodel = new reglib::Model(frame,maskimage);
									reglib::Model * searchmodel = new reglib::Model(frame,erosion_dst);
									bool res = searchmodel->testFrame(0);

									reglib::Model * searchmodelHolder = new reglib::Model();
									searchmodelHolder->submodels.push_back(searchmodel);
									searchmodelHolder->submodels_relativeposes.push_back(Eigen::Matrix4d::Identity());
									searchmodelHolder->last_changed = ++current_model_update;
									searchmodelHolder->recomputeModelPoints();

									//searchmodelHolder->showHistory(viewer);


									//								if(cv::waitKey(0) != 'n'){
									//Todo:: check new model is not flat or L shape

									printf("--- trying to add serach results, if more then one addToDB: results added-----\n");
									//addToDB(modeldatabase, searchmodel,false,true);
									//addToDB(modeldatabase, searchmodelHolder,false,true);
									show_sorted();
								}
							}


							/*
							for(unsigned int j = 0; j < sresult.retrieved_images[i].images.size(); j++){
								//framekeys

								cv_bridge::CvImagePtr ret_image_ptr;
								try {ret_image_ptr = cv_bridge::toCvCopy(sresult.retrieved_images[i].images[j], sensor_msgs::image_encodings::BGR8);}
								catch (cv_bridge::Exception& e) {ROS_ERROR("cv_bridge exception: %s", e.what());exit(-1);}

								cv_bridge::CvImagePtr ret_mask_ptr;
								try {ret_mask_ptr = cv_bridge::toCvCopy(sresult.retrieved_masks[i].images[j], sensor_msgs::image_encodings::MONO8);}
								catch (cv_bridge::Exception& e) {ROS_ERROR("cv_bridge exception: %s", e.what());exit(-1);}

								cv_bridge::CvImagePtr ret_depth_ptr;
								try {ret_depth_ptr = cv_bridge::toCvCopy(sresult.retrieved_depths[i].images[j], sensor_msgs::image_encodings::MONO16);}
								catch (cv_bridge::Exception& e) {ROS_ERROR("cv_bridge exception: %s", e.what());exit(-1);}

								cv::Mat rgbimage	= ret_image_ptr->image;
								cv::Mat maskimage	= ret_mask_ptr->image;
								cv::Mat depthimage	= ret_depth_ptr->image;

								cv::Mat erosion_dst;
								cv::erode( maskimage, erosion_dst, element );

								unsigned short * depthdata = (unsigned short * )depthimage.data;
								unsigned char * rgbdata = (unsigned char * )rgbimage.data;
								unsigned char * maskdata = (unsigned char * )erosion_dst.data;
								int count = 0;
								for(int pixel = 0; pixel < depthimage.rows*depthimage.cols;pixel++){
									if(depthdata[pixel] > 0 && maskdata[pixel] > 0){
										count++;
									}
								}
								if(count > maxval){
									maxval = count;
									maxind = j;
								}
							}

							//for(unsigned int j = 0; j < 1 && j < sresult.retrieved_images[i].images.size(); j++){
							if(maxval > 100){//Atleast some pixels has to be in the mask...
								std::string key = sresult.retrieved_image_paths[i].strings[maxind];
								printf("searchresult key:%s\n",key.c_str());
								if(framekeys.count(key) == 0){

									int j = maxind;

									cv_bridge::CvImagePtr ret_image_ptr;
									try {ret_image_ptr = cv_bridge::toCvCopy(sresult.retrieved_images[i].images[j], sensor_msgs::image_encodings::BGR8);}
									catch (cv_bridge::Exception& e) {ROS_ERROR("cv_bridge exception: %s", e.what());exit(-1);}

									cv_bridge::CvImagePtr ret_mask_ptr;
									try {ret_mask_ptr = cv_bridge::toCvCopy(sresult.retrieved_masks[i].images[j], sensor_msgs::image_encodings::MONO8);}
									catch (cv_bridge::Exception& e) {ROS_ERROR("cv_bridge exception: %s", e.what());exit(-1);}

									cv_bridge::CvImagePtr ret_depth_ptr;
									try {ret_depth_ptr = cv_bridge::toCvCopy(sresult.retrieved_depths[i].images[j], sensor_msgs::image_encodings::MONO16);}
									catch (cv_bridge::Exception& e) {ROS_ERROR("cv_bridge exception: %s", e.what());exit(-1);}

									cv::Mat rgbimage	= ret_image_ptr->image;
									cv::Mat maskimage	= ret_mask_ptr->image;
									cv::Mat depthimage	= ret_depth_ptr->image;
									for (int ii = 0; ii < depthimage.rows; ++ii) {
										for (int jj = 0; jj < depthimage.cols; ++jj) {
											depthimage.at<uint16_t>(ii, jj) *= 5;
										}
									}


									cv::Mat erosion_dst;
									cv::erode( maskimage, erosion_dst, element );

									//								cv::Mat overlap	= rgbimage.clone();
									//								unsigned short * depthdata = (unsigned short * )depthimage.data;
									//								unsigned char * rgbdata = (unsigned char * )rgbimage.data;
									//								unsigned char * maskdata = (unsigned char * )maskimage.data;
									//								unsigned char * overlapdata = (unsigned char * )overlap.data;
									//								for(int pixel = 0; pixel < depthimage.rows*depthimage.cols;pixel++){
									//									if(depthdata[pixel] > 0 && maskdata[pixel] > 0){
									//										overlapdata[3*pixel+0] = rgbdata[3*pixel+0];
									//										overlapdata[3*pixel+1] = rgbdata[3*pixel+1];
									//										overlapdata[3*pixel+2] = rgbdata[3*pixel+2];
									//									}else{
									//										overlapdata[3*pixel+0] = 0;
									//										overlapdata[3*pixel+1] = 0;
									//										overlapdata[3*pixel+2] = 0;
									//									}
									//								}
									//								cv::namedWindow( "rgbimage", cv::WINDOW_AUTOSIZE );			cv::imshow( "rgbimage", rgbimage );
									//								cv::namedWindow( "maskimage", cv::WINDOW_AUTOSIZE );		cv::imshow( "maskimage", maskimage );
									//								cv::namedWindow( "depthimage", cv::WINDOW_AUTOSIZE );		cv::imshow( "depthimage", 100*depthimage );
									//								cv::namedWindow( "overlap", cv::WINDOW_AUTOSIZE );			cv::imshow( "overlap", overlap );

									Eigen::Affine3d epose = Eigen::Affine3d::Identity();
									reglib::RGBDFrame * frame = new reglib::RGBDFrame(cameras[0],rgbimage, depthimage, 0, epose.matrix());
									frame->keyval = key;
									frame->show(true);
									//reglib::Model * searchmodel = new reglib::Model(frame,maskimage);
									reglib::Model * searchmodel = new reglib::Model(frame,erosion_dst);
									bool res = searchmodel->testFrame(0);

									reglib::Model * searchmodelHolder = new reglib::Model();
									searchmodelHolder->submodels.push_back(searchmodel);
									searchmodelHolder->submodels_relativeposes.push_back(Eigen::Matrix4d::Identity());
									searchmodelHolder->last_changed = ++current_model_update;
									searchmodelHolder->recomputeModelPoints();


									//								if(cv::waitKey(0) != 'n'){
									//Todo:: check new model is not flat or L shape

									printf("--- trying to add serach results, if more then one addToDB: results added-----\n");
									//addToDB(modeldatabase, searchmodel,false,true);
									addToDB(modeldatabase, searchmodelHolder,false,true);
									show_sorted();
									//								}
								}

							}
							*/
						}

						break;
					}else{
						printf("searching... timeout in %3.3f\n", +start +search_timeout - getTime());
						usleep(100000);
					}
				}
			}
			//exit(0);
		}

		for(unsigned int i = 0; i < modeldatabase->models.size(); i++){
			for(unsigned int j = 0; j < modeldatabase->models[i]->submodels.size(); j++){
				if( modeldatabase->models[i]->submodels[j]->frames.front() == frontframe){
					Eigen::Matrix4d pose = frontframe->pose * modeldatabase->models[i]->submodels_relativeposes[j].inverse();

					pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = modeldatabase->models[i]->submodels[j]->getPCLcloud(1,true);
					pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZRGB> ());
					pcl::transformPointCloud (*cloud, *transformed_cloud, pose);


					sensor_msgs::PointCloud2 input;
					pcl::toROSMsg (*transformed_cloud,input);//, *transformed_cloud);
					input.header.frame_id = "/map";
					model_last_pub.publish(input);
				}
			}
		}

		newmodel = 0;
		sweepid_counter++;
	}
	return true;
}

//bool fuseModels(quasimodo_msgs::fuse_models::Request  & req, quasimodo_msgs::fuse_models::Response & res){}

int getdir (std::string dir, std::vector<std::string> & files){
	DIR *dp;
	struct dirent *dirp;
	if((dp  = opendir(dir.c_str())) == NULL) {
		cout << "Error(" << errno << ") opening " << dir << endl;
		return errno;
	}

	while ((dirp = readdir(dp)) != NULL) {
		files.push_back(std::string(dirp->d_name));
	}
	closedir(dp);
	return 0;
}

void clearMem(){

	for(auto iterator = cameras.begin(); iterator != cameras.end(); iterator++) {
		delete iterator->second;
	}

	for(auto iterator = frames.begin(); iterator != frames.end(); iterator++) {
		delete iterator->second;
	}

	for(auto iterator = models.begin(); iterator != models.end(); iterator++) {
		reglib::Model * model = iterator->second;
		for(unsigned int i = 0; i < model->frames.size(); i++){
			delete model->frames[i];
		}

		for(unsigned int i = 0; i < model->modelmasks.size(); i++){
			delete model->modelmasks[i];
		}
		delete model;
	}

	for(auto iterator = updaters.begin(); iterator != updaters.end(); iterator++) {
		delete iterator->second;
	}


	if(registration != 0){delete registration;}
	if(modeldatabase != 0){delete modeldatabase;}
}

int main(int argc, char **argv){
	cameras[0]		= new reglib::Camera();
	registration	= new reglib::RegistrationRandom();
	modeldatabase	= 0;//new ModelDatabaseRGBHistogram(5);
	//new ModelDatabaseRetrieval(n);

	ros::init(argc, argv, "quasimodo_model_server");
	ros::NodeHandle n;
	models_new_pub		= n.advertise<quasimodo_msgs::model>("/models/new",		1000);
	models_updated_pub	= n.advertise<quasimodo_msgs::model>("/models/updated", 1000);
	models_deleted_pub	= n.advertise<quasimodo_msgs::model>("/models/deleted", 1000);

	ros::ServiceServer service1 = n.advertiseService("model_from_frame", modelFromFrame);
	ROS_INFO("Ready to add use model_from_frame.");

	ros::ServiceServer service2 = n.advertiseService("index_frame", indexFrame);
	ROS_INFO("Ready to add use index_frame.");

	//ros::ServiceServer service3 = n.advertiseService("fuse_models", fuseModels);
	//ROS_INFO("Ready to add use fuse_models.");

	ros::ServiceServer service4 = n.advertiseService("get_model", getModel);
	ROS_INFO("Ready to add use get_model.");

//<<<<<<< HEAD
//	soma2add = n.serviceClient<soma_manager::SOMA2InsertObjs>("soma2/insert_objs");
//	ROS_INFO("Ready to add use soma2add.");
//=======
//	soma_add = n.serviceClient<soma_manager::SOMAInsertObjs>("soma/insert_objs");
//	ROS_INFO("Ready to add use soma_add.");
//>>>>>>> integrate_object_search

	ros::Subscriber sub = n.subscribe("/retrieval_result", 1, retrievalCallback);
	ROS_INFO("Ready to add recieve search results.");

	database_pcd_pub    = n.advertise<sensor_msgs::PointCloud2>("modelserver/databasepcd", 1000);
	model_history_pub   = n.advertise<sensor_msgs::PointCloud2>("modelserver/model_history", 1000);
	model_last_pub      = n.advertise<sensor_msgs::PointCloud2>("modelserver/last", 1000);
	model_places_pub    = n.advertise<sensor_msgs::PointCloud2>("modelserver/model_places", 1000);

	std::string retrieval_name	= "/quasimodo_retrieval_service";
	std::string conversion_name = "/models/server";
	std::string insert_name		= "/insert_model_service";
	retrieval_client	= n.serviceClient<quasimodo_msgs::query_cloud>				(retrieval_name);
	conversion_client	= n.serviceClient<quasimodo_msgs::model_to_retrieval_query>	(conversion_name);
	insert_client		= n.serviceClient<quasimodo_msgs::insert_model>				(insert_name);


	std::vector<ros::Subscriber> input_model_subs;
	std::vector<ros::Subscriber> soma_input_model_subs;
	std::vector<std::string> modelpcds;

	int inputstate = -1;
	for(int i = 1; i < argc;i++){
		printf("input: %s\n",argv[i]);
		if(		std::string(argv[i]).compare("-c") == 0){	printf("camera input state\n"); inputstate = 1;}
		else if(std::string(argv[i]).compare("-l") == 0){	printf("reading all models at %s (the path defined from -p)\n",savepath.c_str());
			std::vector<std::string> folderdata;
			int r = getdir(savepath+"/",folderdata);
			for(unsigned int fnr = 0; fnr < folderdata.size(); fnr++){
				printf("%s\n",folderdata[fnr].c_str());
			}
			exit(0);}
		else if(std::string(argv[i]).compare("-m") == 0){	printf("model input state\n");	inputstate = 2;}
		else if(std::string(argv[i]).compare("-p") == 0){	printf("path input state\n");	inputstate = 3;}
		else if(std::string(argv[i]).compare("-occlusion_penalty") == 0){printf("occlusion_penalty input state\n");inputstate = 4;}
		else if(std::string(argv[i]).compare("-massreg_timeout") == 0){printf("massreg_timeout input state\n");inputstate = 5;}
		else if(std::string(argv[i]).compare("-search") == 0){printf("pointcloud search input state\n");run_search = true; inputstate = 6;}
		else if(std::string(argv[i]).compare("-v") == 0){           printf("visualization turned on\n");                visualization = true;}
		else if(std::string(argv[i]).compare("-v_init") == 0){      printf("visualization of init turned on\n");        visualization = true; inputstate = 8;}
		else if(std::string(argv[i]).compare("-v_refine") == 0 || std::string(argv[i]).compare("-v_ref") == 0){	printf("visualization refinement turned on\n");     visualization = true; inputstate = 9;}
		else if(std::string(argv[i]).compare("-v_register") == 0 || std::string(argv[i]).compare("-v_reg") == 0){	printf("visualization registration turned on\n");	visualization = true; inputstate = 10;}
		else if(std::string(argv[i]).compare("-v_scoring") == 0 || std::string(argv[i]).compare("-v_score") == 0 || std::string(argv[i]).compare("-v_sco") == 0){	printf("visualization scoring turned on\n");        visualization = true; show_scoring = true;}
		else if(std::string(argv[i]).compare("-v_db") == 0){        printf("visualization db turned on\n");             visualization = true; show_db = true;}
		else if(std::string(argv[i]).compare("-s_db") == 0){        printf("save db turned on\n"); save_db = true;}
		else if(std::string(argv[i]).compare("-intopic") == 0){	printf("intopic input state\n");	inputstate = 11;}
		else if(std::string(argv[i]).compare("-mdb") == 0){	printf("intopic input state\n");		inputstate = 12;}
		else if(std::string(argv[i]).compare("-show_search") == 0){	printf("show_search\n");		show_search = true;}
		else if(std::string(argv[i]).compare("-show_modelbuild") == 0){	printf("show_modelbuild\n");	visualization = true; show_modelbuild = true;}
		else if(std::string(argv[i]).compare("-loadModelsPCDs") == 0){								inputstate = 13;}
		else if(inputstate == 1){
			reglib::Camera * cam = reglib::Camera::load(std::string(argv[i]));
			delete cameras[0];
			cameras[0] = cam;
		}else if(inputstate == 2){
			reglib::Model * model = reglib::Model::load(cameras[0],std::string(argv[i]));
			sweepid_counter = std::max(int(model->modelmasks[0]->sweepid + 1), sweepid_counter);
			modeldatabase->add(model);
			//addToDB(modeldatabase, model,false);
			model->last_changed = ++current_model_update;
			show_sorted();
		}else if(inputstate == 3){
			savepath = std::string(argv[i]);
		}else if(inputstate == 4){
			occlusion_penalty = atof(argv[i]); printf("occlusion_penalty set to %f\n",occlusion_penalty);
		}else if(inputstate == 5){
			massreg_timeout = atof(argv[i]); printf("massreg_timeout set to %f\n",massreg_timeout);
		}else if(inputstate == 6){
			search_timeout = atof(argv[i]); printf("search_timeout set to %f\n",search_timeout);
			if(search_timeout == 0){
				run_search = false;
			}
		}else if(inputstate == 8){
			show_init_lvl = atoi(argv[i]);
		}else if(inputstate == 9){
			show_refine_lvl = atoi(argv[i]);
		}else if(inputstate == 10){
			show_reg_lvl = atoi(argv[i]);
		}else if(inputstate == 11){
			printf("adding %s to input_model_subs\n",argv[i]);
			input_model_subs.push_back(n.subscribe(std::string(argv[i]), 100, modelCallback));
		}else if(inputstate == 12){
			if(atoi(argv[i]) == 0){
				if(modeldatabase != 0){delete modeldatabase;}
				modeldatabase	= new ModelDatabaseBasic();
			}

			if(atoi(argv[i]) == 1){
				if(modeldatabase != 0){delete modeldatabase;}
				modeldatabase	= new ModelDatabaseRGBHistogram(5);
			}

			if(atoi(argv[i]) == 2){
				if(modeldatabase != 0){delete modeldatabase;}
				modeldatabase	= new ModelDatabaseRetrieval(n);
			}
		}else if(inputstate == 13){
			modelpcds.push_back( std::string(argv[i]) );
		}
	}

	if(modeldatabase == 0){modeldatabase	= new ModelDatabaseRetrieval(n);}

	for(unsigned int i = 0; i < modelpcds.size(); i++){
		std::vector<reglib::Model *> mods = quasimodo_brain::loadModelsPCDs(modelpcds[i]);
		for(unsigned int j = 0; j < mods.size(); j++){
			modeldatabase->add(mods[j]);
		}
	}


	//if(input_model_subs.size() == 0){input_model_subs.push_back(n.subscribe("/model/out/topic", 100, modelCallback));}
	if(input_model_subs.size() == 0){input_model_subs.push_back(n.subscribe("/quasimodo/segmentation/out/model", 100, modelCallback));}



	if(soma_input_model_subs.size() == 0){soma_input_model_subs.push_back(n.subscribe("/quasimodo/segmentation/out/soma_segment_id", 10000, somaCallback));}


	if(visualization){
		viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer ("Modelserver Viewer"));
		viewer->addCoordinateSystem(0.1);
		viewer->setBackgroundColor(1.0,0.0,1.0);
	}

	ros::Duration(1.0).sleep();
	chatter_pub = n.advertise<std_msgs::String>("modelserver", 1000);
	std_msgs::String msg;
	msg.data = "starting";
	chatter_pub.publish(msg);
	ros::spinOnce();
	ros::spin();
	return 0;
}
