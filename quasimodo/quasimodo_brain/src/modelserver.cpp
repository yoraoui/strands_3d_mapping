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
#include "quasimodo_msgs/recognize.h"
#include "soma_msgs/SOMAObject.h"

#include "soma_manager/SOMAInsertObjs.h"

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

bool addToDB(ModelDatabase * database, reglib::Model * model, bool add);
bool addIfPossible(ModelDatabase * database, reglib::Model * model, reglib::Model * model2);
void addNewModel(reglib::Model * model);

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
ros::NodeHandle *						nh;
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

ros::ServiceClient retrieval_client;
ros::ServiceClient conversion_client;
ros::ServiceClient insert_client;

double occlusion_penalty	= 10;
double massreg_timeout		= 120;
bool run_search				= false;
double search_timeout		= 30;
int sweepid_counter			= 0;
int current_model_update	= 0;

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
	printf("%i\n",__LINE__);
	char command [1024];
	sprintf(command,"rm -r -f %s/database_tmp",path.c_str());
	int r = system(command);
	printf("%i\n",__LINE__);

	sprintf(command,"mkdir %s/database_tmp",path.c_str());
	r = system(command);
	printf("%i\n",__LINE__);

	for(unsigned int m = 0; m < modeldatabase->models.size(); m++){
		printf("%i\n",__LINE__);
		char buf [1024];
		sprintf(buf,"%s/database_tmp/model_%08i",path.c_str(),m);
		sprintf(command,"mkdir -p %s",buf);
		printf("%i\n",__LINE__);
		r = system(command);
		printf("%i\n",__LINE__);
		modeldatabase->models[m]->save(std::string(buf));
		printf("%i\n",__LINE__);
	}
	//exit(0);
}

void retrievalCallback(const quasimodo_msgs::retrieval_query_result & qr){
    printf("retrievalCallback\n");
    sresult = qr.result;
    new_search_result = true;
    last_search_result_time = getTime();
}

void showModels(std::vector<reglib::Model *> mods){
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


bool getModel(quasimodo_msgs::get_model::Request  & req, quasimodo_msgs::get_model::Response & res){
    int model_id			= req.model_id;
    reglib::Model * model	= models[model_id];
    res.model = getModelMSG(model);
    return true;
}

bool recognizeService(quasimodo_msgs::recognize::Request  & req, quasimodo_msgs::recognize::Response & res){
    reglib::Model * mod = quasimodo_brain::getModelFromSegment(*nh, req.id);
	addNewModel(mod);
	reglib::Model * p = mod->parrent;
	if((p == 0) || ((p->frames.size() == 0) && (p->submodels.size() == 1))){
		req.id = "";
		return false;
	}else{
		req.id = p->soma_id;
		return true;
	}
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

bool addIfPossible(ModelDatabase * database, reglib::Model * model, reglib::Model * model2){
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
			for(unsigned int j = 0; j < ud.updated_models.size();	j++){	addToDB(database, ud.updated_models[j], true);}
			for(unsigned int j = 0; j < ud.new_models.size();	j++){		addToDB(database, ud.new_models[j],		true);}
			return true;
		}
	}else{
		delete mu;
		delete reg;
	}
	return false;
}

bool addToDB(ModelDatabase * database, reglib::Model * model, bool add){// = true){, bool deleteIfFail = false){
printf("start: %s\n",__PRETTY_FUNCTION__);
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
		if(addIfPossible(database,model,res[i])){
			printf("stop: %s\n",__PRETTY_FUNCTION__);
			return true;
		}
	}
printf("stop: %s\n",__PRETTY_FUNCTION__);
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
printf("start: %s\n",__PRETTY_FUNCTION__);
	reglib::RegistrationRandom *	reg	= new reglib::RegistrationRandom();
	reg->visualizationLvl				= show_reg_lvl;
	reglib::ModelUpdaterBasicFuse * mu	= new reglib::ModelUpdaterBasicFuse( model, reg);
	mu->occlusion_penalty               = occlusion_penalty;
	mu->massreg_timeout                 = massreg_timeout;
	mu->viewer							= viewer;
	mu->show_init_lvl					= show_init_lvl;//init show
	mu->show_refine_lvl					= show_refine_lvl;//refine show
	mu->show_scoring					= show_scoring;//fuse scoring show
	mu->makeInitialSetup();

	delete mu;
	delete reg;

	reglib::Model * newmodelHolder = new reglib::Model();
	model->parrent = newmodelHolder;
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
	printf("%i\n",__LINE__);
	show_sorted();

	printf("%i\n",__LINE__);
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

	printf("%i\n",__LINE__);
	for(unsigned int i = 0; i < modeldatabase->models.size(); i++){publish_history(modeldatabase->models[i]->getHistory());}
	printf("%i\n",__LINE__);
	publishDatabasePCD();
	printf("%i\n",__LINE__);
	dumpDatabase(savepath);
	printf("stop: %s\n",__PRETTY_FUNCTION__);
}

void somaCallback(const std_msgs::String & m){printf("somaCallback(%s)\n",m.data.c_str());}

void modelCallback(const quasimodo_msgs::model & m){
	quasimodo_msgs::model mod = m;
	addNewModel(quasimodo_brain::getModelFromMSG(mod,false));
}

void clearMem(){
	for(auto iterator = cameras.begin();	iterator != cameras.end();	iterator++) {delete iterator->second;}
	for(auto iterator = frames.begin();		iterator != frames.end();	iterator++) {delete iterator->second;}
	for(auto iterator = models.begin();		iterator != models.end();	iterator++) {
		reglib::Model * model = iterator->second;
		for(unsigned int i = 0; i < model->frames.size()	; i++){delete model->frames[i];}
		for(unsigned int i = 0; i < model->modelmasks.size(); i++){delete model->modelmasks[i];}
		delete model;
	}
	for(auto iterator = updaters.begin(); iterator != updaters.end(); iterator++) {delete iterator->second;}
	if(registration != 0){delete registration;}
	if(modeldatabase != 0){delete modeldatabase;}
}

int main(int argc, char **argv){
	cameras[0]		= new reglib::Camera();
	registration	= new reglib::RegistrationRandom();
	modeldatabase	= 0;

	ros::init(argc, argv, "quasimodo_model_server");
	ros::NodeHandle n;
    nh = &n;
	models_new_pub		= n.advertise<quasimodo_msgs::model>("/models/new",		1000);
	models_updated_pub	= n.advertise<quasimodo_msgs::model>("/models/updated", 1000);
	models_deleted_pub	= n.advertise<quasimodo_msgs::model>("/models/deleted", 1000);

	//ros::ServiceServer service1 = n.advertiseService("model_from_frame",	modelFromFrame);
	ros::ServiceServer service2 = n.advertiseService("index_frame",			indexFrame);
	ros::ServiceServer service4 = n.advertiseService("get_model",			getModel);
    ros::ServiceServer service5 = n.advertiseService("quasimodo/recognize", recognizeService);
	ROS_INFO("Ready to add use services.");

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

	bool clearQDB = false;

	int inputstate = -1;
	for(int i = 1; i < argc;i++){
		printf("input: %s\n",argv[i]);
		if(		std::string(argv[i]).compare("-c") == 0){	printf("camera input state\n"); inputstate = 1;}
		else if(std::string(argv[i]).compare("-l") == 0){	printf("reading all models at %s (the path defined from -p)\n",savepath.c_str());
			std::vector<std::string> folderdata;
			int r = quasimodo_brain::getdir(savepath+"/",folderdata);
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
		else if(std::string(argv[i]).compare("-clearQDB") == 0){									clearQDB = true;}
		else if(inputstate == 1){
			reglib::Camera * cam = reglib::Camera::load(std::string(argv[i]));
			delete cameras[0];
            cameras[0] = cam;
		}else if(inputstate == 2){
			reglib::Model * model = reglib::Model::load(cameras[0],std::string(argv[i]));
			sweepid_counter = std::max(int(model->modelmasks[0]->sweepid + 1), sweepid_counter);
			modeldatabase->add(model);
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
			if(search_timeout == 0){run_search = false;}
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

//	if(clearQDB){
//		printf("rm -r -f %s\n",(savepath+"/database_tmp").c_str());
//		char command [1024];
//		printf(command,"rm -r -f %s",(savepath+"/database_tmp").c_str());
//		int r = system(command);
//	}


//	if(fileExists(savepath+"/database_tmp")){
//		printf("file exists\n");

//	}else{
//		printf("making savedi: %s\n",(savepath+"/database_tmp").c_str());
//		char command [1024];
//		sprintf(command,"mkdir %s",(savepath+"/database_tmp").c_str());
//		int r = system(command);
//	}
//	//exit(0);

	if(visualization){
		viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer ("Modelserver Viewer"));
		viewer->addCoordinateSystem(0.1);
		viewer->setBackgroundColor(1.0,0.0,1.0);
	}

	if(modeldatabase == 0){modeldatabase	= new ModelDatabaseRetrieval(n);}

	for(unsigned int i = 0; i < modelpcds.size(); i++){
		std::vector<reglib::Model *> mods = quasimodo_brain::loadModelsPCDs(modelpcds[i]);
		for(unsigned int j = 0; j < mods.size(); j++){
			//modeldatabase->add(mods[j]);
			//addNewModel(mods[j]);

			reglib::Model * model = mods[j];

			reglib::RegistrationRandom *	reg	= new reglib::RegistrationRandom();
			reg->visualizationLvl				= show_reg_lvl;
			reglib::ModelUpdaterBasicFuse * mu	= new reglib::ModelUpdaterBasicFuse( model, reg);
			mu->occlusion_penalty               = occlusion_penalty;
			mu->massreg_timeout                 = massreg_timeout;
			mu->viewer							= viewer;
			mu->show_init_lvl					= show_init_lvl;//init show
			mu->show_refine_lvl					= show_refine_lvl;//refine show
			mu->show_scoring					= show_scoring;//fuse scoring show
			mu->makeInitialSetup();

			delete mu;
			delete reg;

			reglib::Model * newmodelHolder = new reglib::Model();
			model->parrent = newmodelHolder;
			newmodelHolder->submodels.push_back(model);
			newmodelHolder->submodels_relativeposes.push_back(Eigen::Matrix4d::Identity());
			newmodelHolder->last_changed = ++current_model_update;
			if(show_modelbuild){
				newmodelHolder->recomputeModelPoints(Eigen::Matrix4d::Identity(),viewer);
			}else{
				newmodelHolder->recomputeModelPoints();
			}
			modeldatabase->add(newmodelHolder);
		}

		//show_sorted();
	}

	if(input_model_subs.size()		== 0){input_model_subs.push_back(		n.subscribe("/quasimodo/segmentation/out/model", 100, modelCallback));}
	if(soma_input_model_subs.size() == 0){soma_input_model_subs.push_back(	n.subscribe("/quasimodo/segmentation/out/soma_segment_id", 10000, somaCallback));}



	ros::spin();
	return 0;
}
