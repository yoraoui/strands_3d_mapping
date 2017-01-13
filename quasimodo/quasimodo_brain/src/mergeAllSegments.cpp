#include "ModelDatabase/ModelDatabase.h"
#include "ModelStorage/ModelStorage.h"
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
ModelStorageFile *                      storage ;
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

double occlusion_penalty	= 5;
double massreg_timeout		= 120;
bool run_search				= false;
int sweepid_counter			= 0;
int current_model_update	= 0;

bool myfunction (reglib::Model * i,reglib::Model * j) { return (i->frames.size() + i->submodels.size())  > (j->frames.size() + j->submodels.size()); }

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

bool addIfPossible(ModelDatabase * database, reglib::Model * model, reglib::Model * model2){
	printf("%s\n",__PRETTY_FUNCTION__);
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
exit(0);

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
	printf("%ld front point: ",long(newmodelHolder));
	newmodelHolder->points.front().print();
	model->updated = true;
	newmodelHolder->updated = true;

	//storage->print();
	modeldatabase->add(newmodelHolder);

	return ;
/*

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
	printf("stop: %s\n",__PRETTY_FUNCTION__);
	*/
}

void somaCallback(const std_msgs::String & m){printf("somaCallback(%s)\n",m.data.c_str());}

void modelCallback(const quasimodo_msgs::model & m){
	printf("----------%s----------\n",__PRETTY_FUNCTION__);
	quasimodo_msgs::model mod = m;
	reglib::Model * model = quasimodo_brain::getModelFromMSG(mod,true);

	addNewModel(model);
	printf("done... handback!\n");
	storage->fullHandback();
}

int main(int argc, char **argv){
	modeldatabase = new ModelDatabaseBasic();
	storage = new ModelStorageFile();
	modeldatabase->setStorage( storage );
	storage->print();
exit(0);
	cameras[0]		= new reglib::Camera();
	registration	= new reglib::RegistrationRandom();

	ros::init(argc, argv, "quasimodo_model_server");
	ros::NodeHandle n;
    nh = &n;
	models_new_pub		= n.advertise<quasimodo_msgs::model>("/models/new",		1000);
	models_updated_pub	= n.advertise<quasimodo_msgs::model>("/models/updated", 1000);
	models_deleted_pub	= n.advertise<quasimodo_msgs::model>("/models/deleted", 1000);

	ros::ServiceServer service4 = n.advertiseService("get_model",			getModel);
	ros::ServiceServer service5 = n.advertiseService("quasimodo/recognize", recognizeService);
	ROS_INFO("Ready to add use services.");

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
	if(input_model_subs.size()		== 0){input_model_subs.push_back(		n.subscribe("/quasimodo/segmentation/out/model", 100, modelCallback));}
	if(soma_input_model_subs.size() == 0){soma_input_model_subs.push_back(	n.subscribe("/quasimodo/segmentation/out/soma_segment_id", 10000, somaCallback));}

	printf("done with loading and setup, starting\n");
	ros::spin();
	return 0;
}
