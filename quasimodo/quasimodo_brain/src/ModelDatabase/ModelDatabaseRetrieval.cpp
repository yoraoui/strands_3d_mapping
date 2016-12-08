#include "ModelDatabaseRetrieval.h"
//#include "mongo/client/dbclient.h"

using namespace std;



ModelDatabaseRetrieval::ModelDatabaseRetrieval(ros::NodeHandle & n, std::string retrieval_name, std::string conversion_name, std::string insert_name){
	retrieval_client	= n.serviceClient<quasimodo_msgs::query_cloud>				(retrieval_name);
	conversion_client	= n.serviceClient<quasimodo_msgs::model_to_retrieval_query>	(conversion_name);
	insert_client		= n.serviceClient<quasimodo_msgs::insert_model>				(insert_name);
	storage = new ModelStorageFile();
}

ModelDatabaseRetrieval::~ModelDatabaseRetrieval(){}

bool ModelDatabaseRetrieval::add(reglib::Model * model){
	double startTime =quasimodo_brain::getTime();
	quasimodo_msgs::insert_model im;
	im.request.model = quasimodo_brain::getModelMSG(model,true);
	im.request.action = im.request.INSERT;
	printf("starting to insert into retrieval database\n");
	if (insert_client.call(im)){
		model->retrieval_object_id = im.response.object_id;
		model->retrieval_vocabulary_id = std::to_string(im.response.vocabulary_id);
		model->keyval = model->retrieval_vocabulary_id;
		printf("ModelDatabaseRetrieval::add :: %5.5fs\n",quasimodo_brain::getTime()-startTime);
		storage->add(model);
		return true;
	}else{
		ROS_ERROR("insert_client service INSERT FAIL!");
		return false;
	}
}

bool ModelDatabaseRetrieval::remove(reglib::Model * model){
	quasimodo_msgs::insert_model im;
	im.request.object_id = model->retrieval_object_id;
	im.request.action = im.request.REMOVE;
	if (insert_client.call(im)){
		storage->remove(model);
		return true;
	}else{
		ROS_ERROR("insert_client service REMOVE FAIL!");
		return false;
	}
	return true;
}

std::vector<reglib::Model *> ModelDatabaseRetrieval::search(reglib::Model * model, int number_of_matches){
	std::vector<reglib::Model *> ret;
	if(model == 0 || model->keyval.length() == 0){return ret;}

	quasimodo_msgs::model_to_retrieval_query m2r;
	m2r.request.model = quasimodo_brain::getModelMSG(model,true);
	if (conversion_client.call(m2r)){
		quasimodo_msgs::query_cloud qc;
		qc.request.query = m2r.response.query;
		qc.request.query.query_kind = qc.request.query.MONGODB_QUERY;
		qc.request.query.number_query = number_of_matches+10;
		if (retrieval_client.call(qc)){
			quasimodo_msgs::retrieval_result result = qc.response.result;
			for(unsigned int i = 0; i < result.vocabulary_ids.size(); i++){
				std::string vid = result.vocabulary_ids[i];
				vid.pop_back();

				if(model->keyval.compare(vid) == 0){continue;}//Returned itself... ignore
				reglib::Model * mod = storage->fetch(vid);
				if(mod != 0){
					if(mod->keyval.compare(model->keyval) != 0){
						ret.push_back(mod);
						if(ret.size() == number_of_matches){return ret;}
					}
				}
			}
		}else{
			ROS_ERROR("retrieval_client service FAIL!");
		}
	}else{
		ROS_ERROR("model_to_retrieval_query service FAIL!");
	}
printf("stop: %s\n",__PRETTY_FUNCTION__);
	return ret;
}
