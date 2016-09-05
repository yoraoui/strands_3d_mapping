#include "ModelDatabaseRetrieval.h"

using namespace std;



ModelDatabaseRetrieval::ModelDatabaseRetrieval(ros::NodeHandle & n, std::string retrieval_name, std::string conversion_name, std::string insert_name){
	retrieval_client	= n.serviceClient<quasimodo_msgs::query_cloud>				(retrieval_name);
	conversion_client	= n.serviceClient<quasimodo_msgs::model_to_retrieval_query>	(conversion_name);
	insert_client		= n.serviceClient<quasimodo_msgs::insert_model>				(insert_name);
}

ModelDatabaseRetrieval::~ModelDatabaseRetrieval(){}

bool ModelDatabaseRetrieval::add(reglib::Model * model){
	quasimodo_msgs::insert_model im;
	im.request.model = quasimodo_brain::getModelMSG(model);
	im.request.action = im.request.INSERT;
	if (insert_client.call(im)){
		object_ids.push_back(im.response.object_id);
		models.push_back(model);
		vocabulary_ids[im.response.vocabulary_id] = model;
		v_ids.push_back(im.response.vocabulary_id);
		printf("number of models: %i\n",models.size());
		return true;
	}else{
		ROS_ERROR("insert_client service INSERT FAIL!");
		return false;
	}
}

bool ModelDatabaseRetrieval::remove(reglib::Model * model){
	for(unsigned int i = 0; i < models.size(); i++){
		if(models[i] == model){
			//Remove stuff
			quasimodo_msgs::insert_model im;
			im.request.object_id = object_ids[i];
			im.request.action = im.request.REMOVE;
			if (insert_client.call(im)){
				vocabulary_ids.erase(v_ids[i]);
				models[i] = models.back();
				models.pop_back();
				object_ids[i] = object_ids.back();
				object_ids.pop_back();
				v_ids[i] = v_ids.back();
				v_ids.pop_back();
				return true;
			}else{
				ROS_ERROR("insert_client service REMOVE FAIL!");
				return false;
			}
		}
	}
	return true;
}

std::vector<reglib::Model *> ModelDatabaseRetrieval::search(reglib::Model * model, int number_of_matches){
	std::vector<reglib::Model *> ret;

	quasimodo_msgs::model_to_retrieval_query m2r;
	m2r.request.model = quasimodo_brain::getModelMSG(model);
	if (conversion_client.call(m2r)){
		quasimodo_msgs::query_cloud qc;
		qc.request.query = m2r.response.query;
		qc.request.query.query_kind = qc.request.query.MONGODB_QUERY;
		qc.request.query.number_query = number_of_matches;

		if (retrieval_client.call(qc)){
			quasimodo_msgs::retrieval_result result = qc.response.result;
			for(unsigned int i = 0; i < result.segment_indices.size(); i++){
				reglib::Model * search_model = vocabulary_ids[result.segment_indices[i].ints.front()];
				if(search_model != model){ret.push_back(search_model);}
				if(ret.size() == number_of_matches){return ret;}
			}
		}else{
			ROS_ERROR("retrieval_client service FAIL!");
		}
	}else{
		ROS_ERROR("model_to_retrieval_query service FAIL!");
	}
	return ret;
}
