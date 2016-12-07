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
	printf("start: %s\n",__PRETTY_FUNCTION__);
	quasimodo_msgs::insert_model im;
	im.request.model = quasimodo_brain::getModelMSG(model,true);
	im.request.action = im.request.INSERT;
	if (insert_client.call(im)){
//		object_ids.push_back(im.response.object_id);
//		models.push_back(model);
//		vocabulary_ids[im.response.vocabulary_id] = model;
//		v_ids.push_back(im.response.vocabulary_id);

		//update model to save new_responses
		//put model in storage
		model->retrieval_object_id = im.response.object_id;
		model->retrieval_vocabulary_id = im.response.vocabulary_id;
		model->keyval = im.response.vocabulary_id;

		printf("object_id: %s\n",model->retrieval_object_id.c_str());
		printf("vocabulary_id: %s\n",model->retrieval_vocabulary_id.c_str());
		storage->add(model);
		return true;
	}else{
		ROS_ERROR("insert_client service INSERT FAIL!");
		return false;
	}
}

bool ModelDatabaseRetrieval::remove(reglib::Model * model){
printf("start: %s\n",__PRETTY_FUNCTION__);
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
//	storage->remove(model);
//	for(unsigned int i = 0; i < models.size(); i++){
//		if(models[i] == model){
//			//Remove stuff
//			quasimodo_msgs::insert_model im;
//			im.request.object_id = object_ids[i];
//			im.request.action = im.request.REMOVE;
//			if (insert_client.call(im)){
//				vocabulary_ids.erase(v_ids[i]);
//				models[i] = models.back();
//				models.pop_back();
//				std::string oid = object_ids[i];
//				object_ids[i] = object_ids.back();
//				object_ids.pop_back();
//				int vid = v_ids[i];
//				v_ids[i] = v_ids.back();
//				v_ids.pop_back();
//				printf("REMOVE: number of models in database: %6.6i removed vocabulary_id: %6.6i last added object_ids: %s\n",models.size(),vid,oid.c_str());
//				return true;
//			}else{
//				ROS_ERROR("insert_client service REMOVE FAIL!");
//				return false;
//			}
//		}
//	}
	return true;
}

std::vector<reglib::Model *> ModelDatabaseRetrieval::search(reglib::Model * model, int number_of_matches){
printf("start: %s\n",__PRETTY_FUNCTION__);

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
			printf("---------------ids in searchresult: %i----------------\n",result.vocabulary_ids.size());
			for(unsigned int i = 0; i < result.vocabulary_ids.size(); i++){

				if(model->keyval.compare(result.vocabulary_ids[i]) == 0){continue;}//Returned itself... ignore
				reglib::Model * mod = storage->fetch(result.vocabulary_ids[i]);
				if(mod != 0 && mod->keyval.compare(model->keyval) != 0){
					ret.push_back(mod);
				}
				if(ret.size() == number_of_matches){return ret;}
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
