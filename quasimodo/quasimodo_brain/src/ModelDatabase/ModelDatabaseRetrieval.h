#ifndef MODELDATABASERETRIEVAL_H
#define MODELDATABASERETRIEVAL_H

#include "ModelDatabase.h"
#include "quasimodo_msgs/query_cloud.h"
#include "quasimodo_msgs/model_to_retrieval_query.h"
#include "quasimodo_msgs/insert_model.h"
#include <map>

class ModelDatabaseRetrieval: public ModelDatabase{
public:
	ros::ServiceClient retrieval_client;
	ros::ServiceClient conversion_client;
	ros::ServiceClient insert_client;


	std::vector< std::string > object_ids;
	std::vector< unsigned long > v_ids;
	std::map<unsigned long, reglib::Model * > vocabulary_ids;

	virtual bool add(reglib::Model * model);
	virtual bool remove(reglib::Model * model);
	virtual std::vector<reglib::Model *> search(reglib::Model * model, int number_of_matches);

	ModelDatabaseRetrieval(ros::NodeHandle & n, std::string retrieval_name = "/quasimodo_retrieval_service", std::string conversion_name = "/models/server",std::string insert_name = "/insert_model_service");
    ~ModelDatabaseRetrieval();
};

#endif // MODELDATABASERETRIEVAL_H
