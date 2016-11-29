#include "ModelStorage.h"

ModelStorage::ModelStorage(){}
ModelStorage::~ModelStorage(){}

//Add pointcloud to database, return index number in database, weight is the bias of the system to perfer this object when searching
bool ModelStorage::add(reglib::Model * model, std::string parrentpath){
	return true;
}

bool ModelStorage::remove(reglib::Model * model){
	return false;
}
		
