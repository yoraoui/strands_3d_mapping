#include "ModelDatabase.h"

ModelDatabase::ModelDatabase(){storage = new ModelStorageFile();}
ModelDatabase::~ModelDatabase(){}

//Add pointcloud to database, return index number in database, weight is the bias of the system to perfer this object when searching
bool ModelDatabase::add(reglib::Model * model){
	storage->add(model);
    modelkeys.insert(model->keyval);
	return true;
}
		
// return true if successfull
// return false if fail
bool ModelDatabase::remove(reglib::Model * model){
	storage->remove(model);
    modelkeys.erase(model->keyval);
	return false;
}


bool ModelDatabase::update(reglib::Model * model){
    remove(model);
    add(model);
    return true;
}


bool ModelDatabase::setStorage(ModelStorage * storage_){
    if(storage != 0){delete storage;}
    storage = storage_;
    std::vector<std::string> ret = storage->loadAllModels();
    for(unsigned int i = 0; i < ret.size(); i++){modelkeys.insert(ret[i]);}
	return true;
}
		
//Find the number_of_matches closest matches in dabase to the pointcloud for index 
std::vector<reglib::Model *> ModelDatabase::search(reglib::Model * model, int number_of_matches){
	return std::vector<reglib::Model *>();
}
