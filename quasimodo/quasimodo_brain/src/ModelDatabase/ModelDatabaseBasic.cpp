#include "ModelDatabaseBasic.h"

ModelDatabaseBasic::ModelDatabaseBasic(){storage = new ModelStorageFile();}
ModelDatabaseBasic::~ModelDatabaseBasic(){}


bool ModelDatabaseBasic::add(reglib::Model * model){
    storage->add(model);
    modelkeys.insert(model->keyval);
	return true;
}

bool ModelDatabaseBasic::remove(reglib::Model * model){
    storage->remove(model);
    modelkeys.erase(model->keyval);
	return false;
}

std::vector<reglib::Model *> ModelDatabaseBasic::search(reglib::Model * model, int number_of_matches){
	std::vector<reglib::Model *> ret;
    for (std::set<std::string>::iterator it=modelkeys.begin(); it!=modelkeys.end(); ++it){
		if(model == 0 || model->keyval.compare(*it) == 0){continue;}
		//printf("------------------------------\n");
		//storage->print();
        reglib::Model * mod = storage->fetch(*it);
		//storage->print();
		//printf("------------------------------\n");

        if(mod != 0 && mod->keyval.compare(model->keyval) != 0){
            ret.push_back(mod);
        }
        if(ret.size() == number_of_matches){return ret;}
    }
	return ret;
}
