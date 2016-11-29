#include "ModelStorageFile.h"

ModelStorageFile::ModelStorageFile(std::string filepath_){
	filepath = filepath_;
	if(!quasimodo_brain::fileExists(filepath+"/database_tmp")){
		char command [1024];
		sprintf(command,"mkdir %s",(filepath+"/database_tmp").c_str());
		int r = system(command);
		if(!quasimodo_brain::fileExists(filepath+"/database_tmp/frames")){
			sprintf(command,"mkdir %s",(filepath+"/database_tmp/frames").c_str());
			r = system(command);
		}
	}
}
ModelStorageFile::~ModelStorageFile(){}


bool ModelStorageFile::add(reglib::Model * model, std::string parrentpath){
//	printf("%s\n",__PRETTY_FUNCTION__);


//	for(unsigned int i = 0; i < model->frames.size(); i++){//Add all frames to the frame database
//		std::string keyval = model->frames[i]->keyval;
//		printf("keyval: %s\n",keyval.c_str());
//	}


//	for(unsigned int i = 0; i < model->submodels.size(); i++){//Add all frames to the frame database
//		add(model->submodels[i],parrentpath+"/sub");
//	}
	//	models.push_back(model);
	//	return true;
	//	//printf("number of models: %i\n",models.size());
	//exit(0);
}

bool ModelStorageFile::remove(reglib::Model * model){
	//printf("%s\n",__PRETTY_FUNCTION__);
	//	for(unsigned int i = 0; i < models.size(); i++){
	//		if(models[i] == model){
	//			models[i] = models.back();
	//			models.pop_back();
	//			return true;
	//		}
	//	}
	//	return false;
	//exit(0);
}
