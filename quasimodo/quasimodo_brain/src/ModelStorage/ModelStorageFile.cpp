#include "ModelStorageFile.h"

ModelStorageFile::ModelStorageFile(std::string filepath_){
	filepath = filepath_+"/database_tmp";
	framepath = filepath+"/frames";
	quasimodo_brain::guaranteeFolder(filepath);
	quasimodo_brain::guaranteeFolder(framepath);

//	if(!quasimodo_brain::fileExists(filepath)){
//		char command [1024];
//		sprintf(command,"mkdir %s",filepath.c_str());
//		int r = system(command);
//	}

//	if(!quasimodo_brain::fileExists(filepath+"/frames")){
//		char command [1024];
//		sprintf(command,"mkdir %s",(filepath+"/frames").c_str());
//		int	r = system(command);
//	}
}
ModelStorageFile::~ModelStorageFile(){}

std::string ModelStorageFile::getNextID(){
	std::string strTemplate = "Model_";
	std::vector<std::string> files;
	int maxind = -1;
	quasimodo_brain::getdir (filepath,files);
	for(unsigned int i = 0; i < files.size(); i++){
		std::string currentStr = files[i];
		if(currentStr.length() >= strTemplate.length()){
			std::string frontpart = currentStr.substr(0,strTemplate.length());
			std::string backpart = currentStr.substr(strTemplate.length(),currentStr.length());
			if(frontpart.compare(strTemplate) == 0 && quasimodo_brain::isNumber(backpart)){
				maxind = std::max(atoi(backpart.c_str()),maxind);
			}
		}
	}

	char buf [1024];
	sprintf(buf,"%s%i",strTemplate.c_str(),maxind+1);
	return std::string(buf);
}


bool ModelStorageFile::add(reglib::Model * model){

	if(model->keyval.length() == 0){model->keyval = getNextID();}

	printf("model keyval: %s\n",model->keyval.c_str());

	std::string modelpath = filepath+"/"+model->keyval;
	quasimodo_brain::guaranteeFolder(modelpath);

	for(unsigned int i = 0; i < model->frames.size(); i++){//Add all frames to the frame database
		std::string path = framepath+"/"+model->frames[i]->keyval;
		if(!quasimodo_brain::fileExists(path+"_data.txt")){
			model->frames[i]->saveFast(path);
		}
	}

	for(unsigned int i = 0; i < model->submodels.size(); i++){//Add all frames to the frame database
		add(model->submodels[i]);
	}

	model->saveFast(modelpath);
	//	models.push_back(model);
	//	return true;
	//	//printf("number of models: %i\n",models.size());
	exit(0);
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
