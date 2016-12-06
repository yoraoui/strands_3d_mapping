#include "ModelStorage.h"

ModelStorage::ModelStorage(){}
ModelStorage::~ModelStorage(){}

std::vector<std::string> ModelStorage::loadAllModels(){
    return std::vector<std::string>();
}
std::string ModelStorage::getNextID(){return "";}

bool ModelStorage::add(reglib::Model * model, std::string key){return true;}
bool ModelStorage::update(reglib::Model * model){return true;}
bool ModelStorage::remove(reglib::Model * model){return false;}
reglib::Model * ModelStorage::fetch(std::string key){return 0;}
void ModelStorage::handback(reglib::Model * model, bool full){}
void ModelStorage::saveSnapshot(){}
pcl::PointCloud<pcl::PointXYZRGB>::Ptr ModelStorage::getSnapshot(){
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr	cloud	(new pcl::PointCloud<pcl::PointXYZRGB>);
    return cloud;
}

void ModelStorage::print(){
	printf("/////////////////////////////////////////////////////////////\n");
	printf("/////////////////////////////////////////////////////////////\n");
	printf("/////////////////////////////////////////////////////////////\n");
	printf("/////////////////////////////////////////////////////////////\n");
	printf("/////////////////////////////////////////////////////////////\n");
	printf("active:\n");
	for (auto it=activeModels.begin(); it!=activeModels.end(); ++it){
	  std::cout << it->first << '\n';
	}
	printf("/////////////////////////////////////////////////////////////\n");
	printf("/////////////////////////////////////////////////////////////\n");
	printf("/////////////////////////////////////////////////////////////\n");
	printf("/////////////////////////////////////////////////////////////\n");
	printf("/////////////////////////////////////////////////////////////\n");
}




void ModelStorage::fullHandback(){
    for (std::map<std::string,reglib::Model * >::iterator it=activeModels.begin(); it!=activeModels.end(); ++it){
		printf("removing %s -> %s\n",it->first.c_str(),it->second->keyval.c_str());
		handback(it->second,true);
    }
}
