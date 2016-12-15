#include "ModelStorageFile.h"

ModelStorageFile::ModelStorageFile(std::string filepath_){
	filepath = filepath_+"/quasimodo_modeldatabase";
    framepath = filepath+"/frames";
    saveCounter = 0;
    quasimodo_brain::guaranteeFolder(filepath);
    quasimodo_brain::guaranteeFolder(framepath);
    while(true){
        char buf [1024];
        sprintf(buf,"%s/quasimodoDB_%10.10i.pcd",filepath.c_str(),saveCounter);
        if(!quasimodo_brain::fileExists(std::string(buf))){break;}
        else{saveCounter++;}
    }
}
ModelStorageFile::~ModelStorageFile(){}

std::vector<std::string> ModelStorageFile::loadAllModels(){
    std::vector<std::string> ret;
    std::vector<std::string> files;
    quasimodo_brain::getdirs(filepath,files);
    for(unsigned int i = 0; i < files.size(); i++){
        std::string path = filepath+"/"+files[i];
        if((files[i].compare(".") != 0) && (files[i].compare("..") != 0) && (path.compare(framepath) != 0)){
            std::ifstream file (path+"/data.bin", std::ios::in | std::ios::binary | std::ios::ate);
            if (file.is_open()){
                unsigned long parrent_keyvallength;
                file.seekg (0, std::ios::beg);
                file.read ((char *)(&parrent_keyvallength), sizeof(unsigned long));
                file.close();
                if(parrent_keyvallength == 0){
                    printf("loaded %s\n",files[i].c_str());
                    keyPathMap[files[i]] = path+"/";
                    ret.push_back(files[i]);
                }
            }
        }
	}
    return ret;
}

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

bool ModelStorageFile::add(reglib::Model * model, std::string key){
	double startTime =quasimodo_brain::getTime();
    if(model->keyval.length() == 0){model->keyval = getNextID();}
    printf("ModelStorageFile::add(%s)\n",model->keyval.c_str());
    std::string modelpath = filepath+"/"+model->keyval;
    quasimodo_brain::guaranteeFolder(modelpath);
	modelpath += "/";

    for(unsigned int i = 0; i < model->frames.size(); i++){//Add all frames to the frame database
        std::string path = framepath+"/"+model->frames[i]->keyval;
        if(!quasimodo_brain::fileExists(path+"_data.txt")){model->frames[i]->saveFast(path);}
    }
    for(unsigned int i = 0; i < model->submodels.size(); i++){//Add all frames to the frame database
        add(model->submodels[i]);
	}
	model->saveFast(modelpath);

    keyPathMap[model->keyval] = modelpath;
    if(model->parrent == 0){
        activeModels[model->keyval] = model;
    }
    print();
}

bool ModelStorageFile::update(reglib::Model * model){
    printf("%s\n",__PRETTY_FUNCTION__);
    return true;
}

bool ModelStorageFile::remove(reglib::Model * model){
    std::string path = keyPathMap[model->keyval];
    boost::filesystem::path dir(path);
    boost::filesystem::remove_all(dir);
    keyPathMap.erase(model->keyval);
    activeModels.erase(model->keyval);
    return true;
}

reglib::Model * ModelStorageFile::fetch(std::string key){
    if (keyPathMap.count(key)>0){
		reglib::Model * model = 0;
		if(activeModels.count(key)!=0){
			model = activeModels.find(key)->second;
        }else{
			model = reglib::Model::loadFast(keyPathMap[key]);
			activeModels[key] = model;
        }
		return model;
    }else{
        return 0;
    }
}

void ModelStorageFile::fullHandback(){
	for (auto it=activeModels.begin(); it!=activeModels.end(); ++it){
        printf("activeModels: %i\n",long(it->second));
		it->second->fullDelete();
		delete it->second;
	}
	activeModels.clear();
}

void ModelStorageFile::handback(reglib::Model * model, bool full){
	activeModels.erase(model->keyval);
	model->fullDelete();
	delete model;
}
void ModelStorageFile::saveSnapshot(){
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = getSnapshot();
    if(cloud->points.size() > 0){
        char buf [1024];
        sprintf(buf,"%s/quasimodoDB_%10.10i.pcd",filepath.c_str(),saveCounter++);
        pcl::io::savePCDFileBinaryCompressed(buf, *getSnapshot());
    }
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr ModelStorageFile::getSnapshot(){
    double startTime = quasimodo_brain::getTime();

    double global_max = 0;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr	cloud	(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (std::map<std::string,std::string>::iterator it=keyPathMap.begin(); it!=keyPathMap.end(); ++it){
        std::ifstream pointfile (it->second+"points.bin", std::ios::in | std::ios::binary | std::ios::ate);
        if (pointfile.is_open()){
            unsigned long size = pointfile.tellg();
            char * buffer = new char [size];
            pointfile.seekg (0, std::ios::beg);
            pointfile.read (buffer, size);
            pointfile.close();
            float *		data	= (float *)buffer;
            long sizeofSuperPoint = 3*(3+1);

            unsigned int nr_points = size/(sizeofSuperPoint*sizeof(float));
            float minx = data[0];
            float maxx = data[0];
            for(unsigned long i = 1; i < nr_points; i++){
                maxx = std::max(data[12*i],maxx);
                minx = std::min(data[12*i],minx);
            }

            for(unsigned long i = 0; i < nr_points; i++){
                pcl::PointXYZRGB p;
                p.x = data[12*i+0]-minx+global_max;
                p.y = data[12*i+1];
                p.z = data[12*i+2];
                p.r = data[12*i+6];
                p.g = data[12*i+7];
                p.b = data[12*i+8];
                cloud->points.push_back(p);
            }
            delete[] buffer;
            global_max += maxx-minx + 0.15;
        }
    }
    return cloud;
}

void ModelStorageFile::print(){
    printf("///////////////////////ModelStorageFile::print()////////////////////////////\n");
    for (auto it=activeModels.begin(); it!=activeModels.end(); ++it){
      std::cout << "active: " << it->first;
      printf(" -> %ld\n",long(it->second));
    }

    for (auto it=keyPathMap.begin(); it!=keyPathMap.end(); ++it){
      std::cout << "keyPathMap: " << it->first << "--->" << it->second <<'\n';
    }
    printf("////////////////////////////////////////////////////////////////////////////\n");

}
