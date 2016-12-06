#ifndef ModelStorageFile_H
#define ModelStorageFile_H

#include "ModelStorage.h"
#include <map>
#include <string>

class ModelStorageFile: public ModelStorage{
	public:
	std::string filepath;
	std::string framepath;
    unsigned long saveCounter;
    std::map<std::string,std::string> keyPathMap;

    virtual std::vector<std::string> loadAllModels();
	virtual std::string getNextID();
    virtual bool add(reglib::Model * model, std::string key = "");
    virtual bool update(reglib::Model * model);
	virtual bool remove(reglib::Model * model);
    virtual reglib::Model * fetch(std::string key);
    virtual void handback(reglib::Model * model, bool full = false);
	virtual void fullHandback();
    virtual void saveSnapshot();
    virtual pcl::PointCloud<pcl::PointXYZRGB>::Ptr getSnapshot();

	ModelStorageFile(std::string filepath_ = "./");
	~ModelStorageFile();
};



#endif // ModelStorageFile_H
