#ifndef ModelStorage_H
#define ModelStorage_H

#include <vector>

// PCL specific includes
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "model/Model.h"
#include "../Util/Util.h"

class ModelStorage{
	public:
    std::map<std::string,reglib::Model *> activeModels;


	virtual std::string getNextID();
	virtual void print();
    virtual std::vector<std::string> loadAllModels();
    virtual bool add(reglib::Model * model, std::string key = "");
    virtual bool update(reglib::Model * model);
    virtual bool remove(reglib::Model * model);
    virtual reglib::Model * fetch(std::string key);
    virtual void handback(reglib::Model * model, bool full = false);
    virtual void fullHandback();
    virtual void saveSnapshot();
    virtual pcl::PointCloud<pcl::PointXYZRGB>::Ptr getSnapshot();
		
	ModelStorage();
	~ModelStorage();
};

#include "ModelStorageFile.h"
#endif // ModelStorage_H
