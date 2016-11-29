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

	virtual std::string getNextID();
	virtual bool add(reglib::Model * model);
	virtual bool remove(reglib::Model * model);
		
	ModelStorage();
	~ModelStorage();
};

#include "ModelStorageFile.h"
#endif // ModelStorage_H
