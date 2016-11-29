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

	//Add pointcloud to database, return index number in database, weight is the bias of the system to perfer this object when searching
	virtual bool add(reglib::Model * model, std::string parrentpath = "");
		
	// return true if successfull
	// return false if fail
	virtual bool remove(reglib::Model * model);
		
	ModelStorage();
	~ModelStorage();
};

#include "ModelStorageFile.h"
#endif // ModelStorage_H
