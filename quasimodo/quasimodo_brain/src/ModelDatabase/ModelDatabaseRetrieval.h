#ifndef MODELDATABASERETRIEVAL_H
#define MODELDATABASERETRIEVAL_H

#include "ModelDatabase.h"

class ModelDatabaseRetrieval: public ModelDatabase{
public:


	virtual void add(reglib::Model * model);
	virtual bool remove(reglib::Model * model);
	virtual std::vector<reglib::Model *> search(reglib::Model * model, int number_of_matches);

	ModelDatabaseRetrieval();
    ~ModelDatabaseRetrieval();
};

#endif // MODELDATABASERETRIEVAL_H
