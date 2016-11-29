#ifndef ModelStorageFile_H
#define ModelStorageFile_H

#include "ModelStorage.h"


class ModelStorageFile: public ModelStorage{
	public:
	std::string filepath;
	std::string framepath;
	virtual std::string getNextID();
	virtual bool add(reglib::Model * model);
	virtual bool remove(reglib::Model * model);
	ModelStorageFile(std::string filepath_ = "./");
	~ModelStorageFile();
};



#endif // ModelStorageFile_H
