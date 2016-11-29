#ifndef ModelStorageFile_H
#define ModelStorageFile_H

#include "ModelStorage.h"


class ModelStorageFile: public ModelStorage{
	public:
	std::string filepath;
	virtual bool add(reglib::Model * model, std::string parrentpath = "");
	virtual bool remove(reglib::Model * model);
	ModelStorageFile(std::string filepath_ = "./");
	~ModelStorageFile();
};



#endif // ModelStorageFile_H
