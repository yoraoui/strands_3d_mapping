#include "ModelStorage.h"

ModelStorage::ModelStorage(){}
ModelStorage::~ModelStorage(){}

std::string ModelStorage::getNextID(){return "";}

bool ModelStorage::add(reglib::Model * model){return true;}
bool ModelStorage::remove(reglib::Model * model){return false;}
		
