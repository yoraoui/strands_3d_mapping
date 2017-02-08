#include "registration/MassRegistrationRecursive.h"

#include <iostream>
#include <fstream>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace reglib
{

MassRegistrationRecursive::MassRegistrationRecursive(MassRegistration * massreg_, double per_split_){
	massreg = massreg_;
	per_split = per_split_;
}

MassRegistrationRecursive::~MassRegistrationRecursive(){}

void MassRegistrationRecursive::addModel(Model * model){
	models.push_back(model);
}


MassFusionResults MassRegistrationRecursive::getTransforms(std::vector<Eigen::Matrix4d> poses, double start, double stop){
	double len = stop-start;
	if(len <= per_split){
		//printf("LEAF: start: %f stop: %f len: %f\n",start,stop,len);
	}else{
		double step = std::max(1.0,len/per_split);
		//printf("+++++++++++++start: %f stop: %f len: %f per_split: %f step: %f\n",start,stop,len,per_split,step);

		for(double i = start; std::round(i) < stop; i += step){
			//printf("--->start: %f stop: %f -> %f\n",i,i+step,std::min(stop,i+step));
			MassFusionResults mfr = getTransforms(poses,std::round(i),std::round(std::min(stop,i+step)));
		}

		//printf("-------------start: %f stop: %f len: %f per_split: %f step: %f\n",start,stop,len,per_split,step);
	}
	return MassFusionResults(poses,-1);
}

MassFusionResults MassRegistrationRecursive::getTransforms(std::vector<Eigen::Matrix4d> poses){
	printf("MassRegistrationRecursive\n");
	return getTransforms(poses,0,poses.size());
}

}
