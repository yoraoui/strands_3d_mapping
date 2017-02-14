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
	if(len == 1){
		return MassFusionResults(poses,-1);
	}else if(len <= per_split){

		printf("+++++++++++++start: %f stop: %f\n",start,stop);

		massreg->clearData();
		std::vector<Eigen::Matrix4d> current_poses;
		for(unsigned int i = start; i < stop; i++){
			current_poses.push_back(poses[start].inverse() * poses[i]);
			massreg->addModel(models[i]);
		}
		MassFusionResults mfr = massreg->getTransforms(current_poses);
		massreg->clearData();

		std::vector<Eigen::Matrix4d> nextposes = poses;
		for(unsigned int i = start; i < stop; i++){
			nextposes[i] = mfr.poses[i-start] * poses[start];
		}
		return MassFusionResults(nextposes,mfr.score);
	}else{
		double step = std::max(1.0,len/per_split);
		printf("+++++++++++++start: %f stop: %f len: %f per_split: %f step: %f\n",start,stop,len,per_split,step);
		//printf("--->start: %f stop: %f -> %f\n",i,i+step,std::min(stop,i+step));

		std::vector<Eigen::Matrix4d> current_poses;
		std::vector<Model *> current_models;
		std::vector<MassFusionResults> subresults;
		for(double i = start; std::round(i) < stop; i += step){
			unsigned int current_start = std::round(i);
			unsigned int current_stop = std::round(std::min(stop,i+step));
			MassFusionResults mfr = getTransforms(poses,current_start,current_stop);
			Model * current_model = new Model();
			current_models.push_back(current_model);
			current_poses.push_back(poses[int(start)].inverse() * poses[current_start]);

			for(unsigned int j = current_start; j < current_stop; j++){
				//poses[j] = mfr.poses[j];
				current_model->submodels_relativeposes.push_back(poses[int(start)].inverse() * poses[j]);
				current_model->submodels.push_back(models[j]);
				models[j]->recomputeModelPoints();
				printf("models[%i]->points.size(): %i\n",j,models[j]->points.size());

			}
			current_model->recomputeModelPoints();
			printf("current_start: %i current_stop: %i current_model->points.size(): %i\n",current_start,current_stop,current_model->points.size());
		}

		massreg->clearData();
		for(unsigned int i = 0; i < current_models.size(); i ++){
			massreg->addModel(current_models[i]);
		}
		MassFusionResults mfr = massreg->getTransforms(current_poses);
		massreg->clearData();

		for(unsigned int i = 0; i < current_models.size(); i++){
			delete current_models[i];
			current_models[i] = 0;
		}
		current_models.clear();

		std::vector<Eigen::Matrix4d> nextposes = poses;
//		unsigned int tmp = 0;
//		for(double i = start; std::round(i) < stop; i += step){
//			unsigned int current_start = std::round(i);
//			unsigned int current_stop = std::round(std::min(stop,i+step));
//			Eigen::Matrix4d change = poses[current_start].inverse() * mfr.poses[tmp];
//			for(unsigned int j = current_start; j < current_stop; j ++){
//				nextposes[j] = change*poses[j];
//			}
//			tmp++;
//		}
		return MassFusionResults(nextposes,mfr.score);
	}
	return MassFusionResults(poses,-1);
}

MassFusionResults MassRegistrationRecursive::getTransforms(std::vector<Eigen::Matrix4d> poses){
	printf("MassRegistrationRecursive\n");
	return getTransforms(poses,0,poses.size());
}

}
