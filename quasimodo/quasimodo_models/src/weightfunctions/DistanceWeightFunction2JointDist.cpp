#include "weightfunctions/DistanceWeightFunction2JointDist.h"

#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>


namespace reglib{

DistanceWeightFunction2JointDist::DistanceWeightFunction2JointDist(){}
DistanceWeightFunction2JointDist::~DistanceWeightFunction2JointDist(){}
double DistanceWeightFunction2JointDist::getNoise(){}

void DistanceWeightFunction2JointDist::recomputeHistogram(std::vector<float> & hist, MatrixXd & mat){}
void DistanceWeightFunction2JointDist::recomputeHistogram(std::vector<float> & hist, double * vec, unsigned int nr_data){}
void DistanceWeightFunction2JointDist::computeModel(double * vec, unsigned int nr_data, unsigned int nr_dim){}
void DistanceWeightFunction2JointDist::computeModel(MatrixXd mat){}
VectorXd DistanceWeightFunction2JointDist::getProbs(MatrixXd mat){
	const unsigned int nr_data = mat.cols();
	const int nr_dim = mat.rows();
	VectorXd weights = VectorXd(nr_data);
	return weights;
}

double DistanceWeightFunction2JointDist::getProb(double d, bool debugg){
	float p = 0;
	return p;
}


double DistanceWeightFunction2JointDist::getProbInfront(double d, bool debugg){}

bool DistanceWeightFunction2JointDist::update(){
	return true;
}

void DistanceWeightFunction2JointDist::reset(){}

std::string DistanceWeightFunction2JointDist::getString(){
	return std::string(DistanceWeightFunction2JointDist);
}

double DistanceWeightFunction2JointDist::getConvergenceThreshold(){
	return convergence_threshold;
}

DistanceWeightFunction2 * DistanceWeightFunction2JointDist::clone(){
    DistanceWeightFunction2 * func = new DistanceWeightFunction2();
    return func;
}

}


