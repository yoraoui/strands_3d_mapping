#ifndef DistanceWeightFunction2JointDist_H
#define DistanceWeightFunction2JointDist_H

#include <cmath>
#include <sys/time.h>
#include "DistanceWeightFunction2.h"

#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "ceres/iteration_callback.h"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "weightfunctions/Distribution.h"
#include "core/Util.h"
#include "weightfunctions/SignalProcessing.h"


using namespace Eigen;
namespace reglib{

class DistanceWeightFunction2JointDist : public DistanceWeightFunction2
{
public:
    virtual DistanceWeightFunction2 * clone();

	DistanceWeightFunction2JointDist();
	~DistanceWeightFunction2JointDist();

    virtual void computeModel(double * vec, unsigned int nr_data, unsigned int dim = 1);
	virtual void computeModel(MatrixXd mat);
	virtual VectorXd getProbs(MatrixXd mat);
	virtual double getProb(double d, bool debugg = false);
	virtual double getProbInp(double d, bool debugg = false);
	virtual double getIRLS(double d, bool debugg = false);
	virtual double getProbInfront(double d, bool debugg = false);
	virtual double getNoise();
	virtual double getConvergenceThreshold();
	virtual bool update();
	virtual void reset();
	virtual std::string getString();
	virtual double getWeight(double invstd, double d,double & infoweight, double & prob, bool debugg = false);
    virtual VectorXd getWeights(std::vector<double > invstd, MatrixXd mat, bool debugg = false);
    virtual void print();
};

}

#endif // DistanceWeightFunction2JointDisttest_H
