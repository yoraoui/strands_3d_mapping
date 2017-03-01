#ifndef DistanceWeightFunction2Tdist_H
#define DistanceWeightFunction2Tdist_H

#include <cmath>
#include <sys/time.h>
#include "DistanceWeightFunction2.h"

#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "ceres/iteration_callback.h"
#include "gflags/gflags.h"
#include "glog/logging.h"

using namespace Eigen;
namespace reglib{

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

class DistanceWeightFunction2Tdist : public DistanceWeightFunction2
{
public:
    double info;
	double meanval;
    double v;

    virtual DistanceWeightFunction2 * clone();

    DistanceWeightFunction2Tdist(double v_ = 5);
    ~DistanceWeightFunction2Tdist();

    virtual void computeModel(double * vec, unsigned int nr_data, unsigned int dim = 1);
	virtual void computeModel(MatrixXd mat);
	virtual VectorXd getProbs(MatrixXd mat);
    virtual double getProb(double d, bool debugg = false);
    virtual double getNoise();
	virtual std::string getString();
};

}

#endif // DistanceWeightFunction2Tdist_H
