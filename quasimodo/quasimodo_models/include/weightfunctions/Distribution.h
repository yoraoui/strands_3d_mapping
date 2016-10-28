#ifndef Distribution_H
#define Distribution_H

#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <Eigen/Dense>

using namespace Eigen;
namespace reglib
{

class Distribution
{
public:
    double regularization;
    double mean;
    bool debugg_print;


    Distribution();
    ~Distribution();
	virtual void reset();
    virtual void train(std::vector<float> & hist, unsigned int nr_bins = 0);
    virtual void update();

    virtual double getNoise();
    virtual void setNoise(double x);

    virtual double getval(double x);
    virtual double getcdf(double x);
    virtual void setRegularization(double x);
    virtual void print();
    virtual void getMaxdMind(double & maxd, double & mind, double prob = 0.0001);
};

}

#include "GaussianDistribution.h"
#include "GeneralizedGaussianDistribution.h"
#endif // Distribution
