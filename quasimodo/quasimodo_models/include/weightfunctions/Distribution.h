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
	double power;
    double minstd;
    bool debugg_print;
    int traincounter;
    double costpen;
    double ratio_costpen;
    std::string name;


    Distribution();
    ~Distribution();

    virtual Distribution * clone();

    virtual void reset();
    virtual void train(std::vector<float> & hist, unsigned int nr_bins = 0);
    virtual void train(float * hist, unsigned int nr_bins);


    virtual void setStart(float * dist, unsigned int nr_data = 0, unsigned int nr_dim = 1);

    virtual void update();

    virtual double getNoise();
    virtual void setNoise(double x);

	virtual double getval(double x);
	virtual double getIRLSreweight(double x);
    virtual double getcdf(double x);
    virtual void setRegularization(double x);
    virtual void setMinStd(double x);
	virtual void rescale(double mul);
    virtual void print();
    virtual void getMaxdMind(double & maxd, double & mind, double prob = 0.001);
    double getDiffScore(std::vector<double> & diffs, std::vector<double> & ratios, double ratioweight);
};

}

#include "GaussianDistribution.h"
#include "GeneralizedGaussianDistribution.h"
#endif // Distribution
