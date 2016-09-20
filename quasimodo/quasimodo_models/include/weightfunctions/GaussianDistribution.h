#ifndef GaussianDistribution_H
#define GaussianDistribution_H

#include <vector>
#include <stdio.h>
#include <stdlib.h> 
#include <Eigen/Dense>
#include "Distribution.h"

using namespace Eigen;
namespace reglib
{

class GaussianDistribution  : public Distribution
{
public:

    const double step_h = 0.00001;
    const unsigned int step_iter = 25;
    const double cutoff_exp = -14;

    double mul;
    double stdval;
    double scaledinformation;

    bool refine_mean;
    bool refine_mul;
    bool refine_std;

    double costpen;
    bool zeromean;
    int nr_refineiters;

    GaussianDistribution(bool refine_std_ = true, bool zeromean = true, bool refine_mean = false, bool refine_mul = false, double costpen_ = 3,int nr_refineiters_ = 1,double mul_ = 1, double mean_ = 0,	double stdval_ = 1);
    //GaussianDistribution(double mul_ = 1, double mean_ = 0,	double stdval_ = 1);
    ~GaussianDistribution();
    virtual void train(std::vector<float> & hist, unsigned int nr_bins = 0);
    virtual void update();
    virtual double getNoise();
    virtual void setNoise(double x);
    virtual double getval(double x);
    virtual double getcdf(double x);
    virtual void print();

    //virtual void getMaxdMind(double & maxd, double & mind, double prob = 0.1);

    double scoreCurrent2(double mul, double mean, double stddiv,  std::vector<float> & X, std::vector<float> & Y, unsigned int nr_data, double costpen);
    double fitStdval2   (double mul, double mean, double std_mid, std::vector<float> & X, std::vector<float> & Y, unsigned int nr_data, double costpen);
    double fitMean2     (double mul, double mean, double std_mid, std::vector<float> & X, std::vector<float> & Y, unsigned int nr_data, double costpen);
    double fitMul2      (double mul, double mean, double std_mid, std::vector<float> & X, std::vector<float> & Y, unsigned int nr_data, double costpen);

};

}

#endif // GaussianDistribution
