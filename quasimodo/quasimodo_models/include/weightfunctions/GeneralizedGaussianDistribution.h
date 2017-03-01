#ifndef GeneralizedGaussianDistribution_H
#define GeneralizedGaussianDistribution_H

#include <vector>
#include <stdio.h>
#include <stdlib.h> 
#include <Eigen/Dense>
#include "Distribution.h"

using namespace Eigen;
namespace reglib
{

class GeneralizedGaussianDistribution  : public Distribution
{
public:

    const double step_h = 0.00001;
    const unsigned int step_iter = 25;
    const double cutoff_exp = -14;

    double mul;
    double stdval;
    double scaledinformation;
	//double power;
	double precision;

    bool refine_mean;
    bool refine_mul;
    bool refine_std;
    bool refine_power;

    bool zeromean;
    int nr_refineiters;

    double start;
    double stop;
    std::vector<float> numcdf_vec;

    GeneralizedGaussianDistribution(bool refine_std_ = true, bool refine_power_ = false, bool zeromean = true, bool refine_mean = false, bool refine_mul = false, double costpen_ = 3,int nr_refineiters_ = 1,double mul_ = 1, double mean_ = 0,	double stdval_ = 1, double power_ = 2);
    //GeneralizedGaussianDistribution(double mul_ = 1, double mean_ = 0,	double stdval_ = 1);
    ~GeneralizedGaussianDistribution();


    virtual Distribution * clone();
    virtual void train(std::vector<float> & hist, unsigned int nr_bins = 0);
    virtual void train(float * hist, unsigned int nr_bins);
    virtual void update();
	virtual double getInp(double x = 0);
    virtual double getNoise();
	virtual void setNoise(double x = 0);
	virtual double getval(double x = 0);
	virtual double getcdf(double x = 0);
    virtual void print();
	virtual void rescale(double mul);

    //virtual void getMaxdMind(double & maxd, double & mind, double prob = 0.1);

    double scoreCurrent3(double mul, double mean, double stddiv,  double power, std::vector<float> & X, std::vector<float> & Y, unsigned int nr_data, double costpen);
    double fitStdval3   (double mul, double mean, double std_mid, double power, std::vector<float> & X, std::vector<float> & Y, unsigned int nr_data, double costpen);
    double fitMean3     (double mul, double mean, double std_mid, double power, std::vector<float> & X, std::vector<float> & Y, unsigned int nr_data, double costpen);
    double fitMul3      (double mul, double mean, double std_mid, double power, std::vector<float> & X, std::vector<float> & Y, unsigned int nr_data, double costpen);
    double fitPower3    (double mul, double mean, double std_mid, double power, std::vector<float> & X, std::vector<float> & Y, unsigned int nr_data, double costpen);

    double scoreCurrent3(double mul, double mean, double stddiv,  double power, float * X, float * Y, unsigned int nr_data, double costpen);
    double fitStdval3   (double mul, double mean, double std_mid, double power, float * X, float * Y, unsigned int nr_data, double costpen);
    double fitMean3     (double mul, double mean, double std_mid, double power, float * X, float * Y, unsigned int nr_data, double costpen);
    double fitMul3      (double mul, double mean, double std_mid, double power, float * X, float * Y, unsigned int nr_data, double costpen);
    double fitPower3    (double mul, double mean, double std_mid, double power, float * X, float * Y, unsigned int nr_data, double costpen);

    virtual void update_numcdf_vec(unsigned int bins = 1000, double prob = 0.000001);
    //double interp(double x);

	virtual double getIRLSreweight(double x);
};

}

#endif // GeneralizedGaussianDistribution
