#ifndef SignalProcessing_H
#define SignalProcessing_H

#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <Eigen/Dense>

using namespace Eigen;
namespace reglib
{

class SignalProcessing
{
public:
    SignalProcessing();
    ~SignalProcessing();
    virtual SignalProcessing * clone();
    virtual void process(std::vector<float> & in, std::vector<float> & out, double stdval, unsigned int nr_bins = 0);
    virtual void process(float * in, float * out, double stdval, unsigned int nr_bins);
};

}

#endif // SignalProcessing
