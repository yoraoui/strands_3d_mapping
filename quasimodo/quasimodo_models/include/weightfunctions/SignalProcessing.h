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
    virtual void process(std::vector<float> & in, std::vector<float> & out, double stdval, unsigned int nr_bins = 0);
};

}

#endif // SignalProcessing
