#ifndef reglibDescriptorORB_H
#define reglibDescriptorORB_H

#include <Eigen/Dense>
#include "Descriptor.h"

namespace reglib
{
class DescriptorORB  : public Descriptor
{
public:

    uint64_t data0;
    uint64_t data1;
    uint64_t data2;
    uint64_t data3;

    DescriptorORB( uint64_t data0, uint64_t data1, uint64_t data2,uint64_t data3);
    ~DescriptorORB();
    virtual void merge(Descriptor & p, double weight = 1);
    virtual void print();
    virtual double distance(DescriptorORB & p);
};
}

#endif
