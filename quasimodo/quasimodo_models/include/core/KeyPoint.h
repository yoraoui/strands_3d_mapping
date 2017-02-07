#ifndef reglibKeyPoint_H
#define reglibKeyPoint_H

#include <Eigen/Dense>

#include "Descriptor.h"
#include "superpoint.h"

namespace reglib
{
class KeyPoint{
public:
    superpoint point;
    Descriptor descriptor;

    KeyPoint(superpoint point = superpoint(),Descriptor descriptor = Descriptor());
    ~KeyPoint();
    virtual void merge(KeyPoint & p, double weight = 1);
};
}

#endif
