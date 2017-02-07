#include "../../include/core/KeyPoint.h"

namespace reglib
{

KeyPoint::KeyPoint(superpoint point_,Descriptor descriptor_){
    point = point_;
    descriptor = descriptor_;
}

KeyPoint::~KeyPoint(){}

void KeyPoint::merge(KeyPoint & p, double weight){
    point.merge(p.point,weight);
    descriptor.merge(p.descriptor,weight);
}

}

