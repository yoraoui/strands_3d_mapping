#ifndef reglibDescriptorExtractor_H
#define reglibDescriptorExtractor_H

#include <Eigen/Dense>

#include "KeyPoint.h"
#include "RGBDFrame.h"
#include "../model/ModelMask.h"

namespace reglib
{
class DescriptorExtractor{
public:
    int debugg_lvl;
    DescriptorExtractor();
    ~DescriptorExtractor();
    virtual std::vector<KeyPoint> extract(RGBDFrame * frame, ModelMask * mask = 0);
    virtual void setDebugg(int lvl_);
};
}

#include "DescriptorExtractorORB.h"
#include "DescriptorExtractorSURF.h"

#endif
