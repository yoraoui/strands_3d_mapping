#ifndef reglibDescriptorExtractorSURF_H
#define reglibDescriptorExtractorSURF_H

#include <Eigen/Dense>

#include "DescriptorExtractor.h"


#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"

namespace reglib
{
class DescriptorExtractorSURF  : public DescriptorExtractor
{
public:
    DescriptorExtractorSURF();
    ~DescriptorExtractorSURF();
    virtual std::vector<KeyPoint> extract(RGBDFrame * frame, ModelMask * mask = 0);
};
}

#endif
