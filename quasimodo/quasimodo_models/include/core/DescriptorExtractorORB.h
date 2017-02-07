#ifndef reglibDescriptorExtractorORB_H
#define reglibDescriptorExtractorORB_H

#include <Eigen/Dense>

#include "DescriptorExtractor.h"


#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"

namespace reglib
{
class DescriptorExtractorORB  : public DescriptorExtractor
{
public:
    cv::ORB orb;

    DescriptorExtractorORB();
    ~DescriptorExtractorORB();
    virtual std::vector<KeyPoint> extract(RGBDFrame * frame, ModelMask * mask = 0);
};
}

#endif
