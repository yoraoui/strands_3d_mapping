#include "../../include/core/DescriptorExtractor.h"

namespace reglib
{

DescriptorExtractor::DescriptorExtractor(){debugg_lvl = 0;}
DescriptorExtractor::~DescriptorExtractor(){}

std::vector<KeyPoint> DescriptorExtractor::extract(RGBDFrame * frame, ModelMask * mask){
    std::vector<KeyPoint> kps;
    return kps;
}

void DescriptorExtractor::setDebugg(int lvl_){debugg_lvl = lvl_;}

}

