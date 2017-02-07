#include "../../include/core/DescriptorORB.h"

namespace reglib
{


DescriptorORB::DescriptorORB(uint64_t data0_, uint64_t data1_, uint64_t data2_,uint64_t data3_){
    data0 = data0_;
    data1 = data1_;
    data2 = data2_;
    data3 = data3_;
    type = 0;
    last_update_frame_id = 0;
}
DescriptorORB::~DescriptorORB(){}

void DescriptorORB::merge(Descriptor & p, double weight){last_update_frame_id = std::max(p.last_update_frame_id,last_update_frame_id);}
void DescriptorORB::print(){}
double DescriptorORB::distance(DescriptorORB & p_){
    if(p_.type == 0){
        DescriptorORB & p = (DescriptorORB &)p_;


    }else{
        if(rand()%100 == 0){printf("Type fail\n");}
        return -1;
    }
}

}

