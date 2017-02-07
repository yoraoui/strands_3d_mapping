#ifndef reglibDescriptor_H
#define reglibDescriptor_H

#include <Eigen/Dense>


namespace reglib
{
class Descriptor{
public:
    int type;
	int last_update_frame_id;
    std::vector<char> data;

    Descriptor();
    Descriptor( uint64_t data0, uint64_t data1, uint64_t data2,uint64_t data3);//ORB
    Descriptor( float * d, int len);//Sift/Surf
    ~Descriptor();
    virtual void merge(Descriptor & p, double weight = 1);
    virtual void print();
    virtual double distance(Descriptor & p);
};
}

#include "DescriptorORB.h"

#endif
