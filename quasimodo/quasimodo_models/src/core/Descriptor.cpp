#include "../../include/core/Descriptor.h"

namespace reglib
{

int popcount_lauradoux(uint64_t *buf, uint32_t size) {
    const uint64_t* data = (uint64_t*) buf;

    const uint64_t m1	= UINT64_C(0x5555555555555555);
    const uint64_t m2	= UINT64_C(0x3333333333333333);
    const uint64_t m4	= UINT64_C(0x0F0F0F0F0F0F0F0F);
    const uint64_t m8	= UINT64_C(0x00FF00FF00FF00FF);
    const uint64_t m16 = UINT64_C(0x0000FFFF0000FFFF);
    const uint64_t h01 = UINT64_C(0x0101010101010101);

    uint32_t bitCount = 0;
    uint32_t i, j;
    uint64_t count1, count2, half1, half2, acc;
    uint64_t x;
    uint32_t limit30 = size - size % 30;

    // 64-bit tree merging (merging3)
    for (i = 0; i < limit30; i += 30, data += 30) {
        acc = 0;
        for (j = 0; j < 30; j += 3) {
            count1	=	data[j];
            count2	=	data[j+1];
            half1	=	data[j+2];
            half2	=	data[j+2];
            half1	&=	m1;
            half2	= (half2	>> 1) & m1;
            count1 -= (count1 >> 1) & m1;
            count2 -= (count2 >> 1) & m1;
            count1 +=	half1;
            count2 +=	half2;
            count1	= (count1 & m2) + ((count1 >> 2) & m2);
            count1 += (count2 & m2) + ((count2 >> 2) & m2);
            acc		+= (count1 & m4) + ((count1 >> 4) & m4);
        }
        acc = (acc & m8) + ((acc >>	8)	& m8);
        acc = (acc			 +	(acc >> 16)) & m16;
        acc =	acc			 +	(acc >> 32);
        bitCount += (uint32_t)acc;
    }

    for (i = 0; i < size - limit30; i++) {
        x = data[i];
        x =	x			 - ((x >> 1)	& m1);
        x = (x & m2) + ((x >> 2)	& m2);
        x = (x			 +	(x >> 4)) & m4;
        bitCount += (uint32_t)((x * h01) >> 56);
    }
    return bitCount;
}

Descriptor::Descriptor(){
    type = -1;
    last_update_frame_id = 0;
}

Descriptor::~Descriptor(){}

Descriptor::Descriptor( uint64_t data0, uint64_t data1, uint64_t data2,uint64_t data3){
    type = 0;
    last_update_frame_id = 0;
    data.resize(4*sizeof(long));
    uint64_t * ldata = (uint64_t *)(&data[0]);
    ldata[0] = data0;
    ldata[1] = data1;
    ldata[2] = data2;
    ldata[3] = data3;
}

Descriptor::Descriptor( float * d, int len){
    type = 1;
    last_update_frame_id = 0;
    data.resize(len*sizeof(float));
    float * fdata = (float *)(&data[0]);
    for(int i = 0; i < len; i++){
        fdata[i] = d[i];
    }
}

void Descriptor::merge(Descriptor & p, double weight){
    last_update_frame_id = std::max(p.last_update_frame_id,last_update_frame_id);
    data = p.data;
}
void Descriptor::print(){}
double Descriptor::distance(Descriptor & p){
    if(type == p.type){
        switch(type) {
            case 0:  {//ORB
                uint64_t * ldata = (uint64_t *)(&data[0]);
                uint64_t * pldata = (uint64_t *)(&(p.data)[0]);
                uint64_t xordata [4];
                xordata[0] = ldata[0] ^ pldata[0];
                xordata[1] = ldata[1] ^ pldata[1];
                xordata[2] = ldata[2] ^ pldata[2];
                xordata[3] = ldata[3] ^ pldata[3];

                int cnt = popcount_lauradoux(xordata, 4);
                return double(cnt)/256.0f;
            }break;
            case 1:  {
                int len1 = data.size()/sizeof(float);
                int len2 = p.data.size()/sizeof(float);
                if(len1 != len2){return -1;}

                float * ldata = (float *)(&data[0]);
                float * pldata = (float *)(&(p.data)[0]);
                double diff = 0;
                for(int i = 0; i < len1; i++){
                    double d = ldata[i]-pldata[i];
                    diff += d*d;
                }
                return diff;
            }break;
            default:    {return -1;}					break;
        }
    }
    return -1;
}

}

