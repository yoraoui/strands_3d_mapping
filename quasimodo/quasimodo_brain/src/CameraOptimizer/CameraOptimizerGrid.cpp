#include "CameraOptimizerGrid.h"


CameraOptimizerGrid::CameraOptimizerGrid(){bias = 100;}
CameraOptimizerGrid::~CameraOptimizerGrid(){}

void CameraOptimizerGrid::shrink(){
    for(unsigned long i = 0; i < mul.size();i++){
        mul[i] *= 0.5;
        sum[i] *= 0.5;

        if(sum[i] < bias){
            mul[i] = bias*mul[i]/sum[i];
            sum[i] = bias;
        }
    }
}

void CameraOptimizerGrid::normalize(){
    double totsum = 0;
    for(unsigned long i = 0; i < mul.size();i++){
        totsum += mul[i]/sum[i];
    }
    double mean = totsum / double(mul.size());
    for(unsigned long i = 0; i < mul.size();i++){
        mul[i] /= mean;
    }
}

double CameraOptimizerGrid::getMax(){
    double maxval = 1;
    for(unsigned long i = 0; i < mul.size();i++){
        maxval = std::max(maxval,mul[i]/sum[i]);
    }
    return maxval;
}
double CameraOptimizerGrid::getMin(){
    double minval = 1;
    for(unsigned long i = 0; i < mul.size();i++){
        minval = std::min(minval,mul[i]/sum[i]);
    }
    return minval;
}
