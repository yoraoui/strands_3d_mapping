#include "weightfunctions/SignalProcessing.h"

namespace reglib
{

SignalProcessing::SignalProcessing(){}
SignalProcessing::~SignalProcessing(){}
void SignalProcessing::process(std::vector<float> & in, std::vector<float> & out, double stdval, unsigned int nr_bins){
    if(nr_bins == 0){nr_bins = in.size();}


    int nr_bins2 = 2*nr_bins;

    std::vector<float> tmphist;
    tmphist.resize(nr_bins2);

    std::vector<float> tmphist_blur;
    tmphist_blur.resize(nr_bins2);

    for(int i = 0; i < nr_bins; i++){
        tmphist[i]			= in[nr_bins-1-i];
        tmphist[nr_bins+i]	= in[i];
        tmphist_blur[i]			= 0;
        tmphist_blur[nr_bins+i]	= 0;
    }

    double info = -0.5/(stdval*stdval);
    double weights[nr_bins2];
    for(int i = 0; i < nr_bins2; i++){weights[i] = 0;}
    for(int i = 0; i < nr_bins2; i++){
        double current = exp(i*i*info);
        weights[i] = current;
        if(current < 0.001){break;}
    }

    int offset = 4.0*stdval;
    offset = std::max(4,offset);
    for(int i = 0; i < nr_bins2; i++){
        double v = tmphist[i];
        if(v == 0){continue;}

        int start	= std::max(0,i-offset);
        int stop	= std::min(nr_bins2,i+offset+1);

        for(int j = start; j < stop; j++){tmphist_blur[j] += v*weights[abs(i-j)];}
    }
    for(int i = 0; i < nr_bins; i++){out[i] = tmphist_blur[i+nr_bins];}

    float bef = 0;
    float aft = 0;
    for(int i = 0; i < nr_bins; i++){
        bef += in[i];
        aft += out[i];
    }

    for(int i = 0; i < nr_bins; i++){out[i] *= bef/aft;}
}

}
