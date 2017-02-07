#include "weightfunctions/Distribution.h"

namespace reglib
{
Distribution::Distribution(){
	debugg_print = false;
	traincounter = 0;
    minstd = 0;
}
Distribution::~Distribution(){}
void	Distribution::setMinStd(double x){minstd = x;}
void	Distribution::reset(){traincounter = 0;}
void    Distribution::train(std::vector<float> & hist, unsigned int nr_bins){   printf("%s in %s not implemented, stopping\n",__PRETTY_FUNCTION__,__FILE__);exit(0);}
void    Distribution::train(float * hist, unsigned int nr_bins){                printf("%s in %s not implemented, stopping\n",__PRETTY_FUNCTION__,__FILE__);exit(0);}
void    Distribution::update(){                                                 printf("%s in %s not implemented, stopping\n",__PRETTY_FUNCTION__,__FILE__);exit(0);}
double  Distribution::getval(double x){                                         printf("%s in %s not implemented, stopping\n",__PRETTY_FUNCTION__,__FILE__);exit(0);}
double  Distribution::getcdf(double x){                                         printf("%s in %s not implemented, stopping\n",__PRETTY_FUNCTION__,__FILE__);exit(0);}
void    Distribution::print(){                                                  printf("%s in %s not implemented, stopping\n",__PRETTY_FUNCTION__,__FILE__);exit(0);}
void    Distribution::setRegularization(double x){regularization = x;update();}
double  Distribution::getNoise(){return 1;}
void    Distribution::setNoise(double x){                                       printf("%s in %s not implemented, stopping\n",__PRETTY_FUNCTION__,__FILE__);exit(0);}
void    Distribution::getMaxdMind(double & maxd, double & mind, double prob){
	//if(debugg_print){printf("%s in %s\n",__PRETTY_FUNCTION__,__FILE__);}
	double midval = getval(mean);
	double minDist = 0;
	double minDistScore = getval(mean+minDist)/midval;
	double maxDist = 1;
	double maxDistScore = getval(mean+maxDist)/midval;

	//Grow to find interval
	while(maxDistScore > prob){
		minDist = maxDist;
		minDistScore = maxDistScore;
		maxDist *= 2;
		maxDistScore = getval(mean+maxDist)/midval;
	}

	//bisect to opt
	for(unsigned int it = 0; it < 40; it++){
		double midDist = (minDist+maxDist)*0.5;
		double midDistScore = getval(mean+midDist)/midval;//getcdf(mean+midDist);

		if(midDistScore < prob){
			maxDist = midDist;
			maxDistScore = midDistScore;
		}else{
			minDist = midDist;
			minDistScore = midDistScore;
		}
		if(fabs(maxDistScore-minDistScore) < prob*0.01){break;}
	}
	double midDist = (minDist+maxDist)*0.5;
	mind = mean-midDist;
	maxd = mean+midDist;
}

void Distribution::rescale(double mul){}

Distribution * Distribution::clone(){
    Distribution * dist = new Distribution();
    dist->regularization = regularization;
    dist->mean = mean;
    dist->minstd = minstd;
    dist->debugg_print = debugg_print;
    dist->traincounter = traincounter;
    return dist;
}

}
