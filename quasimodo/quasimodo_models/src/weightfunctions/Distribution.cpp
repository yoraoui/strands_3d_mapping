#include "weightfunctions/Distribution.h"

namespace reglib
{
Distribution::Distribution(){
	debugg_print = false;
	traincounter = 0;
    minstd = 0;
	power = 2;
    costpen = 3;
    ratio_costpen = 10;
    name = "undefined";
}
Distribution::~Distribution(){}

void    Distribution::setStart(float * dist, unsigned int nr_data, unsigned int nr_dim){

}

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
        if(fabs(maxDistScore-minDistScore) < prob*0.001){break;}
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
	dist->power = power;
    dist->name = name;
    return dist;
}

double Distribution::getIRLSreweight(double x){return 1.0;}

double Distribution::getDiffScore(std::vector<double> & diffs, std::vector<double> & ratios, double mul){

    double sum = 0;
    unsigned int nr_data = diffs.size();
    if(costpen > 0){
        for(unsigned int i = 0; i < nr_data; i++){
            double diff = diffs[i];
            if(diff < 0){	sum -= costpen*diff;}
            else{			sum += diff;}
        }
    }else{
        for(unsigned int i = 0; i < nr_data; i++){
            double diff = diffs[i];
            if(diff < 0){	sum += 1.0*pow(-diff,1.25);}
            else{			sum += diff;}
        }
    }

    double ratioweight = ratio_costpen*mul;

    //small - large = bad
    double ratiosum = 0;
    for(unsigned int i = 1; i < nr_data; i++){
        double ratiodiff;
        if(mean < i){
            ratiodiff = ratios[i]-ratios[i-1];
        }else{
            ratiodiff = ratios[i-1]-ratios[i];
        }
        double ratiocost = 0;
        if(ratiodiff > 0){	ratiocost = ratiodiff*ratioweight;}
        ratiosum += ratiocost;
    }

//    printf("costpen: %f ",costpen);
//    printf("sum: %f ratiosum: %f\n",sum,ratiosum);

    return sum+ratiosum;
}

}
