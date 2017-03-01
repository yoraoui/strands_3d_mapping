#include "weightfunctions/GaussianDistribution.h"

namespace reglib
{


void GaussianDistribution::setStart(float * vec, unsigned int nr_data, unsigned int nr_dim){
    unsigned int nr_inds = nr_data*nr_dim;
    double sum = 0;
    for(unsigned int j = 0; j < nr_inds; j++){sum += vec[j]*vec[j];}
    double var = sum / double(nr_inds-1);
    stdval = sqrt(var);
}

GaussianDistribution::GaussianDistribution(bool refine_std_, bool zeromean_, bool refine_mean_, bool refine_mul_, double costpen_, int nr_refineiters_ ,double mul_, double mean_,double stdval_){
    refine_std  = refine_std_;
    zeromean    = zeromean_;
    refine_mean = refine_mean_;
    refine_mul  = refine_mul_;
    costpen     = costpen_;
    nr_refineiters = nr_refineiters_;
	traincounter = 0;
    minstd      = 0;
	power = 2;

    mul         = mul_;
    mean        = mean_;
    stdval      = stdval_;
    update();
    debugg_print = false;

    ratio_costpen = 10;

    name = "gaussian";
}
//GaussianDistribution::GaussianDistribution(double mul_, double mean_,	double stdval_){
//    mul = mul_;
//    mean = mean_;
//    stdval = stdval_;
//    update();
//}
GaussianDistribution::~GaussianDistribution(){}

double GaussianDistribution::scoreCurrent2(double mul, double mean, double stddiv, std::vector<float> & X, std::vector<float> & Y, unsigned int nr_data, double costpen){
    double info = -0.5/(stddiv*stddiv);
    double sum = 0;
    for(unsigned int i = 0; i < nr_data; i++){
        double dx = X[i] - mean;
        double inp = info*dx*dx;
        if(inp < cutoff_exp){sum += Y[i];}
        else{
            double diff = mul*exp(info*dx*dx) - Y[i];
            if(diff > 0){	sum += costpen*diff;}
            else{			sum -= diff;}
        }
    }
    return sum;
}

double GaussianDistribution::fitStdval2(double mul, double mean, double std_mid, std::vector<float> & X, std::vector<float> & Y, unsigned int nr_data, double costpen){
    int iter = 25;
    double h = 0.000000001;

    double std_max = std_mid*2;
    double std_min = 0;
    for(int i = 0; i < iter; i++){
        std_mid = (std_max+std_min)/2;
        double std_neg = scoreCurrent2(mul,mean,std_mid-h,X,Y,nr_data,costpen);
        double std_pos = scoreCurrent2(mul,mean,std_mid+h,X,Y,nr_data,costpen);
        if(std_neg < std_pos){	std_max = std_mid;}
        else{					std_min = std_mid;}
    }
    return std_mid;
}

double GaussianDistribution::fitMean2(double mul, double mean, double std_mid, std::vector<float> & X, std::vector<float> & Y, unsigned int nr_data, double costpen){
    int iter = 10;
    double h = 0.000000001;
    double mean_max = mean+1;
    double mean_min = mean-1;

    //	double mean_max = mean*2;
    //	double mean_min = 0;

    for(int i = 0; i < iter; i++){
        mean = (mean_max+mean_min)/2;
        double std_neg = scoreCurrent2(mul,mean-h,std_mid,X,Y,nr_data,costpen);
        double std_pos = scoreCurrent2(mul,mean+h,std_mid,X,Y,nr_data,costpen);
        if(std_neg < std_pos){	mean_max = mean;}
        else{					mean_min = mean;}
    }
    return mean;
}

double GaussianDistribution::fitMul2(double mul, double mean, double std_mid, std::vector<float> & X, std::vector<float> & Y, unsigned int nr_data, double costpen){
    int iter = 25;
    double h = 0.000000001;
    double mul_max = mul*2;
    double mul_min = 0;

    for(int i = 0; i < iter; i++){
        mul = (mul_max+mul_min)/2;
        double std_neg = scoreCurrent2(mul-h,mean,std_mid,X,Y,nr_data,costpen);
        double std_pos = scoreCurrent2(mul+h,mean,std_mid,X,Y,nr_data,costpen);
        if(std_neg < std_pos){	mul_max = mul;}
        else{					mul_min = mul;}
    }
    return mul;
}

//X Y


double GaussianDistribution::scoreCurrent2(double mul, double mean, double stddiv, float * X, float * Y, unsigned int nr_data, double costpen){
    double info = -0.5/(stddiv*stddiv);
    double sum = 0;
    for(unsigned int i = 0; i < nr_data; i++){
        //double dx = X[i] - mean;
        double dx = i - mean;
        double inp = info*dx*dx;
        if(inp < cutoff_exp){sum += Y[i];}
        else{
            double diff = mul*exp(info*dx*dx) - Y[i];
            if(diff > 0){	sum += costpen*diff;}
            else{			sum -= diff;}
        }
    }
    return sum;
}

double GaussianDistribution::fitStdval2(double mul, double mean, double std_mid, float * X, float * Y, unsigned int nr_data, double costpen){
    int iter = 25;
    double h = 0.000000001;

    double std_max = std_mid*2;
    double std_min = 0;
    for(int i = 0; i < iter; i++){
        std_mid = (std_max+std_min)/2;
        double std_neg = scoreCurrent2(mul,mean,std_mid-h,X,Y,nr_data,costpen);
        double std_pos = scoreCurrent2(mul,mean,std_mid+h,X,Y,nr_data,costpen);
        if(std_neg < std_pos){	std_max = std_mid;}
        else{					std_min = std_mid;}
    }

    return std_mid;
}

double GaussianDistribution::fitMean2(double mul, double mean, double std_mid, float * X, float * Y, unsigned int nr_data, double costpen){
    int iter = 10;
    double h = 0.000000001;
    double mean_max = mean+1;
    double mean_min = mean-1;

    //	double mean_max = mean*2;
    //	double mean_min = 0;

    for(int i = 0; i < iter; i++){
        mean = (mean_max+mean_min)/2;
        double std_neg = scoreCurrent2(mul,mean-h,std_mid,X,Y,nr_data,costpen);
        double std_pos = scoreCurrent2(mul,mean+h,std_mid,X,Y,nr_data,costpen);
        if(std_neg < std_pos){	mean_max = mean;}
        else{					mean_min = mean;}
    }
    return mean;
}

double GaussianDistribution::fitMul2(double mul, double mean, double std_mid, float * X, float * Y, unsigned int nr_data, double costpen){
    int iter = 25;
    double h = 0.000000001;
    double mul_max = mul*2;
    double mul_min = 0;

    for(int i = 0; i < iter; i++){
        mul = (mul_max+mul_min)/2;
        double std_neg = scoreCurrent2(mul-h,mean,std_mid,X,Y,nr_data,costpen);
        double std_pos = scoreCurrent2(mul+h,mean,std_mid,X,Y,nr_data,costpen);
        if(std_neg < std_pos){	mul_max = mul;}
        else{					mul_min = mul;}
    }
    return mul;
}

void GaussianDistribution::train(std::vector<float> & hist, unsigned int nr_bins){
    if(nr_bins == 0){nr_bins = hist.size();}
    mul = hist[0];
    mean = 0;
    if(!zeromean){
        for(unsigned int k = 1; k < nr_bins; k++){
            if(hist[k] > mul){
                mul = hist[k];
                mean = k;
            }
        }
    }

    std::vector<float> X;
    std::vector<float> Y;
    for(unsigned int k = 0; k < nr_bins; k++){
        if(hist[k]  > mul*0.01){
            X.push_back(k);
            Y.push_back(hist[k]);
        }
    }

    unsigned int nr_data_opt = X.size();


    double ysum = 0;
    for(unsigned int i = 0; i < nr_data_opt; i++){ysum += fabs(Y[i]);}

    double std_mid = 0;
    for(unsigned int i = 0; i < nr_data_opt; i++){std_mid += (X[i]-mean)*(X[i]-mean)*fabs(Y[i])/ysum;}
    stdval = sqrt(std_mid);

    for(int i = 0; i < nr_refineiters; i++){
        if(refine_std){		stdval	= fitStdval2(	mul,mean,stdval,X,Y,nr_data_opt,costpen);}
        if(refine_mean){	mean	= fitMean2(		mul,mean,stdval,X,Y,nr_data_opt,costpen);}
        if(refine_mul){		mul		= fitMul2(		mul,mean,stdval,X,Y,nr_data_opt,costpen);}
    }
    stdval = std::max(stdval,minstd);
	if(debugg_print){print();}
}

void GaussianDistribution::train(float * hist, unsigned int nr_bins){
    mul = hist[0];
    mean = 0;
    if(!zeromean){
        for(unsigned int k = 1; k < nr_bins; k++){
            if(hist[k] > mul){
                mul = hist[k];
                mean = k;
            }
        }
    }

    unsigned int nr_data_opt = nr_bins;
//    float * X = new float[nr_bins];
//    float * Y = new float[nr_bins];
//    for(unsigned int k = 0; k < nr_bins; k++){
//        //if(hist[k]  > mul*0.01){
//        if(true){
//            X[nr_data_opt] = k;
//            Y[nr_data_opt] = hist[k];
//            nr_data_opt++;
//        }
//    }

//    double ysum = 0;
//    for(unsigned int i = 0; i < nr_data_opt; i++){ysum += fabs(Y[i]);}

//    double std_mid = 0;
//    for(unsigned int i = 0; i < nr_data_opt; i++){std_mid += (X[i]-mean)*(X[i]-mean)*fabs(Y[i])/ysum;}
//    stdval = sqrt(std_mid);


    double ysum = 0;
    for(unsigned int i = 0; i < nr_data_opt; i++){ysum += fabs(hist[i]);}

    double std_mid = 0;
    for(unsigned int i = 0; i < nr_data_opt; i++){std_mid += (i-mean)*(i-mean)*fabs(hist[i])/ysum;}
    stdval = sqrt(std_mid);

    for(int i = 0; i < nr_refineiters; i++){
        if(refine_std){		stdval	= fitStdval2(	mul,mean,stdval,0,hist,nr_data_opt,costpen);}
        if(refine_mean){	mean	= fitMean2(		mul,mean,stdval,0,hist,nr_data_opt,costpen);}
        if(refine_mul){		mul		= fitMul2(		mul,mean,stdval,0,hist,nr_data_opt,costpen);}
    }
    stdval = std::max(stdval,minstd);
//    delete[] X;
//    delete[] Y;

	if(debugg_print){print();}
}

void GaussianDistribution::update(){
    scaledinformation = -0.5/((stdval+regularization)*(stdval+regularization));
}

double GaussianDistribution::getval(double x){
    double dx = mean-x;
    return mul*exp(dx*dx*scaledinformation);
}
double GaussianDistribution::getcdf(double x){
    return 0.5 * erfc(-((x-mean)/(stdval+regularization)) * M_SQRT1_2);
}

void GaussianDistribution::setNoise(double x){
    stdval = x;
    update();
}

void GaussianDistribution::print(){
    printf("GaussianDistribution:: mul = %5.5f mean = %5.5f stdval = %5.5f reg = %5.5f\n",mul,mean,stdval,regularization);
}

double GaussianDistribution::getNoise(){return stdval+regularization;}

void GaussianDistribution::rescale(double mul){stdval *= mul;}

void GaussianDistribution::getMaxdMind(double & maxd, double & mind, double prob){
    double var = getNoise()*getNoise();
    double midDist = sqrt(log(prob)/(-0.5/var));
    mind = mean-midDist;
    maxd = mean+midDist;
}


Distribution * GaussianDistribution::clone(){
    //printf("Distribution * GaussianDistribution::clone()\n");
    GaussianDistribution * dist = new GaussianDistribution();
    dist->regularization = regularization;
    dist->mean = mean;
    dist->minstd = minstd;
    dist->debugg_print = debugg_print;
    dist->traincounter = traincounter;
    dist->mul = mul;
    dist->stdval = stdval;
    dist->scaledinformation = scaledinformation;
    dist->refine_mean = refine_mean;
    dist->refine_mul = refine_mul;
    dist->refine_std = refine_std;
    dist->costpen = costpen;
    dist->zeromean = zeromean;
    dist->nr_refineiters = nr_refineiters;
	dist->power = power;
    dist->name = name;
    return dist;
}

}

