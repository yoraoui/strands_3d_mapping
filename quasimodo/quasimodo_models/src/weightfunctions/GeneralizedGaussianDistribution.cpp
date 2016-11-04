#include "weightfunctions/GeneralizedGaussianDistribution.h"
#include <boost/math/special_functions/gamma.hpp>

namespace reglib
{

GeneralizedGaussianDistribution::GeneralizedGaussianDistribution(bool refine_std_, bool refine_power_, bool zeromean_, bool refine_mean_, bool refine_mul_, double costpen_, int nr_refineiters_ ,double mul_, double mean_,double stdval_,double power_){
    refine_std  = refine_std_;
    zeromean    = zeromean_;
    refine_mean = refine_mean_;
    refine_mul  = refine_mul_;
    refine_power = refine_power_;
    costpen     = costpen_;
    nr_refineiters = nr_refineiters_;

    mul         = mul_;
    mean        = mean_;
    stdval      = stdval_;
    power       = power_;
    update();
    debugg_print = false;
	regularization = 0;
}

GeneralizedGaussianDistribution::~GeneralizedGaussianDistribution(){
	numcdf_vec.clear();
}

double GeneralizedGaussianDistribution::fitMul3(double mul, double mean, double std_mid, double power, std::vector<float> & X, std::vector<float> & Y, unsigned int nr_data, double costpen){
    int iter = 25;
    double h = 0.000000001;
    double mul_max = mul*2;
    double mul_min = 0;

    for(int i = 0; i < iter; i++){
        mul = (mul_max+mul_min)/2;
        double std_neg = scoreCurrent3(mul-h,mean,std_mid,power,X,Y,nr_data,costpen);
        double std_pos = scoreCurrent3(mul+h,mean,std_mid,power,X,Y,nr_data,costpen);
        if(std_neg < std_pos){	mul_max = mul;}
        else{					mul_min = mul;}
    }
    return mul;
}


double GeneralizedGaussianDistribution::scoreCurrent3(double mul, double mean, double stddiv, double power, std::vector<float> & X, std::vector<float> & Y, unsigned int nr_data, double costpen){
    double sum = 0;
    double invstd = 1.0/stddiv;
    for(unsigned int i = 0; i < nr_data; i++){
        double dx = fabs(X[i] - mean)*invstd;
        double inp = -0.5*pow(dx,power);
        if(inp < cutoff_exp){sum += Y[i];}
        else{
            double diff = Y[i]*Y[i]*(mul*exp(inp) - Y[i]);
            if(diff > 0){	sum += costpen*diff;}
            else{			sum -= diff;}
        }
    }
    return sum;
}

double GeneralizedGaussianDistribution::fitStdval3(double mul, double mean, double std_mid, double power, std::vector<float> & X, std::vector<float> & Y, unsigned int nr_data, double costpen){
    int iter = 25;
    double h = 0.000000001;

    double std_max = std_mid*2;
    double std_min = 0;
    for(int i = 0; i < iter; i++){
        std_mid = (std_max+std_min)/2;
        double std_neg = scoreCurrent3(mul,mean,std_mid-h,power,X,Y,nr_data,costpen);
        double std_pos = scoreCurrent3(mul,mean,std_mid+h,power,X,Y,nr_data,costpen);
        if(std_neg < std_pos){	std_max = std_mid;}
        else{					std_min = std_mid;}
    }
    return std_mid;
}

double GeneralizedGaussianDistribution::fitPower3(double current_mul, double current_mean, double current_std, double current_power, std::vector<float> & X, std::vector<float> & Y, unsigned int nr_data, double costpen){
    int iter = 20;
    double h = 0.000000001;

    double power_max = current_power*2;
    double power_min = 0.001;
    for(int i = 0; i < iter; i++){
        current_power = (power_max+power_min)/2;
        double std_neg = scoreCurrent3(current_mul,current_mean,current_std,current_power-h,X,Y,nr_data,costpen);
        double std_pos = scoreCurrent3(current_mul,current_mean,current_std,current_power+h,X,Y,nr_data,costpen);
        if(std_neg < std_pos){	power_max = current_power;}
        else{					power_min = current_power;}
    }
    current_power = (power_max+power_min)/2;
	return current_power;
}

double GeneralizedGaussianDistribution::fitMean3(double mul, double mean, double std_mid, double power, std::vector<float> & X, std::vector<float> & Y, unsigned int nr_data, double costpen){
    int iter = 10;
    double h = 0.000000001;
    double mean_max = mean+10;
    double mean_min = mean-10;

    for(int i = 0; i < iter; i++){
        mean = (mean_max+mean_min)/2;
        double std_neg = scoreCurrent3(mul,mean-h,std_mid,power,X,Y,nr_data,costpen);
        double std_pos = scoreCurrent3(mul,mean+h,std_mid,power,X,Y,nr_data,costpen);
        if(std_neg < std_pos){	mean_max = mean;}
        else{					mean_min = mean;}
    }
    return mean;
}

void GeneralizedGaussianDistribution::train(std::vector<float> & hist, unsigned int nr_bins){
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
        if(hist[k]  > mul*0.001){
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
        if(refine_std){		stdval	= fitStdval3(	mul,mean,stdval,power,X,Y,nr_data_opt,costpen);}
        if(refine_mean){	mean	= fitMean3(		mul,mean,stdval,power,X,Y,nr_data_opt,costpen);}
        if(refine_mul){		mul		= fitMul3(		mul,mean,stdval,power,X,Y,nr_data_opt,costpen);}
        if(refine_power){	power	= fitPower3(	mul,mean,stdval,power,X,Y,nr_data_opt,costpen);}
    }

	//print();
}

void GeneralizedGaussianDistribution::update(){
	//printf("%s in %s\n",__PRETTY_FUNCTION__,__FILE__);
    update_numcdf_vec();
}

double GeneralizedGaussianDistribution::getInp(double x){
	return -0.5*pow(fabs(mean-x)/(stdval+regularization),power);
}

double GeneralizedGaussianDistribution::getval(double x){
    return mul*exp(getInp(x));
}
double GeneralizedGaussianDistribution::getcdf(double x){

    double part = (x-start)/(stop-start);
    if(part < 0){return 0;}
   // if(part >= 1){return 1;}
    part *= double(numcdf_vec.size()-1);
    unsigned int id0 = part;
    unsigned int id1 = id0+1;
	if(id1 >= numcdf_vec.size()){return 1;}
    double w0 = double(id1)-part;
    double w1 = part-double(id0);
    double cdfx = numcdf_vec[id0]*w0 + numcdf_vec[id1]*w1;
    return cdfx;
}

void GeneralizedGaussianDistribution::setNoise(double x){
    stdval = x;
    update();
}

void GeneralizedGaussianDistribution::print(){
    printf("%%GeneralizedGaussianDistribution:: mul = %5.5f mean = %5.5f stdval = %5.5f reg = %5.5f power = %5.5f\n",mul,mean,stdval,regularization,power);
}

double GeneralizedGaussianDistribution::getNoise(){return stdval+regularization;}

void GeneralizedGaussianDistribution::update_numcdf_vec(unsigned int bins, double prob){
	start = mean - pow(-2.0*log(prob),1.0/power)*stdval;
	stop  = mean + pow(-2.0*log(prob),1.0/power)*stdval;

    double step     = (stop-start)/(bins-1);
    double sum      = 0;
    numcdf_vec.resize(bins);
    for(unsigned int i = 0; i < bins; i++){
        double x = start+double(i)*step+0.5*step;
        numcdf_vec[i] = sum;
		if(i < bins-1){sum += getval(x);}
    }
    for(unsigned int i = 0; i < bins; i++){numcdf_vec[i] /= sum;}
}


}

