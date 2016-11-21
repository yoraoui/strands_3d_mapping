#include "weightfunctions/DistanceWeightFunction2PPR2.h"

namespace reglib{

DistanceWeightFunction2PPR2::DistanceWeightFunction2PPR2(	double maxd_, int histogram_size_){
	savePath = "";
	saveData.str("");


    fixed_histogram_size = false;

	regularization		= 0.1;
	maxd 				= maxd_;
	mind				= 0;
	histogram_size		= histogram_size_;
	starthistogram_size = histogram_size;
	blurval				= 5;
	stdval				= blurval;
	stdgrow				= 1.1;

    maxnoise = 99999999999999;
	noiseval = 100.0;

	prob.resize(histogram_size+1);
	prob[histogram_size] = 0;

	infront.resize(histogram_size+1);
	infront[histogram_size] = 1;
	
	histogram.resize(histogram_size+1);
	blur_histogram.resize(histogram_size+1);
	noise.resize(histogram_size+1);
	noisecdf.resize(histogram_size+1);

	refine_mean = false;
	refine_mul = false;
	refine_std = true;

	mulval = 1;
	meanval = 0;

	startmaxd = maxd_;
	startreg = 0.05;
	threshold = false;
	uniform_bias = true;
	debugg_print = false;
	scale_convergence = true;
	nr_inliers = 1;
	nr_refineiters = 1;

	max_under_mean = true;

	update_size = false;
	target_length = 10.0;
	data_per_bin = 30;
	blur = 0.05;

	bidir = false;
	iter = 0;

	maxp = 0.99;

	rescaling = true;

	interp = true;

	first = true;

    zeromean = false;

	costpen = 3.0;

	startreg			= 2;
	blurval				= 4;
	stdval				= 100;
	stdgrow				= 1.0;
	noiseval			= 100.0;
	debugg_print		= true;//false;//true;
	threshold			= false;
	scale_convergence	= true;
	blur				= 0.03;
	data_per_bin		= 30;
	costpen				= 2;
	uniform_bias		= false;//(setting&4 != 0);
	refine_mean			= false;//(setting&2 != 0);
	refine_mul			= false;//(setting&1 != 0);
	update_size			= true;//(setting&2 != 1);
	max_under_mean		= true;//(setting&1 != 0);
	rescaling			= false;//(setting&1 != 0);
	interp				= true;//(setting&1 != 0);
	refine_std			= true;
	nr_refineiters		= 1;
	convergence_threshold = 0.05;

	ggd = false;
	compute_infront = false;
}

DistanceWeightFunction2PPR2::~DistanceWeightFunction2PPR2(){}

class Gaussian3 {
	public:
	double mul;
	double mean;
	double stdval;
	double scaledinformation;
	void update(){scaledinformation = -0.5/(stdval*stdval);}
	
	Gaussian3(double mul_, double mean_,	double stdval_){
		mul = mul_;
		mean = mean_;
		stdval = stdval_;
		update();
	}
	
	double getval(double x){
		double dx = mean-x;
		return mul*exp(dx*dx*scaledinformation);
	}

	double getcdf(double x)
	{
	   return 0.5 * erfc(-((x-mean)/stdval) * M_SQRT1_2);
	}


    void print(){
        printf("Gaussian3:: mul = %5.5f mean = %5.5f stdval = %5.5f\n",mul,mean,stdval);
    }
};

class gaussian2 {
	public:
	gaussian2(double mean, double mul, double x, double y) : mean_(mean), mul_(mul), x_(x), y_(y) {}
	bool operator()(const double* const p,double* residuals) const {
		 double stddiv = p[0];
		if(mul_ > 0){
			double dx = x_-mean_;
			residuals[0]  = 0;
			residuals[0] += mul_*exp(-0.5*dx*dx/(stddiv*stddiv));
			residuals[0] -= y_;
            if(residuals[0] > 0){	residuals[0] =  sqrt( 5.0*residuals[0]);}
			else{					residuals[0] = -sqrt(-residuals[0]);}
		}else{residuals[0] = 99999999999999999999999.0;}
		return true;
	}
	private:
	const double mean_;
	const double mul_;
	const double x_;
	const double y_;
};

double getCurrentTime3(){
	struct timeval start;
	gettimeofday(&start, NULL);
	double sec = start.tv_sec;
	double usec = start.tv_usec;
	double t = sec+usec*0.000001;
	return t;
}


double scoreCurrentHist(double power, std::vector<float> & A, std::vector<float> & B){
	unsigned int nr_data = A.size();
	double sumA = 0;
	double sumB = 0;

	for(unsigned int i = 0; i < nr_data; i++){
		sumA += A[i];
		sumB += pow(B[i],power);
	}

	double sumdiff = 0;
	for(unsigned int i = 0; i < nr_data; i++){
		double diff = fabs(A[i]/sumA - pow(B[i],power)/sumB);
		sumdiff += diff*diff;
	}
	return sumdiff;
}

double fitCurrentHist2(std::vector<float> & A, std::vector<float> & B){
	int iter = 25;
	double h = 0.000001;

	double p_max = 10;
	double p_min = 0;
	double p_mid = (p_max+p_min)/2;
	for(int i = 0; i < iter; i++){
		p_mid = (p_max+p_min)/2;
		double p_neg = scoreCurrentHist(p_mid-h, A, B);
		double p_pos = scoreCurrentHist(p_mid+h, A, B);
		if(p_neg < p_pos){	p_max = p_mid;}
		else{				p_min = p_mid;}
	}
	return p_mid;
}

double fitCurrentHist(std::vector<float> & A, std::vector<float> & B){
	double pbest = 0;
	double pmin = 99999999;
	for(double p = 1; p <= 3; p+=0.01){

		double pscore = scoreCurrentHist(p, A, B);
		if(pscore < pmin){
			pbest = p;
			pmin = pscore;
		}
	}
	return pbest;
}

void blurHistogram3(std::vector<float> & blur_hist,  std::vector<float> & hist, float stdval, int histogramsize, bool debugg_print = false){
	int nr_data = histogramsize;//blur_hist.size();

//if(debugg_print){
//	for(int i = 0; i < nr_data; i++){
//		printf("%i -> %f\n",i,hist[i]);
//	}
//}

	int nr_data2 = 2*nr_data;

	std::vector<float> tmphist;
	tmphist.resize(nr_data2);

	std::vector<float> tmphist_blur;
	tmphist_blur.resize(nr_data2);

	for(int i = 0; i < nr_data; i++){
		tmphist[i]			= hist[nr_data-1-i];
		tmphist[nr_data+i]	= hist[i];
		tmphist_blur[i]			= 0;
		tmphist_blur[nr_data+i]	= 0;
	}

	double info = -0.5/(stdval*stdval);
	double weights[nr_data2];
	for(int i = 0; i < nr_data2; i++){weights[i] = 0;}
	for(int i = 0; i < nr_data2; i++){
		double current = exp(i*i*info);
		weights[i] = current;
		if(current < 0.001){break;}
	}

	int offset = 4.0*stdval;
	offset = std::max(4,offset);
	for(int i = 0; i < nr_data2; i++){
		double v = tmphist[i];
		if(v == 0){continue;}

		//if(debugg_print){printf("v: %5.5f\n",v);}

		int start	= std::max(0,i-offset);
		int stop	= std::min(nr_data2,i+offset+1);

		for(int j = start; j < stop; j++){tmphist_blur[j] += v*weights[abs(i-j)];}
		//if(debugg_print){exit(0);}
	}
	for(int i = 0; i < nr_data; i++){blur_hist[i] = tmphist_blur[i+nr_data];}

	float bef = 0;
	float aft = 0;
	for(int i = 0; i < nr_data; i++){
		bef += hist[i];
		aft += blur_hist[i];
	}

	for(int i = 0; i < nr_data; i++){blur_hist[i] *= bef/aft;}

//	printf("hist = [");				for(int k = 0; k < 300 && k < hist.size();		k++){printf("%i ",int(hist[k]));}		printf("];\n");
//	printf("hist_smooth = [");		for(int k = 0; k < 300 && k < blur_hist.size();	k++){printf("%i ",int(blur_hist[k]));}	printf("];\n");
//	double p = fitCurrentHist(hist, blur_hist);
//	printf("p = %f;\n",p);
//	double p2 = fitCurrentHist2(hist, blur_hist);
//	printf("p2 = %f;\n",p2);
//	for(double p = 1.0; p < 3; p+=0.001){
//		printf("%3.3f -> %8.8f\n",p,scoreCurrentHist(p, hist, blur_hist));
//	}
//	exit(0);
}


void blurHistogram2(std::vector<float> & blur_hist,  std::vector<float> & hist, float stdval,bool debugg_print = false){
	int nr_data = blur_hist.size();
	int nr_data2 = 2*nr_data;

	std::vector<float> tmphist;
	tmphist.resize(nr_data2);

	std::vector<float> tmphist_blur;
	tmphist_blur.resize(nr_data2);

	for(int i = 0; i < nr_data; i++){
		tmphist[i]			= hist[nr_data-1-i];
		tmphist[nr_data+i]	= hist[i];
		tmphist_blur[i]			= 0;
		tmphist_blur[nr_data+i]	= 0;
	}

	double info = -0.5/(stdval*stdval);
	double weights[nr_data2];
	for(int i = 0; i < nr_data2; i++){weights[i] = 0;}
	for(int i = 0; i < nr_data2; i++){
		double current = exp(i*i*info);
		weights[i] = current;
		if(current < 0.001){break;}
	}

	int offset = 4.0*stdval;
	offset = std::max(4,offset);
	for(int i = 0; i < nr_data2; i++){
		double v = tmphist[i];
		if(v == 0){continue;}

        int start	= std::max(0,i-offset);
        int stop	= std::min(nr_data2,i+offset+1);

		for(int j = start; j < stop; j++){tmphist_blur[j] += v*weights[abs(i-j)];}
	}
	for(int i = 0; i < nr_data; i++){blur_hist[i] = tmphist_blur[i+nr_data];}

	float bef = 0;
	float aft = 0;
	for(int i = 0; i < nr_data; i++){
		bef += hist[i];
		aft += blur_hist[i];
	}

	for(int i = 0; i < nr_data; i++){blur_hist[i] *= bef/aft;}

//	printf("hist = [");				for(int k = 0; k < 300 && k < hist.size();		k++){printf("%i ",int(hist[k]));}		printf("];\n");
//	printf("hist_smooth = [");		for(int k = 0; k < 300 && k < blur_hist.size();	k++){printf("%i ",int(blur_hist[k]));}	printf("];\n");
//	double p = fitCurrentHist(hist, blur_hist);
//	printf("p = %f;\n",p);
//	double p2 = fitCurrentHist2(hist, blur_hist);
//	printf("p2 = %f;\n",p2);
//	for(double p = 1.0; p < 3; p+=0.001){
//		printf("%3.3f -> %8.8f\n",p,scoreCurrentHist(p, hist, blur_hist));
//	}
//	exit(0);
}

void blurHistogramBidir2(std::vector<float> & blur_hist,  std::vector<float> & hist, float stdval,bool debugg_print = false){
	int nr_data = blur_hist.size();
	double info = -0.5/(stdval*stdval);
	double weights[nr_data];
	for(int i = 0; i < nr_data; i++){
		weights[i] = 0;
	}

	for(int i = 0; i < nr_data; i++){
		double current = exp(i*i*info);
		weights[i] = current;

	}

	int offset = 4.0*stdval;
	offset = std::max(4,offset);
	for(int i = 0; i < nr_data; i++){
		double v = hist[i];
		if(v == 0){continue;}

		int start	= std::max(0,i-offset);
		int stop	= std::min(nr_data,i+offset+1);

		for(int j = start; j < stop; j++){
			double w = weights[abs(i-j)];
			blur_hist[j] += v*w;///sum;
		}
	}

	float bef = 0;
	float aft = 0;
	for(int i = 0; i < nr_data; i++){
		bef += hist[i];
		aft += blur_hist[i];
	}

	for(int i = 0; i < nr_data; i++){blur_hist[i] *= bef/aft;}
}

const double step_h = 0.00001;
const unsigned int step_iter = 25;
const double cutoff_exp = -14;

double scoreCurrent3(double mul, double mean, double stddiv, double power, std::vector<float> & X, std::vector<float> & Y, unsigned int nr_data, double costpen){
	double sum = 0;
	double invstd = 1.0/stddiv;
	for(unsigned int i = 0; i < nr_data; i++){
		double dx = fabs(X[i] - mean)*invstd;
		double inp = -0.5*pow(dx,power);
		if(inp < cutoff_exp){sum += Y[i];}
		else{
			//double diff = Y[i]*Y[i]*(mul*exp(inp) - Y[i]);
			double diff = (mul*exp(inp) - Y[i]);
			//			if(diff > 0){	sum += costpen*diff;}
			//			else{			sum -= diff;}
			if(diff > 0){	sum += diff*diff;}
			else{			sum -= diff;}
		}
	}
	return sum;
}

double fitStdval3(double mul, double mean, double std_mid, double power, std::vector<float> & X, std::vector<float> & Y, unsigned int nr_data, double costpen){
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

double fitPower3(double mul, double mean, double std_mid, double power, std::vector<float> & X, std::vector<float> & Y, unsigned int nr_data, double costpen){
	int iter = 25;
	double h = 0.000000001;

	double power_max = power*2;
	double power_min = 0.001;
	for(int i = 0; i < iter; i++){
		power = (power_max+power_min)/2;
		double std_neg = scoreCurrent3(mul,mean,std_mid,power-h,X,Y,nr_data,costpen);
		double std_pos = scoreCurrent3(mul,mean,std_mid,power+h,X,Y,nr_data,costpen);
		if(std_neg < std_pos){	power_max = power;}
		else{					power_min = power;}
	}
	power = (power_max+power_min)/2;
	return power;
}

double fitMean3(double mul, double mean, double std_mid, double power, std::vector<float> & X, std::vector<float> & Y, unsigned int nr_data, double costpen){
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


double scoreCurrent2(double bias, double mul, double mean, double stddiv, std::vector<float> & X, std::vector<float> & Y, unsigned int nr_data, double costpen){
	double info = -0.5/(stddiv*stddiv);
	double sum = 0;
	for(unsigned int i = 0; i < nr_data; i++){
		double dx = X[i] - mean;
		double inp = info*dx*dx;
		if(inp < cutoff_exp){sum += Y[i];}
		else{
			double diff = mul*exp(info*dx*dx) - Y[i];
			//if(diff > 0){	sum += costpen*diff;}
			if(diff > 0){	sum += diff*diff;}
			else{			sum -= diff;}
		}
	}
	return sum;
}

double fitStdval2(double bias, double mul, double mean, double std_mid, std::vector<float> & X, std::vector<float> & Y, unsigned int nr_data, double costpen){
	int iter = 25;
	double h = 0.000000001;

	double std_max = std_mid*2;
	double std_min = 0;
	for(int i = 0; i < iter; i++){
		std_mid = (std_max+std_min)/2;
		double std_neg = scoreCurrent2(bias,mul,mean,std_mid-h,X,Y,nr_data,costpen);
		double std_pos = scoreCurrent2(bias,mul,mean,std_mid+h,X,Y,nr_data,costpen);
		if(std_neg < std_pos){	std_max = std_mid;}
		else{					std_min = std_mid;}
	}
	return std_mid;
}

double fitBias2(double bias, double mul, double mean, double std_mid, std::vector<float> & X, std::vector<float> & Y, unsigned int nr_data, double costpen){
	int iter = 25;
	double h = 0.000000001;
	double bias_max = bias*2;
	double bias_min = 0;

	for(int i = 0; i < iter; i++){
		bias = (bias_max+bias_min)/2;
		double std_neg = scoreCurrent2(bias-h,mul,mean,std_mid,X,Y,nr_data,costpen);
		double std_pos = scoreCurrent2(bias+h,mul,mean,std_mid,X,Y,nr_data,costpen);

		if(std_neg < std_pos){	bias_max = bias;}
		else{					bias_min = bias;}
	}
	return bias;
}

double fitMean2(double bias,double mul, double mean, double std_mid, std::vector<float> & X, std::vector<float> & Y, unsigned int nr_data, double costpen){
	int iter = 10;
	double h = 0.000000001;
	double mean_max = mean+1;
	double mean_min = mean-1;

	//	double mean_max = mean*2;
	//	double mean_min = 0;

	for(int i = 0; i < iter; i++){
		mean = (mean_max+mean_min)/2;
		double std_neg = scoreCurrent2(bias,mul,mean-h,std_mid,X,Y,nr_data,costpen);
		double std_pos = scoreCurrent2(bias,mul,mean+h,std_mid,X,Y,nr_data,costpen);
		if(std_neg < std_pos){	mean_max = mean;}
		else{					mean_min = mean;}
	}
	return mean;
}

double fitMul2(double bias, double mul, double mean, double std_mid, std::vector<float> & X, std::vector<float> & Y, unsigned int nr_data, double costpen){
	int iter = 25;
	double h = 0.000000001;
	double mul_max = mul*2;
	double mul_min = 0;

	for(int i = 0; i < iter; i++){
		mul = (mul_max+mul_min)/2;
		double std_neg = scoreCurrent2(bias,mul-h,mean,std_mid,X,Y,nr_data,costpen);
		double std_pos = scoreCurrent2(bias,mul+h,mean,std_mid,X,Y,nr_data,costpen);
		if(std_neg < std_pos){	mul_max = mul;}
		else{					mul_min = mul;}
	}
	return mul;
}

Gaussian3 getModel(double & stdval,std::vector<float> & hist, bool uniform_bias, bool refine_mean, bool refine_mul, bool refine_std, int nr_refineiters, double costpen, bool zeromean){
	double mul = hist[0];
	double mean = 0;
	unsigned int nr_bins = hist.size();
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
	double bias = 0;
	if(uniform_bias){
		stdval = 0.01;
		double ysum = 0;
		for(unsigned int i = 0; i < nr_data_opt; i++){bias += fabs(Y[i]);}
		bias /= nr_data_opt;
	}

	double ysum = 0;
	for(unsigned int i = 0; i < nr_data_opt; i++){ysum += fabs(Y[i]);}

	double std_mid = 0;
	for(unsigned int i = 0; i < nr_data_opt; i++){std_mid += (X[i]-mean)*(X[i]-mean)*fabs(Y[i]-bias)/ysum;}
	stdval = sqrt(std_mid);

	for(int i = 0; i < nr_refineiters; i++){
		if(uniform_bias){	bias	= fitBias2(		bias, mul,mean,stdval,X,Y,nr_data_opt,costpen);}
		if(refine_std){		stdval	= fitStdval2(	bias, mul,mean,stdval,X,Y,nr_data_opt,costpen);}
		if(refine_mean){	mean	= fitMean2(		bias, mul,mean,stdval,X,Y,nr_data_opt,costpen);}
		if(refine_mul){		mul		= fitMul2(		bias, mul,mean,stdval,X,Y,nr_data_opt,costpen);}
	}
	return Gaussian3(mul,mean,stdval);
}
/*
void getGGModel(std::vector<float> & hist, unsigned int nr_bins){


	double mul = hist[0];
	double mean = 0;
	//unsigned int nr_bins = hist.size();

	for(unsigned int k = 1; k < nr_bins; k++){
		if(hist[k] > mul){
			mul = hist[k];
			mean = k;
		}
	}

	double maxhist = mul;


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
	double stdval = sqrt(std_mid);

	double power = 2;
	//printf("mean: %10.10f mul: %10.10f\n",mean,mul,std_mid,power);

	printf("hist = [");
	for(unsigned int k = 0; k < nr_bins; k++){printf("%5.5f ",float(hist[k])/maxhist);}
	printf("];\n");

	char buf [1024*1024];
	sprintf(buf,"ggd = [");
	int nr_refineiters = 10;

	double pre_mean = mean;
	double pre_std = stdval;
	double pre_power = power;

	for(int i = 0; i < nr_refineiters; i++){
		printf("ggd%i = [",i);
		for(unsigned int k = 0; k < nr_bins; k++){
			printf("%5.5f ",mul/maxhist * exp(-0.5*pow(fabs(double(k)-mean)/stdval,power)));
		}
		printf("];\n");

		if(i != 0){sprintf(buf,";");}
		sprintf(buf,"%10.10f, %10.10f, %10.10f, %10.10f",mul,mean,stdval,power);

		stdval	= fitStdval3(	mul,mean,stdval,power,X,Y,nr_data_opt,3);
		power	= fitPower3(	mul,mean,stdval,power,X,Y,nr_data_opt,3);
		mean	= fitMean3(		mul,mean,stdval,power,X,Y,nr_data_opt,3);
		if(fabs(mean - pre_mean) < 0.1 && fabs(stdval - pre_std) < 0.1 && fabs(power - pre_power) < 0.01){
			break;
		}
		pre_mean = mean;
		pre_std = stdval;
		pre_power = power;
	}
	sprintf(buf,"];\n");
	printf("%s\n",buf);

	printf("mean: %10.10f mul: %10.10f std: %10.10f power: %10.10f\n",mean,mul,std_mid,power);
	//return Gaussian3(mul,mean,stdval);
}
*/

double DistanceWeightFunction2PPR2::getNoise(){return regularization+noiseval;}// + stdval*double(histogram_size)/maxd;}

#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

void DistanceWeightFunction2PPR2::computeModel(MatrixXd mat){
	const unsigned int nr_data = mat.cols();
	const int nr_dim = mat.rows();

double startTime = getCurrentTime3();
//printf("\n################################################### ITER:%i ############################################################\n",iter);
	if(debugg_print){printf("\n################################################### ITER:%i ############################################################\n",iter);}
	if(!fixed_histogram_size){
		if(first){
			first = false;
			double sum = 0;
			for(unsigned int j = 0; j < nr_data; j++){
				for(int k = 0; k < nr_dim; k++){sum += mat(k,j)*mat(k,j);}
			}
			stdval2 = sqrt(sum / double(nr_data*nr_dim));
			if(debugg_print){printf("stdval2: %f\n",stdval2);}
		}

		if(update_size){
			//maxd = fabs(meanval2) + (stdval2 + regularization)*target_length;
			if(bidir){
				maxd = meanval2 + 2.0*(stdval2 + regularization)*target_length;
				mind = meanval2 - 2.0*(stdval2 + regularization)*target_length;
			}else{
				maxd = fabs(meanval2) + (stdval2 + regularization)*target_length;
				mind = 0;
			}
			if(debugg_print){
				printf("maxd: %f\n",maxd);
				printf("mind: %f\n",mind);
			}
		}

		double nr_inside = 0;
		for(unsigned int j = 0; j < nr_data; j++){
			for(int k = 0; k < nr_dim; k++){
				//if(fabs(mat(k,j)) < maxd){nr_inside++;}
				double d = mat(k,j);
				if(mind < d && d < maxd){nr_inside++;}
			}
		}

		if(update_size){
			nr_inside = std::min(nr_inside,double(nr_dim)*nr_inliers);
			int new_histogram_size = std::min(int(prob.size()),std::max(50,int(0.5 + nr_inside/data_per_bin)));
			histogram_size = std::min(1000,new_histogram_size);//std::min(int(prob.size()),std::max(50,std::min(1000,int(0.5 + nr_inside/data_per_bin))));
			blurval = blur*double(histogram_size)*float(histogram_size)/float(new_histogram_size);

			if(debugg_print){printf("nr_inside: %f histogram_size: %i blurval: %f\n",nr_inside,histogram_size,blurval);}
		}
	}
//printf("time taken to %5.5i: %10.10f\n",__LINE__,getCurrentTime3()-startTime); startTime = getCurrentTime3();
	histogram_mul	= double(histogram_size)/maxd;
	histogram_mul2	= double(histogram_size)/(maxd-mind);
//	if(bidir){	histogram_mul2 = double(histogram_size)/(2.0*maxd);}
//	else{		histogram_mul2 = double(histogram_size)/maxd;}

	if(debugg_print){printf("histogram_mul: %5.5f histogram_size: %5.5f maxd: %5.5f mind: %5.5f\n",histogram_mul,double(histogram_size),maxd,mind);}

	double start_time = getCurrentTime3();

	for(int j = 0; j < histogram_size; j++){histogram[j] = 0;}
	for(unsigned int j = 0; j < nr_data; j++){
		for(int k = 0; k < nr_dim; k++){
			double ind = getInd(mat(k,j));
			if(ind >= 0 && (ind+0.00001) < histogram_size){
				histogram[int(ind+0.00001)]++;
			}
		}
	}
//printf("time taken to %5.5i: %10.10f\n",__LINE__,getCurrentTime3()-startTime); startTime = getCurrentTime3();
	//double ind = getInd(d,debugg);

	start_time = getCurrentTime3();
	blurHistogram3(blur_histogram,histogram,blurval,histogram_size,debugg_print);
//printf("time taken to %5.5i: %10.10f\n",__LINE__,getCurrentTime3()-startTime); startTime = getCurrentTime3();
//	stdval2 = 0;
//	Gaussian3 g  = getModel(stdval, blur_histogram,	uniform_bias,refine_mean,refine_mul,refine_std,nr_refineiters,costpen,zeromean);
//	//Gaussian3 g2 = getModel(stdval2,histogram,		uniform_bias,refine_mean,refine_mul,refine_std,nr_refineiters,costpen,zeromean);
//	stdval2 = std::max(1.0,stdval2);
//	stdval2 = maxd*stdval2/float(histogram_size);

//	g.stdval = std::max(0.0001*double(histogram_size)/(maxd),std::min(g.stdval,maxnoise*double(histogram_size)/maxd));

//	noiseval = maxd*g.stdval/float(histogram_size);

//	stdval	= g.stdval;

//	double Ib = 1/(blurval*blurval);
//	double Ic = 1/(stdval*stdval);
//	double Ia = Ib*Ic/(Ib-Ic);
//	double corrected = sqrt(1.0/Ia);
//	double rescaleval = 1;

//	if(rescaling){
//		if(stdval > blurval){
//			g.stdval = corrected;
//			rescaleval = stdval/corrected;
//		}else{g.stdval = 1;}
//	}

//	stdval = g.stdval;
//	mulval	= g.mul;
//	meanval	= g.mean;

//	meanval2 = maxd*meanval/float(histogram_size);

//	g.stdval += histogram_size*regularization/maxd;
//	g.update();


	Gaussian3 g  = getModel(stdval, blur_histogram,	uniform_bias,refine_mean,refine_mul,refine_std,nr_refineiters,costpen,zeromean);
//    g.print();

//    GaussianDistribution * gd = new GaussianDistribution(refine_std,zeromean,refine_mean,refine_mul,costpen,nr_refineiters);
//    gd->train(blur_histogram,histogram_size);
//    gd->print();
//    delete gd;
//printf("time taken to %5.5i: %10.10f\n",__LINE__,getCurrentTime3()-startTime); startTime = getCurrentTime3();
	g.stdval = std::max(0.0001*double(histogram_size)/(maxd),std::min(g.stdval,maxnoise*double(histogram_size)/maxd));

	//noiseval = maxd*g.stdval/float(histogram_size);
	noiseval = (maxd-mind)*g.stdval/float(histogram_size);
	if(bidir){noiseval *= 0.5;}

	stdval = g.stdval;
	mulval	= g.mul;
	meanval	= g.mean;

	meanval2 = (maxd-mind)*(meanval/float(histogram_size)) + mind;
	stdval2 = noiseval;//getNoise();
	//meanval2 = maxd*meanval/float(histogram_size);

	g.stdval += histogram_size*regularization/maxd;
	g.update();

	for(int k = 0; k < histogram_size; k++){noise[k] = g.getval(k);}
	if(compute_infront){
		for(int k = 0; k < histogram_size; k++){noisecdf[k] = g.getcdf(k);}
	}
//printf("time taken to %5.5i: %10.10f\n",__LINE__,getCurrentTime3()-startTime); startTime = getCurrentTime3();
//	std::vector<float> new_histogram;
//	if(rescaling){
//		new_histogram.resize(blur_histogram.size());
//		double sum = 0;
//		for(int k = 0; k < histogram_size; k++){
//			double ks = double(k)*rescaleval;
//			double w2 = ks-int(ks);
//			double w1 = 1-w2;
//			double newh = blur_histogram[histogram_size-1];
//			if(ks < histogram_size){
//				newh = blur_histogram[int(ks)]*w1 + blur_histogram[int(ks+1)]*w2;
//				sum +=  newh;
//			}
//			new_histogram[k] = newh;
//		}
//	}
	
	for(int k = 0; k < histogram_size; k++){
		if(max_under_mean && k < g.mean){	prob[k] = maxp;	}
		else{
			double hs = blur_histogram[k]+1;
			prob[k] = std::min(maxp , noise[k]/hs);//never fully trust any data
			infront[k] = (1-prob[k])*noisecdf[k];
		}
	}
//printf("time taken to %5.5i: %10.10f\n",__LINE__,getCurrentTime3()-startTime); startTime = getCurrentTime3();
	nr_inliers = 0;
	for(unsigned int j = 0; j < nr_data; j++){
		float inl  = 1;
		float ninl = 1;
		for(int k = 0; k < nr_dim; k++){
			float p = getProb(mat(k,j));
			inl *= p;
			ninl *= 1.0-p;
		}
		double d = inl / (inl+ninl);
		nr_inliers += d;
	}
//printf("time taken to %5.5i: %10.10f\n",__LINE__,getCurrentTime3()-startTime); startTime = getCurrentTime3();//SLOW
	if(debugg_print){printf("hist = [");			for(int k = 0; k < 3000 && k < histogram_size; k++){printf("%i ",	int(histogram[k]));}		printf("];\n");}
	if(debugg_print){printf("noise = [");			for(int k = 0; k < 3000 && k < histogram_size; k++){printf("%i ",	int(noise[k]));}			printf("];\n");}
	if(debugg_print){printf("noisecdf = [");		for(int k = 0; k < 3000 && k < histogram_size; k++){printf("%2.2f ",int(noisecdf[k]));}			printf("];\n");}
	if(debugg_print){printf("hist_smooth = [");		for(int k = 0; k < 3000 && k < histogram_size; k++){printf("%i ",	int(blur_histogram[k]));}	printf("];\n");}
	if(debugg_print){printf("prob = [");			for(int k = 0; k < 3000 && k < histogram_size; k++){printf("%2.2f ",prob[k]);}					printf("];\n");}
	if(debugg_print){printf("infront = [");			for(int k = 0; k < 3000 && k < histogram_size; k++){printf("%2.2f ",infront[k]);}				printf("];\n");}
	if(debugg_print){printf("clf; hold on; plot(hist_smooth,'r','LineWidth',2); plot(hist,'m','LineWidth',2); plot(noise,'b','LineWidth',2); plot(prob*max(noise),'g','LineWidth',2); plot(infront*max(noise),'g','LineWidth',2);\n");}

	if(false){printf("meanoffset: %f stdval2: %f stdval: %f regularization: %f\n",meanval2,stdval2,noiseval,regularization);}
	if(!fixed_histogram_size && update_size ){

		double next_maxd;
		double next_mind;
		if(bidir){
			next_maxd = meanval2 + 2.0*(stdval2 + regularization)*target_length;
			next_mind = meanval2 - 2.0*(stdval2 + regularization)*target_length;
		}else{
			next_maxd = fabs(meanval2) + (stdval2 + regularization)*target_length;
			next_mind = 0;
		}

		double logdiff = log(next_maxd/maxd);

		//Ensure overlap
		if(next_maxd < mind){		return;}
		if(next_mind > maxd){		return;}
		if(next_maxd == next_mind){	return;}

		double overlap = 0;
		if(next_maxd <= maxd && next_mind >= mind){overlap = next_maxd	-next_mind;}
		if(next_maxd >= maxd && next_mind <= mind){overlap = maxd		-	  mind;}
		if(next_maxd <= maxd && next_mind <= mind){overlap = next_maxd	-	  mind;}
		if(next_maxd >= maxd && next_mind >= mind){overlap =      maxd	-next_mind;}

		double ratio = 2*overlap/(maxd+next_maxd-mind-next_mind);
		double newlogdiff = log(ratio);

		//if(bidir){
//			printf("meanoffset: %f stdval2: %f stdval: %f regularization: %f\n",meanval2,stdval2,noiseval,regularization);
//			printf("mind: %5.5f maxd: %5.5f noise: %5.5f noise+reg: %5.5f\n",mind,maxd,noiseval,getNoise());
//			printf("next_maxd: %5.5f maxd: %5.5f\n",next_maxd,maxd);
//			printf("next_mind: %5.5f mind: %5.5f\n",next_mind,mind);
//			printf("logdiff: %5.5f\n",logdiff);
//			printf("newlogdiff: %5.5f\n",newlogdiff);
//			printf("ratio: %5.5f = %5.5f / %5.5f\n",ratio,2*overlap,(maxd+next_maxd-mind-next_mind) );
		//}


		//double next_maxd  = meanval2 + (stdval2 + regularization)*target_length;


		//if(fabs(logdiff) > 0.2 && iter < 30){
		if(fabs(newlogdiff) > 0.2 && iter < 30){
			iter++;
			computeModel(mat);
			return;
		}else{iter = 0;}
	}
//printf("time taken to %5.5i: %10.10f\n",__LINE__,getCurrentTime3()-startTime); startTime = getCurrentTime3();
	//if(bidir){exit(0);}
}

VectorXd DistanceWeightFunction2PPR2::getProbs(MatrixXd mat){
	const unsigned int nr_data = mat.cols();
	const int nr_dim = mat.rows();

	nr_inliers = 0;
	VectorXd weights = VectorXd(nr_data);
	for(unsigned int j = 0; j < nr_data; j++){
		float inl  = 1;
		float ninl = 1;
		for(int k = 0; k < nr_dim; k++){
			float p = getProb(mat(k,j));
			inl *= p;
			ninl *= 1.0-p;
		}
		double d = inl / (inl+ninl);
		nr_inliers += d;
		weights(j) = d;
	}

	if(threshold){
		for(unsigned int j = 0; j < nr_data; j++){weights(j) = weights(j) > 0.5;}
	}
	return weights;
}

double DistanceWeightFunction2PPR2::getProb(double d, bool debugg){
	double ind = getInd(d,debugg);
	float p = 0;
	if(interp){
		double w2 = ind-int(ind);
		double w1 = 1-w2;
		if(ind >= 0 && (ind+1) < histogram_size){
			p = prob[int(ind)]*w1 + prob[int(ind+1)]*w2;
		}
	}else{
		if(ind >= 0 && ind < histogram_size){
			p = prob[int(ind)];
		}
	}


	if(debugg){printf("d: %5.5f -> ind: %5.5f -> p: %5.5f histogramsize: %5.5i maxd: %5.5f\n",d,ind,p,histogram_size,maxd);}
	return p;

/*
 * 	//const float histogram_mul = float(histogram_size)/maxd;
	double ind = fabs(d)*histogram_mul;
	double w2 = ind-int(ind);
	double w1 = 1-w2;
	float p = 0;
	if(ind >= 0 && (ind+0.5) < histogram_size){
		if(interp){	p = prob[int(ind)]*w1 + prob[int(ind+1)]*w2;
		}else{		p = prob[int(ind+0.5)];}
	}
	return p;
	*/
}


double DistanceWeightFunction2PPR2::getProbInfront(double d, bool debugg){
	double ind = getInd(d,debugg);
	float p;
	if(ind > meanval){	p = 1;}
	else{				p = 0;}

	if(interp){
		double w2 = ind-int(ind);
		double w1 = 1-w2;
		if(ind >= 0 && (ind+1) < histogram_size){
			p = infront[int(ind)]*w1 + infront[int(ind+1)]*w2;
		}
	}else{
		if(ind >= 0 && ind < histogram_size){
			p = infront[int(ind)];
		}
	}


	if(debugg){printf("d: %5.5f -> ind: %5.5f -> infront: %5.5f histogramsize: %5.5i maxd: %5.5f\n",d,ind,p,histogram_size,maxd);}
	return p;

/*
 * 	//const float histogram_mul = float(histogram_size)/maxd;
	double ind = fabs(d)*histogram_mul;
	double w2 = ind-int(ind);
	double w1 = 1-w2;
	float p = 0;
	if(ind >= 0 && (ind+0.5) < histogram_size){
		if(interp){	p = prob[int(ind)]*w1 + prob[int(ind+1)]*w2;
		}else{		p = prob[int(ind+0.5)];}
	}
	return p;
	*/
}

bool DistanceWeightFunction2PPR2::update(){
	if(true){
		std::vector<float> new_histogram = histogram;
		std::vector<float> new_blur_histogram = blur_histogram;

		float old_sum_prob = 0;
        for(int k = 0; k < histogram_size; k++){old_sum_prob += prob[k] * histogram[k];}

		Gaussian3 g = Gaussian3(mulval,meanval,stdval);//getModel(stdval,blur_histogram,uniform_bias);

		int iteration = 0;
        double regularization_before = regularization;
        for(int i = 0; i < 500; i++){
			iteration++;
            regularization *= 0.5;
			double change = histogram_size*regularization/maxd;
            //printf("DistanceWeightFunction2PPR2::update() %i -> reg %f change %f\n",i,regularization,change);

			if(change < 0.01*stdval){return true;}

			g.stdval += change;
			g.update();

			float new_sum_prob = 0;
			for(int k = 0; k < histogram_size; k++){
				double hs = new_blur_histogram[k] +0.0000001;
				new_sum_prob += std::min(maxp , g.getval(k)/hs) * new_histogram[k];
			}

            //printf("new_sum_prob %f old_sum_prob %f ratio: %f\n",new_sum_prob,old_sum_prob,new_sum_prob/old_sum_prob);
			if(new_sum_prob < 0.99*old_sum_prob || regularization < regularization_before/double(target_length*5)){return false;}
			g.stdval -= change;
			g.update();
		}
        return true;
	}else{
		regularization *= 0.5;
	}
}

void DistanceWeightFunction2PPR2::reset(){
	nr_inliers = 9999999;
	regularization	= startreg;
	maxd			= startmaxd;
	if(bidir){mind = -maxd;}
	else{mind = 0;}
	histogram_size	= starthistogram_size;

	stdval		= maxd/target_length;
	stdval2		= maxd/target_length;
	meanval		= 0;
	meanval2	= 0;
	iter = 0;
	first = true;

}

std::string DistanceWeightFunction2PPR2::getString(){
	char buf [1024];
	sprintf(buf,"PPR_%i_%i_%i_%i",int(1000.0*startreg),int(1000.0*blur),interp,int(data_per_bin));
	return std::string(buf);
}

double DistanceWeightFunction2PPR2::getConvergenceThreshold(){
	if(scale_convergence){
		//return convergence_threshold*(regularization + maxd*stdval/double(histogram_size))/sqrt(nr_inliers);
		//printf("%5.5f * %5.5f * %5.5f = %15.15f\n",convergence_threshold,getNoise(),1.0/sqrt(nr_inliers),convergence_threshold*getNoise()/sqrt(nr_inliers));
		return convergence_threshold*getNoise()/sqrt(nr_inliers);
	}else{
		return convergence_threshold;
	}
}

inline double DistanceWeightFunction2PPR2::getInd(double d, bool debugg){
//	if(bidir){
//		return 0.00001+(d+maxd)*histogram_mul2;
//	}else{
//		return 0.00001+fabs(d)*histogram_mul2;
//	}
	if(bidir){
		return 0.00001+(d-mind)*histogram_mul2;
	}else{
		return 0.00001+fabs(d)*histogram_mul2;
	}
}
}


