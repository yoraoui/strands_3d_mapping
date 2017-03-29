#include "weightfunctions/DistanceWeightFunction2.h"

namespace reglib
{

DistanceWeightFunction2::DistanceWeightFunction2(){
    name = "func";
	p = 0.5;
	f = PNORM;
	convergence_threshold = 0.0001;
	savePath = "";
	saveData.str("");
}
DistanceWeightFunction2::~DistanceWeightFunction2(){}

void DistanceWeightFunction2::setTune(){}

void DistanceWeightFunction2::setDebugg(bool debugg){
    debugg_print = debugg;
}

MatrixXd DistanceWeightFunction2::getMat(std::vector<double> & vec){
	unsigned long nr_data = vec.size();
	Eigen::VectorXd v (nr_data);
	for(unsigned long i = 0; i < nr_data; i++){v(i) = vec[i];}
	return v;
}

void DistanceWeightFunction2::computeModel(std::vector<double> & vec){
	computeModel(getMat(vec));
}

void DistanceWeightFunction2::computeModel(double * vec, unsigned int nr_data, unsigned int dim){
    //printf("Error: %s not implemented\n",__PRETTY_FUNCTION__);
}


VectorXd DistanceWeightFunction2::getProbs(std::vector<double> & vec){
	return getProbs(getMat(vec));
}

void tukey_weight(Eigen::VectorXd& r, double p) {
	for(int i=0; i<r.rows(); ++i) {
		if(r(i) > p) r(i) = 0.0;
		else r(i) = std::pow((1.0 - std::pow(r(i)/p,2.0)), 2.0);
    }
}
void threshold_weight(Eigen::VectorXd& r, double p) {			for(int i=0; i<r.rows(); ++i) {r(i) = fabs(r(i)) < p;}}
void uniform_weight(Eigen::VectorXd& r) {						r = Eigen::VectorXd::Ones(r.rows());}
void pnorm_weight(Eigen::VectorXd& r, double p, double reg) {	for(int i=0; i<r.rows(); ++i) {r(i) = p/(std::pow(r(i),2-p) + reg);}}
void fair_weight(Eigen::VectorXd& r, double p) {				for(int i=0; i<r.rows(); ++i) {r(i) = 1.0/(1.0 + r(i)/p);}}
void logistic_weight(Eigen::VectorXd& r, double p) {			for(int i=0; i<r.rows(); ++i) {r(i) = (p/r(i))*std::tanh(r(i)/p);}}
struct sort_pred { bool operator()(const std::pair<int,double> &left, const std::pair<int,double> &right) {return left.second < right.second;} };

void trimmed_weight(Eigen::VectorXd& r, double p) {
    std::vector<std::pair<int, double> > sortedDist(r.rows());
    for(int i=0; i<r.rows(); ++i) {sortedDist[i] = std::pair<int, double>(i,r(i));}
    std::sort(sortedDist.begin(), sortedDist.end(), sort_pred());
    r.setZero();
    int nbV = r.rows()*p;
    for(int i=0; i<nbV; ++i) {r(sortedDist[i].first) = 1.0;}
}

void robust_weight(Function f, Eigen::VectorXd& r, double p) {
    switch(f) {
		case THRESHOLD: threshold_weight(r,p); break;
        case PNORM: pnorm_weight(r,p); break;
        case TUKEY: tukey_weight(r,p); break;
        case FAIR: fair_weight(r,p); break;
        case LOGISTIC: logistic_weight(r,p); break;
        case TRIMMED: trimmed_weight(r,p); break;
        case NONE: uniform_weight(r); break;
        default: uniform_weight(r); break;
    }
}

void DistanceWeightFunction2::computeModel(MatrixXd mat){}
VectorXd DistanceWeightFunction2::getProbs(MatrixXd mat){
	VectorXd W = mat.colwise().norm();
	robust_weight(f, W , p);
	return W;//VectorXf(mat.rows());
}

double DistanceWeightFunction2::getProb(double d, bool debugg){
    switch(f) {
        case THRESHOLD:
            return fabs(d) < p;
        break;
        case PNORM:
            return p/(std::pow(fabs(d),2-p) + 0.00001);
        break;
        case TUKEY:
            if(fabs(d) > p){ return 0.0; }
            else { return std::pow((1.0 - std::pow(fabs(d)/p,2.0)), 2.0); }
        break;
        case FAIR:
            return 1.0/(1.0 + fabs(d)/p);
        break;
        case LOGISTIC:
            return (p/fabs(d))*std::tanh(fabs(d)/p);
        break;
        case TRIMMED: return 1; break;
        case NONE: return 1; break;
        default: return 1; break;
    }

	return 0;
}

double DistanceWeightFunction2::getProbInfront(double d, bool debugg){
	printf("double DistanceWeightFunction2::getProbInfront(double d){ not implemented\n");
	exit(0);
	return 0;
}


double DistanceWeightFunction2::getProbInfront(double start, double stop, bool debugg){
    printf("double DistanceWeightFunction2::getProbInfront(double start, double stop){ not implemented\n");
    exit(0);
    return 0;
}

double DistanceWeightFunction2::getNoise(){
    //if(f == THRESHOLD){return p / 4;}
    return 0.002;
}

bool DistanceWeightFunction2::update(){return true;}
void DistanceWeightFunction2::reset(){}

std::string DistanceWeightFunction2::getString(){
	std::string ty = "";
	switch(f) {
		case THRESHOLD: ty = "THRESHOLD"; break;
		case PNORM:		ty = "PNORM"; break;
		case TUKEY:		ty = "TUKEY"; break;
		case FAIR:		ty = "FAIR"; break;
		case LOGISTIC:	ty = "LOGISTIC"; break;
		case TRIMMED:	ty = "TRIMMED"; break;
		case NONE:		ty = "NONE"; break;
		default:		ty = "NONE"; break;
	}
	char buf [1024];
	sprintf(buf,"%s%4.4i",ty.c_str(),int(1000.0*p));
	return std::string(buf);
}

double DistanceWeightFunction2::getConvergenceThreshold(){
	return convergence_threshold;
}

DistanceWeightFunction2 * DistanceWeightFunction2::clone(){
    DistanceWeightFunction2 * func  = new DistanceWeightFunction2();
    func->name                      = name;
    func->f                         = f;
    func->p                         = p;
    func->regularization            = regularization;
    func->convergence_threshold     = convergence_threshold;
    func->debugg_print              = debugg_print;
    func->savePath                  = savePath;
    return func;
}

double DistanceWeightFunction2::getWeight(double invstd, double d, double & infoweight, double & prob, bool debugg){
	double invnoise = invstd/getNoise();

	infoweight = invnoise*invnoise;
	prob = getProb(d*invstd,debugg);
	return infoweight*prob;
}

VectorXd DistanceWeightFunction2::getWeights(std::vector<double > invstd, MatrixXd mat, bool debugg){
    VectorXd probs = getProbs(mat);
//    for(unsigned int i = 0; i < invstd.size(); i++){
//        double
//    }
    return probs;
}

void DistanceWeightFunction2::print(){

}

double DistanceWeightFunction2::getPower(){
    return 2;
}

}


