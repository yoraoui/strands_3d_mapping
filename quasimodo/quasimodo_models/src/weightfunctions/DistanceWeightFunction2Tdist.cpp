#include "weightfunctions/DistanceWeightFunction2Tdist.h"

namespace reglib{

DistanceWeightFunction2Tdist::DistanceWeightFunction2Tdist(	double v_){
	savePath = "";
	saveData.str("");
    v = v_;
}
DistanceWeightFunction2Tdist::~DistanceWeightFunction2Tdist(){}

double DistanceWeightFunction2Tdist::getNoise(){return 0.001;}

void DistanceWeightFunction2Tdist::computeModel(MatrixXd mat){
	const unsigned int nr_data = mat.cols();
	const int nr_dim = mat.rows();
    meanval = 0;
//    for(unsigned int i = 0; i < nr_dim; i++){
//        for(unsigned int j = 0; j < nr_data; j++){
//            meanval += mat(i,j);
//        }
//    }
//    meanval /= double(nr_dim*nr_data);

    double var = 0;
    for(unsigned int i = 0; i < nr_dim; i++){
        for(unsigned int j = 0; j < nr_data; j++){
            var += (mat(i,j)-meanval)*(mat(i,j)-meanval);
        }
    }

    var /= double(nr_dim*nr_data);
    info = 1.0/var;

    for(int i = 0; i < 100; i++){
        double var2 = 0;
        for(unsigned int i = 0; i < nr_dim; i++){
            for(unsigned int j = 0; j < nr_data; j++){
                double r2 = mat(i,j)*mat(i,j);
                var2 += r2*(v+1)/(v+r2*info);
            }
        }
        var2 /= double(nr_dim*nr_data);
        info = 1.0/var2;

        if(var2/var > 0.99){break;}
        var = var2;
    }
//exit(0);
    //printf("stdval: %f\n",sqrt(var));
}


void DistanceWeightFunction2Tdist::computeModel(double * vec, unsigned int nr_data, unsigned int dim){
    unsigned int nr_inds = nr_data*dim;
    double sum = 0;
    for(unsigned int j = 0; j < nr_inds; j++){sum += vec[j]*vec[j];}
    double var = sum / double(nr_inds);
    info = 1.0/var;

    for(int i = 0; i < 100; i++){
        double var2 = 0;
        for(unsigned int j = 0; j < nr_inds; j++){
            double r2 = vec[j]*vec[j];
            var2 += r2*(v+1)/(v+r2*info);
        }
        var2 /= double(nr_inds);
        info = 1.0/var2;

        if(var2/var > 0.99){break;}
        var = var2;
    }
    //printf("stdval: %f\n",sqrt(var));
}

double  DistanceWeightFunction2Tdist::getProb(double d,bool debugg){
    return (v+1)/(v+(d*d)*info);
}

VectorXd DistanceWeightFunction2Tdist::getProbs(MatrixXd mat){
    const unsigned int nr_data = mat.cols();
    const int nr_dim = mat.rows();
    VectorXd weights = VectorXd(nr_data);
    for(unsigned int j = 0; j < nr_data; j++){
        float desum = 0;
        for(int k = 0; k < nr_dim; k++){
            float di = mat(k,j);
            desum += di*di;
        }
        weights(j) = getProb(sqrt(desum)/sqrt(nr_dim));
    }
	return weights;
}

std::string DistanceWeightFunction2Tdist::getString(){return "Tdist";}

DistanceWeightFunction2 * DistanceWeightFunction2Tdist::clone(){
    DistanceWeightFunction2Tdist * func  = new DistanceWeightFunction2Tdist();
    func->name                      = name;
    func->f                         = f;
    func->p                         = p;
    func->regularization            = regularization;
    func->convergence_threshold     = convergence_threshold;
    func->debugg_print              = debugg_print;
    func->savePath                  = savePath;
    func->meanval                   = meanval;
    func->info                      = info;
    func->v                         = v;
    return func;
}
}


