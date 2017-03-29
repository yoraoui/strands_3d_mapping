#include "weightfunctions/DistanceWeightFunction2JointDist.h"

#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

namespace reglib{


DistanceWeightFunction2JointDist::DistanceWeightFunction2JointDist( int type_ ){
    type =type_;
    reglib::GeneralizedGaussianDistribution * gd = new reglib::GeneralizedGaussianDistribution(true,false,false);
    if(type == 0){gd->power = 2;}
    if(type == 1){gd->power = 1.0;}
    if(type == 2){gd->refine_power = true;}
    gd->nr_refineiters = 6;
    gd->costpen = 10;
    gd->ratio_costpen = 0;
    gd->debugg_print = false;
    reglib::DistanceWeightFunction2PPR3 * sfunc = new reglib::DistanceWeightFunction2PPR3(gd);
    sfunc->startreg                             = 1;
    sfunc->blur                                 = 0.02;//0.03;
    sfunc->data_per_bin                         = 40;
    sfunc->debugg_print                         = false;
    sfunc->max_under_mean                       = false;

    sfunc->reg_shrinkage                        = 0.5;
    sfunc->useIRLSreweight                      = type != 0;
    sfunc->name = "sfunc";

    original        = sfunc;
    first           = true;
    multidim        = true;
    debugg_print    = false;
    regularization  = 1;
}

DistanceWeightFunction2JointDist::DistanceWeightFunction2JointDist(DistanceWeightFunction2 * original_){
    original = original_;
    first = true;
    multidim = true;
}

DistanceWeightFunction2JointDist::~DistanceWeightFunction2JointDist(){
    delete original;
    for(unsigned int i = 0; i < active.size(); i++){delete active[i];}
}

void DistanceWeightFunction2JointDist::setDebugg(bool debugg){
    debugg_print = debugg;
    original->setDebugg(debugg);
    for(unsigned int i = 0; i < active.size(); i++){
       active[i]->setDebugg(debugg);
    }
}

double DistanceWeightFunction2JointDist::getNoise(){
    if(multidim){
        double sum = 0;
        for(unsigned int i = 0; i < active.size(); i++){
            sum += active[i]->getNoise();
        }
        return sum / double(active.size());
    }else{
        return original->getNoise();
    }
}

void DistanceWeightFunction2JointDist::computeModel(double * vec, unsigned int nr_data, unsigned int nr_dim){
    if(multidim){
        for(unsigned int i = 0; i < nr_dim; i++){
            active[i]->computeModel(&(vec[nr_data*i]),nr_data,1);
        }
    }else{
        original->computeModel(vec,nr_data,nr_dim);
    }
}

void DistanceWeightFunction2JointDist::setTune(){
    original->setTune();
    for(unsigned int i = 0; i < active.size(); i++){
        active[i]->setTune();
    }
}

void DistanceWeightFunction2JointDist::computeModel(MatrixXd mat){
    const unsigned int nr_data = mat.cols();
    const unsigned int nr_dim = mat.rows();
    double stddiv = 0;
    for(unsigned int i = 0; i < nr_dim; i++){
        if(multidim && active.size() <= i){active.push_back(original->clone());}

        if(first){
            double current_stddiv = 0;
            for(unsigned int j = 0; j < nr_data; j++){
                current_stddiv += mat(i,j)*mat(i,j);
            }

            if(multidim){
                active[i]->regularization = sqrt(current_stddiv/double(nr_data));
            }
            stddiv += current_stddiv;
        }

        if(multidim){
            active[i]->computeModel(mat.row(i));
        }
        //if(first){printf("regularization: %f\n",active[i]->regularization);}
    }

    if(!multidim){
        if(first){
            original->regularization = sqrt(stddiv/double(nr_data*nr_dim));
        }
        original->computeModel(mat);
    }
    first = false;
    //exit(0);
}

VectorXd DistanceWeightFunction2JointDist::getProbs(MatrixXd mat){
    const unsigned int nr_dim = mat.rows();
    const unsigned int nr_data = mat.cols();
    std::vector<VectorXd> probs;

    VectorXd W (nr_data);

    double pi = 0;

    if(multidim){
        for(unsigned int i = 0; i < nr_dim; i++){
            probs.push_back(active[i]->getProbs(mat.row(i)));
        }
    }else{
        for(unsigned int i = 0; i < nr_dim; i++){
            probs.push_back(original->getProbs(mat.row(i)));
        }
    }

    for(unsigned int i = 0; i < nr_dim; i++){
        pi += probs[i].sum();
    }

    pi /= double(nr_dim*nr_data);
    double po = 1-pi;

    double gamma = pow(pi/po,nr_dim-1);

    for(unsigned int j = 0; j < nr_data; j++){
        double prod_inl = 1;
        double prod_ol = 1;
        for(unsigned int i = 0; i < nr_dim; i++){
            double inl = probs[i](j);
            double ol = 1-inl;

            prod_inl *= inl;
            prod_ol *= ol;
        }

        W(j) = prod_inl / (prod_inl + gamma*prod_ol);
    }

    return W;

}

double DistanceWeightFunction2JointDist::getProb(double d, bool debugg){
    return original->getProb(d,debugg);
}

double DistanceWeightFunction2JointDist::getProbInfront(double d, bool debugg){
    return original->getProbInfront(d,debugg);
}

bool DistanceWeightFunction2JointDist::update(){
    if(multidim){
        for(unsigned int i = 0; i < active.size(); i++){
            active[i]->update();
        }
    }else{
        original->update();
    }
    return true;
}

void DistanceWeightFunction2JointDist::reset(){
    original->reset();
    for(unsigned int i = 0; i < active.size(); i++){
        active[i]->reset();
    }
    first = true;
}

std::string DistanceWeightFunction2JointDist::getString(){
    return original->getString();
}

double DistanceWeightFunction2JointDist::getConvergenceThreshold(){
    if(multidim){
        double sum = 0;
        for(unsigned int i = 0; i < active.size(); i++){
            sum += active[i]->getConvergenceThreshold();
        }
        sum /= double(active.size());
    }else{
        return original->getConvergenceThreshold();
    }
}

DistanceWeightFunction2 * DistanceWeightFunction2JointDist::clone(){
    DistanceWeightFunction2JointDist * func = new DistanceWeightFunction2JointDist(type);
    func->name                      = name;
    func->f                         = f;
    func->p                         = p;
    func->regularization            = regularization;
    func->convergence_threshold     = convergence_threshold;
    func->setDebugg(debugg_print);
    func->savePath                  = savePath;
    func->multidim                  = multidim;
    func->first                     = first;
    return func;
}


double DistanceWeightFunction2JointDist::getWeight(double invstd, double d,double & infoweight, double & prob, bool debugg){
    return original->getWeight(invstd, d,infoweight, prob, debugg);
}

VectorXd DistanceWeightFunction2JointDist::getWeights(std::vector<double > invstdvec, MatrixXd mat, bool debugg){
    const unsigned int nr_dim = mat.rows();
    const unsigned int nr_data = mat.cols();
    std::vector<VectorXd> probs;

    VectorXd W (nr_data);

    double pi = 0;

    if(multidim){
        for(unsigned int i = 0; i < nr_dim; i++){
            probs.push_back(active[i]->getProbs(mat.row(i)));
        }
    }else{
        for(unsigned int i = 0; i < nr_dim; i++){
            probs.push_back(original->getProbs(mat.row(i)));
        }
    }

    for(unsigned int i = 0; i < nr_dim; i++){
        pi += probs[i].sum();
    }

    pi /= double(nr_dim*nr_data);
    double po = 1-pi;

    double gamma = pow(pi/po,nr_dim-1);

    for(unsigned int j = 0; j < nr_data; j++){
        double prod_inl = 1;
        double prod_ol = 1;
        for(unsigned int i = 0; i < nr_dim; i++){
            double inl = probs[i](j);
            double ol = 1-inl;

            prod_inl *= inl;
            prod_ol *= ol;
        }

        W(j) = prod_inl / (prod_inl + gamma*prod_ol);
    }


    double power = 2;
    if(multidim){
        power = 0;
        for(unsigned int i = 0; i < nr_dim; i++){
            power += active[i]->getPower();
        }
        power /= double(nr_dim);
    }else{
        power = original->getPower();
    }

    //printf("power: %f\n",power);

    if(power != 2){
        for(unsigned int j = 0; j < nr_data; j++){
            double de = 0;
            for(unsigned int i = 0; i < nr_dim; i++){
                de += mat(i,j)*mat(i,j);
            }
            de = sqrt(de);

            W(j) *= pow(std::max(0.00001,de),power-2);
        }
    }


    return W;
}

void DistanceWeightFunction2JointDist::print(){
    if(multidim){
        for(unsigned int i = 0; i < active.size(); i++){
            active[i]->print();
        }
    }else{
        original->print();
    }
}


}


