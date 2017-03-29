#include "weightfunctions/DistanceWeightFunction2PPR3.h"

#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

namespace reglib{

DistanceWeightFunction2PPR3::DistanceWeightFunction2PPR3(Distribution * dist_,	double maxd_, int histogram_size_){
    noise_min = 0;
	savePath = "";
	saveData.str("");

	fixed_histogram_size = false;

	useIRLSreweight = false;

	regularization		= 0.1;
	maxd 				= maxd_;
	mind				= 0;
	histogram_size		= histogram_size_;
	starthistogram_size = histogram_size;
	stdval				= blurval;

	maxnoise = 99999999999999;
	noiseval = 100.0;

    prob = new float[histogram_size+1];
	irls = new float[histogram_size+1];
    infront = new float[histogram_size+1];
    histogram = new float[histogram_size+1];
    blur_histogram = new float[histogram_size+1];
    noise = new float[histogram_size+1];
    noisecdf = new float[histogram_size+1];

    prob[histogram_size] = 0;
    infront[histogram_size] = 1;

	mulval = 1;
    meanval = 0;

	startmaxd = maxd_;


	scale_convergence = true;
	nr_inliers = 1;

	target_length = 10.0;

	bidir = false;
	iter = 0;

	maxp = 0.99;

	first = true;


	startreg			= 2;
	blurval				= 4;
	stdval				= 100;
	noiseval			= 100.0;
	debugg_print		= true;
	threshold			= false;
	blur				= 0.03;
	data_per_bin		= 30;

    min_histogram_size  = 50;
    max_histogram_size  = 1000;

	update_size			= true;//(setting&2 != 1);
	max_under_mean		= true;//(setting&1 != 0);
	interp				= true;//(setting&1 != 0);

	nr_refineiters		= 1;
	convergence_threshold = 0.05;
	compute_infront = false;

	sp      = new SignalProcessing();
	dist    = dist_;//new GeneralizedGaussianDistribution(true,true);//GaussianDistribution();//GeneralizedGaussianDistribution();//GaussianDistribution();

    tune = false;
}

DistanceWeightFunction2PPR3::DistanceWeightFunction2PPR3(	double maxd_, int histogram_size_){
    noise_min = 0;

	fixed_histogram_size = false;

	useIRLSreweight = false;

	regularization		= 0.1;
	maxd 				= maxd_;
	mind				= 0;
	histogram_size		= histogram_size_;
	starthistogram_size = histogram_size;
	stdval				= blurval;

	maxnoise = 99999999999999;
	noiseval = 100.0;

    prob = new float[histogram_size+1];
	irls = new float[histogram_size+1];
    infront = new float[histogram_size+1];
    histogram = new float[histogram_size+1];
    blur_histogram = new float[histogram_size+1];
    noise = new float[histogram_size+1];
    noisecdf = new float[histogram_size+1];

    prob[histogram_size] = 0;
    infront[histogram_size] = 1;


	mulval = 1;
	meanval = 0;

	startmaxd = maxd_;


	scale_convergence = true;
	nr_inliers = 1;

	target_length = 10.0;

	bidir = false;
	iter = 0;

	maxp = 0.99;

	first = true;


	startreg			= 2;
	blurval				= 4;
	stdval				= 100;
	noiseval			= 100.0;
	debugg_print		= true;
	threshold			= false;
	blur				= 0.03;
	data_per_bin		= 30;

    min_histogram_size  = 50;
    max_histogram_size  = 1000;

	update_size			= true;//(setting&2 != 1);
	max_under_mean		= true;//(setting&1 != 0);
	interp				= true;//(setting&1 != 0);

	nr_refineiters		= 1;
	convergence_threshold = 0.05;
	compute_infront = false;

	sp      = new SignalProcessing();
	dist    = new GeneralizedGaussianDistribution(true,true);//GaussianDistribution();//GeneralizedGaussianDistribution();//GaussianDistribution();
    tune = false;
}

DistanceWeightFunction2PPR3::~DistanceWeightFunction2PPR3(){
    if(sp != 0)             {delete sp;}
	if(dist != 0)           {delete dist;}
	if(prob != 0)           {delete prob;}
	if(irls != 0)           {delete irls;}
    if(infront != 0)        {delete infront;}
    if(histogram != 0)      {delete histogram;}
    if(blur_histogram != 0) {delete blur_histogram;}
    if(noise != 0)          {delete noise;}
    if(noisecdf != 0)       {delete noisecdf;}
}

double DistanceWeightFunction2PPR3::getNoise(){
	if(bidir){
        return 0.5*fabs(maxd-mind)*dist->getNoise()/double(histogram_size);//0.5*fabs(maxd-mind);
	}else{
        return fabs(maxd-mind)*dist->getNoise()/double(histogram_size);//fabs(maxd-mind);
	}
}

void DistanceWeightFunction2PPR3::recomputeHistogram(float * hist, MatrixXd & mat){
	const unsigned int nr_data = mat.cols();
	const int nr_dim = mat.rows();
	histogram_mul2	= double(histogram_size)/(maxd-mind);
	for(int j = 0; j < histogram_size; j++){hist[j] = 0;}
	for(unsigned int j = 0; j < nr_data; j++){
		for(int k = 0; k < nr_dim; k++){
			double ind = getInd(mat(k,j));
			if(ind >= 0 && (ind+0.00001) < histogram_size){
				hist[int(ind+0.00001)]++;
			}
		}
	}
}

void DistanceWeightFunction2PPR3::recomputeHistogram(float * hist, double * vec, unsigned int nr_data){
    histogram_mul2	= double(histogram_size)/(maxd-mind);
    for(int j = 0; j < histogram_size; j++){hist[j] = 0;}
    for(unsigned int j = 0; j < nr_data; j++){
        //if(vec[j] == 0){continue;}
        double ind = getInd(vec[j]);
        if(ind >= 0 && (ind+0.00001) < histogram_size){
            hist[int(ind+0.00001)]++;
        }
    }
}

void DistanceWeightFunction2PPR3::recomputeProbs(){
	for(int k = 0; k < histogram_size; k++){noise[k] = dist->getval(k);}
	if(compute_infront){
		for(int k = 0; k < histogram_size; k++){noisecdf[k] = dist->getcdf(k);}
    }else{
        for(int k = 0; k < histogram_size; k++){noisecdf[k] = 0;}
    }

	double maxhist = blur_histogram[0];
	for(int k = 1; k < histogram_size; k++){maxhist = std::max(maxhist,double(blur_histogram[k]));}
	double minhist = maxhist*0.01;

	for(int k = 0; k < histogram_size; k++){
		if(max_under_mean && k < dist->mean){	prob[k] = maxp;	}
		else{
			double hs = std::max(minhist,std::max(1.0,double(blur_histogram[k])));
            prob[k] = noise[k]/hs;//never fully trust any data
            if(prob[k] > maxp){prob[k] = maxp;}
            //prob[k] = std::min(maxp , noise[k]/hs);//never fully trust any data
		}
		infront[k] = (1-prob[k])*noisecdf[k];
		irls[k] = dist->getIRLSreweight(k);
	}

    tinl = 0;
    toth = 0;
	for(int k = 0; k < histogram_size; k++){
		tinl += histogram[k]*prob[k];
		toth += histogram[k];
	}


    if(false && debugg_print){
        for(int k = 0; k < histogram_size; k+=10){
            printf("%i -> hist: %f prob[k]: %f noisecdf[k]: %f infront[k]: %f\n",k,double(blur_histogram[k]),prob[k],noisecdf[k],infront[k]);
        }
    }
}

void DistanceWeightFunction2PPR3::setTune(){
    tune = true;
    //dist->costpen = toth/tinl;
}


void DistanceWeightFunction2PPR3::initComputeModel(){
    if( debugg_print){printf("\n%%################################################### ITER:%i ############################################################\n",iter);}
    if(	savePath.size() != 0 && iter == 0){	saveData.str("");}
    if(	savePath.size() != 0){saveData << "\n%################################################### ITER:" << iter << " ############################################################\n";}
    if( iter == 0){dist->reset();}
}


inline void DistanceWeightFunction2PPR3::setInitialNoise(double stdval){
    if(debugg_print){printf("%%stdval: %f\n",stdval);}
    histogram_mul2	= double(histogram_size)/(maxd-mind);

    if(debugg_print){dist->print();}
    dist->setNoise(stdval*histogram_mul2);
    dist->setRegularization(regularization*histogram_mul2);

    if(debugg_print){dist->print();}
}

void DistanceWeightFunction2PPR3::computeModel(double * vec, unsigned int nr_data, unsigned int nr_dim){

    double start_time1 = getTime();
    double init_start_time = start_time1;
    double start_time = start_time1;

    unsigned int nr_inds = nr_data*nr_dim;
    initComputeModel();
//if(debugg_print){printf("%%init time: %7.7fs\n",getTime()-init_start_time);}
    if(!fixed_histogram_size){
        if(first){
            double sum = 0;
            for(unsigned int j = 0; j < nr_inds; j++){sum += vec[j]*vec[j];}
            double var = sum / double(nr_inds);
            double stdval = sqrt(var);

            if(debugg_print){printf("%% START stdval: %7.7fs\n",stdval);}

            if(debugg_print){printf("%% maxd: %7.7fs\n",maxd);}

            dist->power = 2;
            setInitialNoise(stdval);
            //dist->setStart(vec,nr_data,nr_dim);
            //if(debugg_print){printf("%%setInitialNoise time: %7.7fs\n",getTime()-start_time);}
            start_time = getTime();
        }
        if(update_size){
            //if(debugg_print){printf("%%update_size\n");dist->print();}
            double next_maxd;
            double next_mind;
            start_time = getTime();

            //return 0.5*fabs(maxd-mind)*dist->getNoise()/double(histogram_size);

            dist->getMaxdMind(next_maxd,next_mind,0.000001);
            if(first){
                next_maxd *= (maxd-mind)/double(histogram_size);
                next_mind = -next_maxd;
            }else {
                next_maxd = getDfromInd(next_maxd);
                next_mind = getDfromInd(next_mind);
            }
            //if(debugg_print){printf("%%getMaxdMind time: %7.7fs\n",getTime()-start_time);}
            if(!bidir){next_mind = 0;}
            double ratioChange = fabs(maxd-mind)/fabs(next_maxd-next_mind);
            dist->rescale(1.0/ratioChange);
            mind = next_mind;
            maxd = next_maxd;
            //if(debugg_print){printf("%%rescale time: %7.7fs\n",getTime()-start_time);}
            start_time = getTime();

            double nr_inside = 0;
            for(unsigned int j = 0; j < nr_inds; j++){
                double d = vec[j];
                if(mind < d && d < maxd){nr_inside++;}
            }
            //if(debugg_print){printf("%%nr_inside time: %5.5fs\n",getTime()-start_time);}

            int new_histogram_size = std::min(int(starthistogram_size),std::max(min_histogram_size,int(0.5 + nr_inside/data_per_bin)));
            histogram_size = std::min(max_histogram_size,new_histogram_size);
            blurval = blur*double(histogram_size)*float(histogram_size)/float(new_histogram_size);
            //if(debugg_print){printf("%%nr_inside: %f histogram_size: %i blurval: %f\n",nr_inside,histogram_size,blurval);}
        }
        first = false;
    }


   // if(debugg_print){printf("%%full time: %7.7fs line %i\n",getTime()-start_time1,__LINE__);}
    if(debugg_print){printf("%%init time: %7.7fs\n",getTime()-init_start_time);}


    if(debugg_print){printf("%%histogram_size: %5.5f maxd: %5.5f mind: %5.5f\n",double(histogram_size),maxd,mind);}
    start_time = getTime();
    recomputeHistogram(histogram,vec,nr_inds);
    if(debugg_print){printf("%%computing histogram time: %7.7fs\n",getTime()-start_time);}

    if(debugg_print){printf("%%full time: %7.7fs line %i\n",getTime()-start_time1,__LINE__);}

    start_time = getTime();
    sp->process(histogram,blur_histogram,blurval,histogram_size);
    if(debugg_print){printf("%%SignalProcessing time: %7.7fs\n",getTime()-start_time);}
    if(debugg_print){printf("%%full time: %7.7fs line %i\n",getTime()-start_time1,__LINE__);}

    start_time = getTime();
    dist->setRegularization(double(histogram_size)*regularization/(maxd-mind));
    dist->setMinStd(double(histogram_size)*noise_min/(maxd-mind));
    dist->train(blur_histogram,histogram_size);
    dist->update();
    if(debugg_print){printf("%%train time: %7.7fs\n",getTime()-start_time);dist->print();}

    start_time = getTime();
    recomputeProbs();
    if(debugg_print){printf("%%recomputeProbs time: %7.7fs\n",getTime()-start_time);}



//    start_time = getTime();
//    getProbs(mat);
//    if(debugg_print){printf("%%get number of inliers time: %7.7fs nr_inliers: %f\n",getTime()-start_time,nr_inliers);}

//	double inl = 0;
//	double toth = 0;
//	for(int k = 0; k < histogram_size; k++){
//		inl += histogram[k]*prob[k];
//		toth += histogram[k];
//	}

    double pi = toth/tinl;
    double newcostpen1 = std::max(1.0,std::min(pow(pi,1.0),10.0));

    //if(tune){dist->costpen = 0.05*newcostpen1 + 0.95*dist->costpen;}
    if(tune){dist->costpen = newcostpen1;}

	if(debugg_print){printf("%% toth %f inl %f outlier ratio: %f dist->costpen : %f\n ",toth,tinl,toth/tinl,dist->costpen);}
	if(debugg_print){printf("%% newcostpen1: %f \n ",newcostpen1);}


    if(debugg_print){printf("X =           [");			for(int k = 0; k < 3000 && k < histogram_size; k++){printf("%5.5f ",mind + (maxd-mind)*double(k)/double(int(histogram_size)-1) );}		printf("];\n");}
    if(debugg_print){printf("hist =        [");			for(int k = 0; k < 3000 && k < histogram_size; k++){printf("%i ",	int(histogram[k]));}                printf("];\n");}
    if(debugg_print){printf("hist_smooth = [");         for(int k = 0; k < 3000 && k < histogram_size; k++){printf("%i ",	int(blur_histogram[k]));}           printf("];\n");}
    if(debugg_print){printf("noise =       [");			for(int k = 0; k < 3000 && k < histogram_size; k++){printf("%i ",	int(noise[k]));}                    printf("];\n");}
    if(debugg_print && compute_infront){printf("noisecdf = [");		for(int k = 0; k < 3000 && k < histogram_size; k++){printf("%2.2f ",int(noisecdf[k]));}		printf("];\n");}
    if(debugg_print){printf("prob =        [");			for(int k = 0; k < 3000 && k < histogram_size; k++){printf("%2.2f ",prob[k]);}                          printf("];\n");}
    if(debugg_print && compute_infront){printf("infront = [");			for(int k = 0; k < 3000 && k < histogram_size; k++){printf("%2.2f ",infront[k]);}		printf("];\n");}
    if(debugg_print){printf("figure(%i);",iter+1);}
    if(debugg_print){printf("clf; hold on; plot(X,hist_smooth,'r','LineWidth',2); plot(X,hist,'m','LineWidth',2); plot(X,noise,'b','LineWidth',2); plot(X,prob*max(noise),'g','LineWidth',2); ");}
    if(debugg_print && compute_infront){printf("plot(X,infront*max(noise),'g','LineWidth',2);");}

    if(debugg_print){printf("legend('histogram smoothed','histogram raw','noise estimate','max(noise estimate)*P(Inlier)'");}
    if(debugg_print && compute_infront){printf(",'max(noise estimate)*P(Infront)'");}
    if(debugg_print){printf(");\n");}

	if(debugg_print){printf("\n");}
    if(debugg_print){printf("%% getNoise = %7.7f\n",getNoise());}
	if(debugg_print){
		double inl = 0;
		double toth = 0;
		for(int k = 0; k < histogram_size; k++){
			inl += histogram[k]*prob[k];
			toth += histogram[k];
		}
		printf("%% inliers = %5.5f ratio: %5.5f\n",inl,inl/toth);
	}

    if(	savePath.size() != 0){
        saveData << "d = [";
        for(int k = 0; k < histogram_size; k++){saveData << (maxd-mind)*double(k)/double(histogram_size-1)+mind << " ";}
        saveData << "];\n";

        saveData << "hist = [";
        for(int k = 0; k < histogram_size; k++){saveData << histogram[k] << " ";}
        saveData << "];\n";
        saveData << "hist_smooth = [";
        for(int k = 0; k < histogram_size; k++){saveData << blur_histogram[k] << " ";}
        saveData << "];\n";
        saveData << "noise = [";
        for(int k = 0; k < histogram_size; k++){saveData << noise[k] << " ";}
        saveData << "];\n";
        saveData << "noisecdf = [";
        for(int k = 0; k < histogram_size; k++){saveData << noisecdf[k] << " ";}
        saveData << "];\n";
        saveData << "prob = [";
        for(int k = 0; k < histogram_size; k++){saveData << prob[k] << " ";}
        saveData << "];\n";
        saveData << "infront = [";
        for(int k = 0; k < histogram_size; k++){saveData << infront[k] << " ";}
        saveData << "];\n";
        saveData << "figure(" << iter+1 <<");\n";
        saveData << "clf; hold on;\n";
        saveData << "title('Distribution of residuals and corresponding estimates')\n";
        saveData << "xlabel('residual values')\n";
        saveData << "ylabel('number of data')\n";

        saveData << "plot(d,hist_smooth,'r','LineWidth',2);\n";
        saveData << "plot(d,hist,'m','LineWidth',2);\n";
        saveData << "plot(d,noise,'b','LineWidth',2);\n";
        saveData << "plot(prob*max(noise),'g','LineWidth',2);\n";
        if(compute_infront){saveData << "plot(d,infront*max(noise),'c','LineWidth',2);\n";}
        saveData << "legend('histogram smoothed','histogram raw','noise estimate','P(Inlier)'";
        if(compute_infront){saveData << ",'P(Infront)'";}
        saveData << ");\n";
    }

    if(!fixed_histogram_size && update_size ){
        double next_maxd;
        double next_mind;
        start_time = getTime();

        dist->getMaxdMind(next_maxd,next_mind,0.000001);

        if(debugg_print){
            printf("%%getMaxdMind time: %5.5fs\n",getTime()-start_time);
            printf("%%current: %f %f\n",mind,maxd);
            printf("%%next: %f %f\n",next_mind,next_maxd);
        }

        next_maxd = getDfromInd(next_maxd);
        next_mind = getDfromInd(next_mind);
        //next_maxd *= (maxd-mind)/double(histogram_size);
        //next_mind *= (maxd-mind)/double(histogram_size);

        if(debugg_print){    printf("DistanceWeightFunction2PPR3::computeMode -> %s\n",name.c_str());
            printf("%%next: %f %f\n",next_mind,next_maxd);
        }

        if(!bidir){next_mind = 0;}

        //Ensure overlap
        if(next_maxd < mind){		return;}
        if(next_mind > maxd){		return;}
        if(next_maxd == next_mind){	return;}

        if(debugg_print){printf("%%next: %f %f\n",next_mind,next_maxd);}
        double overlap = 0;
        if(next_maxd <= maxd && next_mind >= mind){     overlap = next_maxd-next_mind;}
        else if(next_maxd >= maxd && next_mind <= mind){overlap =      maxd-	 mind;}
        else if(next_maxd <= maxd && next_mind <= mind){overlap = next_maxd-	 mind;}
        else if(next_maxd >= maxd && next_mind >= mind){overlap =      maxd-next_mind;}

        double ratio = 2*overlap/(maxd+next_maxd-mind-next_mind);
        double newlogdiff = log(ratio);
        if(debugg_print){printf("%%overlap: %5.5f ratio: %5.5f newlogdiff: %5.5f\n",overlap,ratio,newlogdiff);}

        if(fabs(newlogdiff) > 0.05 && iter < 10){

            iter++;
            computeModel(vec,nr_data,nr_dim);
            return;
        }else{
            iter = 0;
            if(	savePath.size() != 0){
                std::ofstream myfile;
                myfile.open (savePath);
                myfile << saveData.str();
                myfile.close();
            }
        }
    }else{
        if(	savePath.size() != 0){
            std::ofstream myfile;
            myfile.open (savePath);
            myfile << saveData.str();
            myfile.close();
        }
    }

    if(debugg_print){printf("%%full time: %7.7fs\n",getTime()-start_time1);}
    if(debugg_print){printf("%%stdval:  %7.7f regularization: %7.7f\n",getNoise(),regularization);}
}

void DistanceWeightFunction2PPR3::computeModel(MatrixXd mat){
	const unsigned int nr_data = mat.cols();
	const int nr_dim = mat.rows();
	double start_time;

//    if( debugg_print){printf("\n%%################################################### ITER:%i ############################################################\n",iter);}
//	if(	savePath.size() != 0 && iter == 0){	saveData.str("");}
//	if(	savePath.size() != 0){saveData << "\n%################################################### ITER:" << iter << " ############################################################\n";}
//    if( iter == 0){dist->reset();}

    initComputeModel();

	if(!fixed_histogram_size){
        if(first && (iter == 0)){
            if(debugg_print){printf("%%--------------------\n");}
			double sum = 0;
			for(unsigned int j = 0; j < nr_data; j++){
				for(int k = 0; k < nr_dim; k++){sum += mat(k,j)*mat(k,j);}
			}
            stdval = sqrt(sum / double(nr_data*nr_dim));
            setInitialNoise(stdval);
//			if(debugg_print){printf("%%stdval: %f\n",stdval);}
//            histogram_mul2	= double(histogram_size)/(maxd-mind);
//            dist->setNoise(stdval*histogram_mul2);
		}

		if(update_size){
            if(debugg_print){printf("%%update_size\n");dist->print();}
			double next_maxd;
			double next_mind;
			start_time = getTime();
            dist->getMaxdMind(next_maxd,next_mind,0.000001);

            if(first && (iter == 0)){
                next_maxd *= (maxd-mind)/double(histogram_size);
                next_mind = -next_maxd;
            }else {
                next_maxd = getDfromInd(next_maxd);
                next_mind = getDfromInd(next_mind);
            }
			if(debugg_print){printf("%%getMaxdMind time: %5.5fs\n",getTime()-start_time);}
			if(!bidir){next_mind = 0;}
			double ratioChange = fabs(maxd-mind)/fabs(next_maxd-next_mind);

			dist->rescale(1.0/ratioChange);
			mind = next_mind;
			maxd = next_maxd;

			start_time = getTime();
			double nr_inside = 0;
			for(unsigned int j = 0; j < nr_data; j++){
				for(int k = 0; k < nr_dim; k++){
                    double d = mat(k,j);
                    if(bidir){
                        if(mind < d && d < maxd){nr_inside++;}
                    }else{
                        if(fabs(d) < maxd){nr_inside++;}
                    }
				}
			}
			if(debugg_print){printf("%%nr_inside time: %5.5fs\n",getTime()-start_time);}

            //int new_histogram_size = std::min(int(starthistogram_size),std::max(min_histogram_size,int(0.5 + nr_inside/data_per_bin)));
            int test_histogram_size = int(0.5 + nr_inside/data_per_bin);
            int new_histogram_size = std::min(int(starthistogram_size),std::max(min_histogram_size,test_histogram_size));
            histogram_size = std::min(max_histogram_size,new_histogram_size);
            blurval = blur*double(histogram_size)*float(histogram_size)/float(new_histogram_size);
            //blurval = blur*double(histogram_size)*double(histogram_size)/double(test_histogram_size);

            if(debugg_print){printf("%%test_histogram_size (%i) \n",test_histogram_size);}
            if(debugg_print){printf("%%blur (%f) * histogram_size (%f) * histogram_size(%f)/new_histogram_size(%f))\n",blur,double(histogram_size),float(histogram_size),float(new_histogram_size));}
			if(debugg_print){printf("%%nr_inside: %f histogram_size: %i blurval: %f\n",nr_inside,histogram_size,blurval);}
		}
        //first = false;
	}

//    double nr_inside_fully = 0;
//    for(unsigned int j = 0; j < nr_data; j++){
//        int all_good = 1;
//        for(int k = 0; k < nr_dim; k++){
//            double d = mat(k,j);
//            if(!(mind < d && d < maxd)){all_good = 0; break;}
//        }
//        nr_inside_fully += all_good;
//    }

	if(debugg_print){printf("%%histogram_size: %5.5f maxd: %5.5f mind: %5.5f\n",double(histogram_size),maxd,mind);}
	start_time = getTime();
	recomputeHistogram(histogram,mat);
	if(debugg_print){printf("%%computing histogram time: %5.5fs\n",getTime()-start_time);}

	start_time = getTime();
	sp->process(histogram,blur_histogram,blurval,histogram_size);
	if(debugg_print){printf("%%SignalProcessing time: %5.5fs\n",getTime()-start_time);}

    start_time = getTime();
    dist->setRegularization(0);
    //dist->setRegularization(double(histogram_size)*regularization/(maxd-mind));
	dist->train(blur_histogram,histogram_size);

//    //if(debugg_print && first){printf("first noise: %f\n",getNoise());}
//    if(first){
//        //printf("first noise: %f\n",getNoise());
//        regularization = std::max(0.0,stdval-getNoise());
//    }
    //first = false;

    dist->setRegularization(double(histogram_size)*regularization/(maxd-mind));

	dist->update();
	if(debugg_print){printf("%%train time: %5.5fs\n",getTime()-start_time);dist->print();}

	start_time = getTime();
	recomputeProbs();
	if(debugg_print){printf("%%recomputeProbs time: %5.5fs\n",getTime()-start_time);}

    start_time = getTime();
    //Eigen::VectorXd W = getProbs(mat);

    //nr_inliers
    if(debugg_print){printf("%%get number of inliers time: %5.5fs nr_inliers: %f\n",getTime()-start_time,nr_inliers);}
    //if(debugg_print){printf("%% outlier ratio: %f ",double(nr_inside_fully)/double(nr_inliers));}


//    for(int k = 0; k < histogram_size; k++){
//		tinl += histogram[k]*prob[k];
//        toth += histogram[k];
//    }


    double pi = toth/tinl;
    double newcostpen1 = std::max(1.0,std::min(pow(pi,1.0),10.0));

    //newcostpen1 = 0.05*newcostpen1 + 0.95*dist->costpen;
    //dist->costpen = 10;//0.1*newcostpen1 + 0.9*dist->costpen;
    //dist->costpen = 0.01*newcostpen1 + 0.99*dist->costpen;

    if(tune){
        dist->costpen = newcostpen1;
    }

    if(debugg_print){printf("%% dist->costpen: %5.5f pi: %5.5f newcostpen1: %5.5f\n",dist->costpen,pi,newcostpen1);}

    //dist->costpen = pow(toth/inl,2.0);//pow(double(nr_data)/double(nr_inliers);
	if(debugg_print){printf("%% toth %f inl %f outlier ratio: %f dist->costpen : %f\n ",toth,tinl,toth/tinl,dist->costpen);}
    //if(debugg_print){printf("%% nr_inside_fully: %f inliers: %f ratio: %f\n ",nr_inside_fully,nr_inliers,nr_inliers/nr_inside_fully);}
    if(debugg_print){printf("%% newcostpen1: %f\n ",newcostpen1);}
    if(debugg_print){printf("X =           [");			for(int k = 0; k < 3000 && k < histogram_size; k++){printf("%5.5f ",mind + (maxd-mind)*double(k)/double(int(histogram_size)-1) );}		printf("];\n");}
    if(debugg_print){printf("hist =        [");			for(int k = 0; k < 3000 && k < histogram_size; k++){printf("%i ",	int(histogram[k]));}                    printf("];\n");}
    if(debugg_print){printf("hist_smooth = [");         for(int k = 0; k < 3000 && k < histogram_size; k++){printf("%i ",	int(blur_histogram[k]));}               printf("];\n");}
    if(debugg_print){printf("noise =       [");                 for(int k = 0; k < 3000 && k < histogram_size; k++){printf("%i ",	int(noise[k]));}                        printf("];\n");}
    if(debugg_print && compute_infront){printf("noisecdf = [");		for(int k = 0; k < 3000 && k < histogram_size; k++){printf("%2.2f ",int(noisecdf[k]));}			printf("];\n");}
    if(debugg_print){printf("prob =        [");			for(int k = 0; k < 3000 && k < histogram_size; k++){printf("%2.2f ",prob[k]);}                              printf("];\n");}
    if(debugg_print && compute_infront){printf("infront = [");			for(int k = 0; k < 3000 && k < histogram_size; k++){printf("%2.2f ",infront[k]);}			printf("];\n");}
    if(debugg_print){printf("figure(%i);",iter+1);}
    if(debugg_print){printf("clf; hold on; plot(X,hist_smooth,'r','LineWidth',2); plot(X,hist,'m','LineWidth',2); plot(X,noise,'b','LineWidth',2); plot(X,prob*max(noise),'g','LineWidth',2); ");}
    if(debugg_print && compute_infront){printf("plot(X,infront*max(noise),'g','LineWidth',2);");}
    if(debugg_print){printf("\n");}
    if(debugg_print){printf("legend('histogram smoothed','histogram raw','noise estimate','max(noise estimate)*P(Inlier)'");}
    if(debugg_print && compute_infront){printf(",'max(noise estimate)*P(Infront)'");}
    if(debugg_print){printf(");\n");}


    if(debugg_print){printf("%% getNoise = %7.7f\n",getNoise());}
    if(debugg_print){

		printf("%% inliers = %5.5f ratio: %5.5f\n",tinl,tinl/toth);
    }

	if(	savePath.size() != 0){
		saveData << "d = [";
		for(int k = 0; k < histogram_size; k++){saveData << (maxd-mind)*double(k)/double(histogram_size-1)+mind << " ";}
		saveData << "];\n";

		saveData << "hist = [";
		for(int k = 0; k < histogram_size; k++){saveData << histogram[k] << " ";}
		saveData << "];\n";
		saveData << "hist_smooth = [";
		for(int k = 0; k < histogram_size; k++){saveData << blur_histogram[k] << " ";}
		saveData << "];\n";
		saveData << "noise = [";
		for(int k = 0; k < histogram_size; k++){saveData << noise[k] << " ";}
		saveData << "];\n";
		saveData << "noisecdf = [";
		for(int k = 0; k < histogram_size; k++){saveData << noisecdf[k] << " ";}
		saveData << "];\n";
		saveData << "prob = [";
		for(int k = 0; k < histogram_size; k++){saveData << prob[k] << " ";}
		saveData << "];\n";
		saveData << "infront = [";
		for(int k = 0; k < histogram_size; k++){saveData << infront[k] << " ";}
		saveData << "];\n";
		saveData << "figure(" << iter+1 <<");\n";
		saveData << "clf; hold on;\n";
		saveData << "title('Distribution of residuals and corresponding estimates')\n";
		saveData << "xlabel('residual values')\n";
		saveData << "ylabel('number of data')\n";

		saveData << "plot(d,hist_smooth,'r','LineWidth',2);\n";
		saveData << "plot(d,hist,'m','LineWidth',2);\n";
		saveData << "plot(d,noise,'b','LineWidth',2);\n";
		saveData << "plot(prob*max(noise),'g','LineWidth',2);\n";
		if(compute_infront){saveData << "plot(d,infront*max(noise),'c','LineWidth',2);\n";}
		saveData << "legend('histogram smoothed','histogram raw','noise estimate','P(Inlier)'";
		if(compute_infront){saveData << ",'P(Infront)'";}
		saveData << ");\n";
	}

	if(!fixed_histogram_size && update_size ){

		double next_maxd;
		double next_mind;
		start_time = getTime();

        dist->getMaxdMind(next_maxd,next_mind,0.000001);

		if(debugg_print){
			printf("%%getMaxdMind time: %5.5fs\n",getTime()-start_time);
			printf("%%current: %f %f\n",mind,maxd);
			printf("%%next: %f %f\n",next_mind,next_maxd);
		}

        next_maxd = getDfromInd(next_maxd);
        next_mind = getDfromInd(next_mind);
        //next_maxd *= (maxd-mind)/double(histogram_size);
        //next_mind *= (maxd-mind)/double(histogram_size);

        if(debugg_print){
            printf("%%next: %f %f\n",next_mind,next_maxd);
        }

		if(!bidir){next_mind = 0;}

		//Ensure overlap
        if(next_maxd < mind){ 		return;}
        if(next_mind > maxd){		return;}
        if(next_maxd == next_mind){	return;}

		if(debugg_print){printf("%%next: %f %f\n",next_mind,next_maxd);}
		double overlap = 0;
		if(next_maxd <= maxd && next_mind >= mind){     overlap = next_maxd-next_mind;}
		else if(next_maxd >= maxd && next_mind <= mind){overlap =      maxd-	 mind;}
		else if(next_maxd <= maxd && next_mind <= mind){overlap = next_maxd-	 mind;}
		else if(next_maxd >= maxd && next_mind >= mind){overlap =      maxd-next_mind;}

		double ratio = 2*overlap/(maxd+next_maxd-mind-next_mind);
		double newlogdiff = log(ratio);
		if(debugg_print){printf("%%overlap: %5.5f ratio: %5.5f newlogdiff: %5.5f\n",overlap,ratio,newlogdiff);}

        if(fabs(newlogdiff) > 0.025 && iter < 5){

			iter++;
			computeModel(mat);
			return;
		}else{
			iter = 0;
			if(	savePath.size() != 0){
				std::ofstream myfile;
				myfile.open (savePath);
				myfile << saveData.str();
				myfile.close();
			}
		}
	}else{
		if(	savePath.size() != 0){
			std::ofstream myfile;
			myfile.open (savePath);
			myfile << saveData.str();
			myfile.close();
		}
	}

    first = false;
}

VectorXd DistanceWeightFunction2PPR3::getProbs(MatrixXd mat){
	const unsigned int nr_data = mat.cols();
	const int nr_dim = mat.rows();

	nr_inliers = 0;
	VectorXd weights = VectorXd(nr_data);
	for(unsigned int j = 0; j < nr_data; j++){

        double inl  = 1;//tinl;
        double ninl = 1;//toth;

		double norm = inl+ninl;
		inl /= norm;
		ninl /= norm;

        double desum = 0;
		for(int k = 0; k < nr_dim; k++){
            double di = mat(k,j);
			desum += di*di;
            double p = getProbInp(di);
            inl *= p;
            ninl *= 1.0-p;
		}
        double d = inl / (inl+ninl);
		nr_inliers += d;
        weights(j) = d;
	}

//    for(unsigned int j = 0; j < nr_data; j+= nr_data-1)
//    {
//        int j = 0;
//        double inl  = tinl;
//        double ninl = toth;


//        printf("tinl %12.12f toth %12.12f\n",tinl,toth);

//        double norm = inl+ninl;
//        inl /= norm;
//        ninl /= norm;

//        double desum = 0;
//        for(int k = 0; k < nr_dim; k++){
//            double di = mat(k,j);
//            desum += di*di;
//            double p = getProbInp(di,true);
//            inl *= p;
//            ninl *= 1.0-p;

//            printf("inl %12.12f ninl %12.12f\n",inl,ninl);
//        }
//        double d = inl / (inl+ninl);
//        printf("%i -> %12.12f\n",j,d);
//    }
	if(threshold){
		for(unsigned int j = 0; j < nr_data; j++){weights(j) = weights(j) > 0.5;}
	}
	return weights;
}

double DistanceWeightFunction2PPR3::getProbInp(double d, bool debugg){
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

    if(debugg){printf("d: %9.9f -> ind: %7.7f -> p: %12.12f histogramsize: %7.7i maxd: %9.9f\n",d,ind,p,histogram_size,maxd);}
	return p;
}

double DistanceWeightFunction2PPR3::getProb(double d, bool debugg){
	float p = getProbInp(d,debugg);
	float irlsw = 1;
	if(useIRLSreweight){irlsw = getIRLS(d);}

//	double priorInlier = tinl/toth;
//	double priorOutlier = 1-priorInlier;

//	double np = 1-p;
//	p *= priorInlier;
//	np *= priorOutlier;

//	p = p /(p+np);

//printf("p: %f\n",p);
	//if(debugg){printf("d: %5.5f -> ind: %5.5f -> p: %5.5f irlsw: %5.5f histogramsize: %5.5i maxd: %5.5f\n",d,ind,p,irlsw,histogram_size,maxd);}
	return irlsw*p;
}

double DistanceWeightFunction2PPR3::getIRLS(double d, bool debugg){
	double ind = getInd(d,debugg);
	float p = 0;
	if(interp){
		double w2 = ind-int(ind);
		double w1 = 1-w2;
		if(ind >= 0 && (ind+1) < histogram_size){
			p = irls[int(ind)]*w1 + irls[int(ind+1)]*w2;
		}
	}else{
		if(ind >= 0 && ind < histogram_size){
			p = irls[int(ind)];
		}
	}

	if(debugg){printf("d: %5.5f -> ind: %5.5f -> p: %5.5f histogramsize: %5.5i maxd: %5.5f\n",d,ind,p,histogram_size,maxd);}
	return p;
}

double DistanceWeightFunction2PPR3::getProbInfront(double d, bool debugg){
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
}



double DistanceWeightFunction2PPR3::getProbInfront(double start, double stop, bool debugg){
    return getProbInfront(start,debugg);

    printf("double DistanceWeightFunction2PPR3::getProbInfront(double start, double stop){ not implemented\n");
    double pinf = getProbInfront(start);
    printf("%f %f-> %f\n",start,stop,pinf);

    double ind_start = getInd(start,debugg);
    double w2_start = ind_start-int(ind_start);
    double w1_start = 1-w2_start;


    double ind_stop = getInd(stop,debugg);
    double w2_stop = ind_stop-int(ind_stop);
    double w1_stop = 1-w2_stop;

    printf("histogram_size: %i\n",int(histogram_size));
    printf("start: %f -> %f %f",ind_start,w2_start,w1_start);
    printf("  stop: %f -> %f %f \n",ind_stop,w2_stop,w1_stop);

    //infront[k] = (1-prob[k])*noisecdf[k];

    if(pinf > 0.5){exit(0);}
    return pinf;
}

bool DistanceWeightFunction2PPR3::update(){
    if(debugg_print){printf("%% ============================================================ update() =============================================================\n");}
	double start_time;

	double old_sum_prob = 0;
	for(int k = 0; k < histogram_size; k++){old_sum_prob += prob[k] * histogram[k];}

    int endLen;
    for(endLen = 0; endLen < histogram_size; endLen++){
        if(prob[endLen] < 0.01){break;}
    }

    if(debugg_print){
        printf("p = [");for(int k = 0; k < endLen; k++){printf(" %2.2f",prob[k]);}printf("];\n");
        printf("h = [");for(int k = 0; k < endLen; k++){printf(" %4.4i",int(histogram[k]));}printf("];\n");
        printf("----\n");
    }

	int iteration = 0;
	double regularization_before = regularization;
	for(int i = 0; i < 500; i++){
		iteration++;
        regularization *= reg_shrinkage;

		start_time = getTime();
        double before_noise = getNoise();

		dist->setRegularization(double(histogram_size)*regularization/(maxd-mind));
		dist->update();
        double after_noise = getNoise();

        //printf("%%before_noise: %5.5f after_noise: %5.5f regularization:%5.5f \n",before_noise,after_noise,regularization);

        //double noise_before = func->getNoise()*func->dist->getNoise()/double(func->histogram_size);//func->dist->getNoise();

        //if(debugg_print){printf("%%before_noise: %5.5fs after_noise: %5.5fs\n",before_noise,after_noise);}

        //if(debugg_print){printf("%%train time: %5.5fs\n",getTime()-start_time);dist->print();}

		start_time = getTime();
		recomputeProbs();
        //if(debugg_print){printf("%%recomputeProbs time: %5.5fs\n",getTime()-start_time);}

        for(endLen = 0; endLen < histogram_size; endLen++){
            if(prob[endLen] < 0.01){break;}
        }

        if(debugg_print){
            printf("p = [");for(int k = 0; k < endLen; k++){printf(" %2.2f",prob[k]);}printf("];\n");
        }



		if(0.99*before_noise < after_noise){return true;}

		double new_sum_prob = 0;
        for(int k = 0; k < endLen; k++){new_sum_prob += prob[k] * histogram[k];}

        //if(debugg_print){printf("%% new_sum_prob: %5.5f old_sum_prob: %5.5f regularization: %5.5f regularization_before: %5.5f\n",new_sum_prob,old_sum_prob,regularization,regularization_before);}

        //printf("new_sum_prob %f old_sum_prob %f ratio: %f\n",new_sum_prob,old_sum_prob,new_sum_prob/old_sum_prob);
        if( 0.5*before_noise < after_noise || new_sum_prob < 0.99*old_sum_prob || regularization*10 < regularization_before){return false;}
	}
	return true;
}

void DistanceWeightFunction2PPR3::reset(){
	nr_inliers = 9999999;
	regularization	= startreg;
	maxd			= startmaxd;
	if(bidir){mind = -maxd;}
	else{mind = 0;}
	histogram_size	= starthistogram_size;

    dist->costpen = 10.0;
    tune = false;

	stdval		= maxd/target_length;
	stdval2		= maxd/target_length;
	meanval		= 0;
	meanval2	= 0;
	iter = 0;
	first = true;

}

std::string DistanceWeightFunction2PPR3::getString(){
	char buf [1024];
    sprintf(buf,"PPR3_%i_%i_%i_%i",int(1000.0*startreg),int(1000.0*blur),interp,int(data_per_bin));
	return std::string(buf);
}

double DistanceWeightFunction2PPR3::getConvergenceThreshold(){
	if(scale_convergence){
		return convergence_threshold*getNoise()/sqrt(nr_inliers);
	}else{
		return convergence_threshold;
	}
}

inline double DistanceWeightFunction2PPR3::getInd(double d, bool debugg){
    if(bidir){
        return 0.00001+(d-mind)*histogram_mul2;
    }else{
        return 0.00001+fabs(d)*histogram_mul2;
    }
}

inline double DistanceWeightFunction2PPR3::getDfromInd(double ind, bool debugg){
    if(bidir){
        return ind/histogram_mul2+mind;
    }else{
        return ind/histogram_mul2;
    }
}

DistanceWeightFunction2 * DistanceWeightFunction2PPR3::clone(){
    DistanceWeightFunction2PPR3 * func = new DistanceWeightFunction2PPR3(dist->clone(),startmaxd,starthistogram_size);
    func->name                      = name;
    func->f                         = f;
    func->p                         = p;
    func->regularization            = regularization;
    func->convergence_threshold     = convergence_threshold;
    func->debugg_print              = debugg_print;
    func->savePath                  = savePath;

    if(func->sp != 0)             {delete func->sp;}
    func->sp = sp->clone();

    func->fixed_histogram_size = fixed_histogram_size;
    func->stdval = stdval;
    func->stdval2 = stdval2;
    func->mulval = mulval;
    func->meanval = meanval;
    func->meanval2 = meanval2;
    func->maxnoise = maxnoise;
    func->zeromean = zeromean;
    func->costpen = costpen;
    func->maxp = maxp;
    func->first = first;
    func->update_size = update_size;
    func->target_length = target_length;
    func->data_per_bin = data_per_bin;
    func->meanoffset = meanoffset;
    func->blur = blur;
    func->startmaxd = startmaxd;
    func->maxd = maxd;
    func->mind = mind;
    func->noise_min = noise_min;
    func->histogram_size = histogram_size;
    func->starthistogram_size = starthistogram_size;
    func->blurval = blurval;
    func->stdgrow = stdgrow;
    func->noiseval = noiseval;
    func->startreg = startreg;
    func->ggd = ggd;
    func->compute_infront = compute_infront;

    func->nr_refineiters = nr_refineiters;
    func->refine_mean = refine_mean;
    func->refine_mul = refine_mul;
    func->refine_std = refine_std;
    func->threshold = threshold;
    func->uniform_bias = uniform_bias;
    func->scale_convergence = scale_convergence;
    func->nr_inliers = nr_inliers;
    func->rescaling = rescaling;
    func->interp = interp;
    func->max_under_mean = max_under_mean;
    func->bidir = bidir;
    func->iter = iter;
    func->histogram_mul = histogram_mul;
    func->histogram_mul2 = histogram_mul2;
    func->min_histogram_size = min_histogram_size;
    func->max_histogram_size = max_histogram_size;
    func->reg_shrinkage = reg_shrinkage;
	func->useIRLSreweight = useIRLSreweight;
    func->tune = tune;
    return func;
}


double DistanceWeightFunction2PPR3::getWeight(double invstd, double d,double & infoweight, double & prob, bool debugg){
	double dscaled = d*invstd;
	double invnoise = invstd/getNoise();
	double power = dist->power;
	if(!useIRLSreweight || power == 2){
		infoweight = invnoise*invnoise;
		prob = getProb(d*invstd,debugg);
		return infoweight*prob;
	}else{
		infoweight = pow(invnoise,power);
		prob = getProb(d*invstd,debugg);
		return infoweight*prob;
		return pow(invnoise,power)*getProb(dscaled,debugg)/pow(std::max(0.0001,dscaled),2.0-power);
	}
}


VectorXd DistanceWeightFunction2PPR3::getWeights(std::vector<double > invstdvec, MatrixXd mat, bool debugg){
    VectorXd probs = getProbs(mat);

    //printf("probs(0) = %10.10f\n",probs(0));

    double noise = 1.0/getNoise();
    double power = dist->power;

    double mind = 0.0001;
    double initw = std::pow(mind,2-power);

    int rows = mat.rows();
    int cols = mat.cols();
    for(unsigned int i = 0; i < cols; i++){
        double invstd = invstdvec[i];
        double d = 0;
        for(unsigned int j = 0; j < rows; j++){
            double dscaled2 = mat(j,i);
            d += dscaled2*dscaled2;
        }
        double dscaled = sqrt(dscaled)*invstd;
        double invnoise = invstd*noise;
        if(!useIRLSreweight || power == 2){
            double infoweight = invnoise*invnoise;
            probs(i) *= infoweight;
        }else{
            probs(i) *= pow(invnoise,power)/pow(std::max(0.0001,dscaled),2.0-power);
            //probs(i) = initw/std::pow(std::max(fabs(d),mind),2-power);
        }
    }
    //printf("probs(0) = %10.10f invst[0] = %10.10f invnoise %10.10f infoweight: %10.10f\n",probs(0),invstdvec[0],invstdvec[0]*noise,invstdvec[0]*noise*invstdvec[0]*noise);



    if(debugg){
        for(double d = 0; d <= 0.2; d+=0.001){
            printf("D: %4.4f -> prob: %5.5f pweight: %5.5f total: %5.5f\n",d,pow(getProb(d),rows),initw/std::pow(std::max(fabs(d),mind),2-power),pow(getProb(d),rows)*initw/std::pow(std::max(fabs(d),mind),2-power));
        }
        exit(0);
    }

    return probs;
}


void DistanceWeightFunction2PPR3::print(){
    dist->print();
}

double DistanceWeightFunction2PPR3::getPower(){
    return dist->power;
}


}


