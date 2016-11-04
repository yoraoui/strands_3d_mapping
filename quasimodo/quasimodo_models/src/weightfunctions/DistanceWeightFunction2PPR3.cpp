#include "weightfunctions/DistanceWeightFunction2PPR3.h"

#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

double getTime3(){
	struct timeval start1;
	gettimeofday(&start1, NULL);
	return double(start1.tv_sec+(start1.tv_usec/1000000.0));
}

namespace reglib{

DistanceWeightFunction2PPR3::DistanceWeightFunction2PPR3(Distribution * dist_,	double maxd_, int histogram_size_){
	savePath = "";
	saveData.str("");

	fixed_histogram_size = false;

	regularization		= 0.1;
	maxd 				= maxd_;
	mind				= 0;
	histogram_size		= histogram_size_;
	starthistogram_size = histogram_size;
	stdval				= blurval;

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


	update_size			= true;//(setting&2 != 1);
	max_under_mean		= true;//(setting&1 != 0);
	interp				= true;//(setting&1 != 0);

	nr_refineiters		= 1;
	convergence_threshold = 0.05;
	compute_infront = false;

	sp      = new SignalProcessing();
	dist    = dist_;//new GeneralizedGaussianDistribution(true,true);//GaussianDistribution();//GeneralizedGaussianDistribution();//GaussianDistribution();
}

DistanceWeightFunction2PPR3::DistanceWeightFunction2PPR3(	double maxd_, int histogram_size_){

	fixed_histogram_size = false;

	regularization		= 0.1;
	maxd 				= maxd_;
	mind				= 0;
	histogram_size		= histogram_size_;
	starthistogram_size = histogram_size;
	stdval				= blurval;

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


	update_size			= true;//(setting&2 != 1);
	max_under_mean		= true;//(setting&1 != 0);
	interp				= true;//(setting&1 != 0);

	nr_refineiters		= 1;
	convergence_threshold = 0.05;
	compute_infront = false;

	sp      = new SignalProcessing();
	dist    = new GeneralizedGaussianDistribution(true,true);//GaussianDistribution();//GeneralizedGaussianDistribution();//GaussianDistribution();
}

DistanceWeightFunction2PPR3::~DistanceWeightFunction2PPR3(){
	if(sp != 0){delete sp;}
	if(dist != 0){delete dist;}
}

double DistanceWeightFunction2PPR3::getNoise(){return regularization+noiseval;}

void DistanceWeightFunction2PPR3::recomputeHistogram(std::vector<float> & hist, MatrixXd & mat){
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

void DistanceWeightFunction2PPR3::recomputeProbs(){
	for(int k = 0; k < histogram_size; k++){noise[k] = dist->getval(k);}
	if(compute_infront){
		for(int k = 0; k < histogram_size; k++){noisecdf[k] = dist->getcdf(k);}
	}

	double maxhist = blur_histogram[0];
	for(int k = 1; k < histogram_size; k++){maxhist = std::max(maxhist,double(blur_histogram[k]));}
	double minhist = maxhist*0.01;

	for(int k = 0; k < histogram_size; k++){
		if(max_under_mean && k < dist->mean){	prob[k] = maxp;	}
		else{
			double hs = std::max(minhist,std::max(1.0,double(blur_histogram[k])));
			prob[k] = std::min(maxp , noise[k]/hs);//never fully trust any data
		}
        infront[k] = (1-prob[k])*noisecdf[k];
	}

    if(false && debugg_print){
        for(int k = 0; k < histogram_size; k+=10){
            printf("%i -> hist: %f prob[k]: %f noisecdf[k]: %f infront[k]: %f\n",k,double(blur_histogram[k]),prob[k],noisecdf[k],infront[k]);
        }
    }
    //exit(0);
}

void DistanceWeightFunction2PPR3::computeModel(MatrixXd mat){
	const unsigned int nr_data = mat.cols();
	const int nr_dim = mat.rows();
	double start_time;

	if(debugg_print){printf("\n%%################################################### ITER:%i ############################################################\n",iter);}

	if(	savePath.size() != 0 && iter == 0){	saveData.str("");}

	if(	savePath.size() != 0){
		saveData << "\n%################################################### ITER:" << iter << " ############################################################\n";
	}



	if(!fixed_histogram_size){
		if(first){

			double sum = 0;
			for(unsigned int j = 0; j < nr_data; j++){
				for(int k = 0; k < nr_dim; k++){sum += mat(k,j)*mat(k,j);}
			}
			double stdval = sqrt(sum / double(nr_data*nr_dim));
			if(debugg_print){printf("%%stdval: %f\n",stdval);}
            histogram_mul2	= double(histogram_size)/(maxd-mind);
            dist->setNoise(stdval*histogram_mul2);
		}

		if(update_size){

            if(debugg_print){printf("%%update_size\n");dist->print();}
			double next_maxd;
			double next_mind;
			start_time = getTime();
            dist->getMaxdMind(next_maxd,next_mind,0.0000001);
//			next_maxd *= (maxd-mind)/double(histogram_size);
//			next_mind *= (maxd-mind)/double(histogram_size);
            if(first){
                next_maxd *= (maxd-mind)/double(histogram_size);
                next_mind = -next_maxd;
            }else {
                next_maxd = getDfromInd(next_maxd);
                next_mind = getDfromInd(next_mind);
            }
			if(debugg_print){printf("%%getMaxdMind time: %5.5fs\n",getTime()-start_time);}
			if(!bidir){next_mind = 0;}
			mind = next_mind;
			maxd = next_maxd;

			start_time = getTime();
			double nr_inside = 0;
			for(unsigned int j = 0; j < nr_data; j++){
				for(int k = 0; k < nr_dim; k++){
					double d = mat(k,j);
					if(mind < d && d < maxd){nr_inside++;}
				}
			}
			if(debugg_print){printf("%%nr_inside time: %5.5fs\n",getTime()-start_time);}

			int new_histogram_size = std::min(int(histogram.size()),std::max(50,int(0.5 + nr_inside/data_per_bin)));
			histogram_size = std::min(1000,new_histogram_size);
			blurval = blur*double(histogram_size)*float(histogram_size)/float(new_histogram_size);
			if(debugg_print){printf("%%nr_inside: %f histogram_size: %i blurval: %f\n",nr_inside,histogram_size,blurval);}
		}
        first = false;
	}

	if(debugg_print){printf("%%histogram_size: %5.5f maxd: %5.5f mind: %5.5f\n",double(histogram_size),maxd,mind);}
	start_time = getTime();
	recomputeHistogram(histogram,mat);
	if(debugg_print){printf("%%computing histogram time: %5.5fs\n",getTime()-start_time);}

	start_time = getTime();
	sp->process(histogram,blur_histogram,blurval,histogram_size);
	if(debugg_print){printf("%%SignalProcessing time: %5.5fs\n",getTime()-start_time);}

	start_time = getTime();
	dist->setRegularization(double(histogram_size)*regularization/(maxd-mind));
	dist->train(blur_histogram,histogram_size);
	dist->update();
	if(debugg_print){printf("%%train time: %5.5fs\n",getTime()-start_time);dist->print();}

	start_time = getTime();
	recomputeProbs();
	if(debugg_print){printf("%%recomputeProbs time: %5.5fs\n",getTime()-start_time);}

	start_time = getTime();
	getProbs(mat);
	if(debugg_print){printf("%%get number of inliers time: %5.5fs nr_inliers: %f\n",getTime()-start_time,nr_inliers);}

	if(debugg_print){printf("hist = [");			for(int k = 0; k < 3000 && k < histogram_size; k++){printf("%i ",	int(histogram[k]));}		printf("];\n");}
	if(debugg_print){printf("hist_smooth = [");		for(int k = 0; k < 3000 && k < histogram_size; k++){printf("%i ",	int(blur_histogram[k]));}	printf("];\n");}
	if(debugg_print){printf("noise = [");			for(int k = 0; k < 3000 && k < histogram_size; k++){printf("%i ",	int(noise[k]));}			printf("];\n");}
	if(debugg_print){printf("noisecdf = [");		for(int k = 0; k < 3000 && k < histogram_size; k++){printf("%2.2f ",int(noisecdf[k]));}			printf("];\n");}
	if(debugg_print){printf("prob = [");			for(int k = 0; k < 3000 && k < histogram_size; k++){printf("%2.2f ",prob[k]);}					printf("];\n");}
	if(debugg_print){printf("infront = [");			for(int k = 0; k < 3000 && k < histogram_size; k++){printf("%2.2f ",infront[k]);}				printf("];\n");}
    if(debugg_print){printf("figure(%i);",iter+1);}
    if(debugg_print){printf("clf; hold on; plot(hist_smooth,'r','LineWidth',2); plot(hist,'m','LineWidth',2); plot(noise,'b','LineWidth',2); plot(prob*max(noise),'g','LineWidth',2); plot(infront*max(noise),'g','LineWidth',2);\n");}

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

        dist->getMaxdMind(next_maxd,next_mind,0.0000001);

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

        if(fabs(newlogdiff) > 0.05 && iter < 5){
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
}

VectorXd DistanceWeightFunction2PPR3::getProbs(MatrixXd mat){
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

double DistanceWeightFunction2PPR3::getProb(double d, bool debugg){
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

bool DistanceWeightFunction2PPR3::update(){
	double start_time;

	double old_sum_prob = 0;
	for(int k = 0; k < histogram_size; k++){old_sum_prob += prob[k] * histogram[k];}

	int iteration = 0;
	double regularization_before = regularization;
	for(int i = 0; i < 500; i++){
		iteration++;
		regularization *= 0.5;

		start_time = getTime();
		double before_noise = dist->getNoise();
		dist->setRegularization(double(histogram_size)*regularization/(maxd-mind));
		dist->update();
		double after_noise = dist->getNoise();
		if(debugg_print){printf("%%train time: %5.5fs\n",getTime()-start_time);dist->print();}

		start_time = getTime();
		recomputeProbs();
		if(debugg_print){printf("%%recomputeProbs time: %5.5fs\n",getTime()-start_time);}

		if(0.99*before_noise < after_noise){return true;}

		double new_sum_prob = 0;
		for(int k = 0; k < histogram_size; k++){new_sum_prob += prob[k] * histogram[k];}

		//printf("new_sum_prob %f old_sum_prob %f ratio: %f\n",new_sum_prob,old_sum_prob,new_sum_prob/old_sum_prob);
		if(new_sum_prob < 0.99*old_sum_prob || regularization*10 < regularization_before){return false;}
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

	stdval		= maxd/target_length;
	stdval2		= maxd/target_length;
	meanval		= 0;
	meanval2	= 0;
	iter = 0;
	first = true;

}

std::string DistanceWeightFunction2PPR3::getString(){
	char buf [1024];
	sprintf(buf,"PPR_%i_%i_%i_%i",int(1000.0*startreg),int(1000.0*blur),interp,int(data_per_bin));
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
}


