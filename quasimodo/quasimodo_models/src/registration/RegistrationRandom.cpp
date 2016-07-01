#include "registration/RegistrationRandom.h"
#include <iostream>
#include <fstream>
#include <omp.h>

namespace reglib
{

RegistrationRandom::RegistrationRandom(){
	only_initial_guess		= false;
	visualizationLvl = 1;
	refinement = new RegistrationRefinement();
}
RegistrationRandom::~RegistrationRandom(){
	delete refinement;
}

void RegistrationRandom::setSrc(CloudData * src_){
	src = src_;
	refinement->setSrc(src_);
}
void RegistrationRandom::setDst(CloudData * dst_){
	dst = dst_;
	refinement->setDst(dst_);
}

double getTime(){
	struct timeval start1;
	gettimeofday(&start1, NULL);
	return double(start1.tv_sec+(start1.tv_usec/1000000.0));
}

double transformationdiff(Eigen::Affine3d & A, Eigen::Affine3d & B, double rotationweight){
	Eigen::Affine3d C = A.inverse()*B;
	double r = fabs(1-C(0,0))+fabs(C(0,1))+fabs(C(0,2))  +  fabs(C(1,0))+fabs(1-C(1,1))+fabs(C(1,2))  +  fabs(C(2,0))+fabs(C(2,1))+fabs(1-C(2,2));
	double t = sqrt(C(0,3)*C(0,3)+C(1,3)*C(1,3)+C(2,3)*C(2,3));
	return r*rotationweight+t;
}


bool compareFusionResults (FusionResults i,FusionResults j) { return i.score > j.score; }

bool RegistrationRandom::issame(FusionResults fr1, FusionResults fr2, int stepxsmall){
	Eigen::Matrix4d current_guess = fr1.guess.inverse()*fr2.guess;
	float m00 = current_guess(0,0); float m01 = current_guess(0,1); float m02 = current_guess(0,2); float m03 = current_guess(0,3);
	float m10 = current_guess(1,0); float m11 = current_guess(1,1); float m12 = current_guess(1,2); float m13 = current_guess(1,3);
	float m20 = current_guess(2,0); float m21 = current_guess(2,1); float m22 = current_guess(2,2); float m23 = current_guess(2,3);

	double sum = 0;
	double count = 0;
	unsigned int s_nr_data = src->data.cols();
	for(unsigned int i = 0; i < s_nr_data/stepxsmall; i++){
		float x		= src->data(0,i*stepxsmall);
		float y		= src->data(1,i*stepxsmall);
		float z		= src->data(2,i*stepxsmall);
		float dx	= m00*x + m01*y + m02*z + m03 - x;
		float dy	= m10*x + m11*y + m12*z + m13 - y;
		float dz	= m20*x + m21*y + m22*z + m23 - z;
		sum += sqrt(dx*dx+dy*dy+dz*dz);
		count++;
	}
	double mean = sum/count;
	//printf("mean: %f\n",mean);
	return mean < 20*fr1.stop;
}

FusionResults RegistrationRandom::getTransform(Eigen::MatrixXd guess){

	unsigned int s_nr_data = src->data.cols();//std::min(int(src->data.cols()),int(500000));
	unsigned int d_nr_data = dst->data.cols();
    refinement->allow_regularization = true;
	//printf("s_nr_data: %i d_nr_data: %i\n",s_nr_data,d_nr_data);

	int stepy = std::max(1,int(d_nr_data)/100000);

	Eigen::Matrix<double, 3, Eigen::Dynamic> Y;
	Eigen::Matrix<double, 3, Eigen::Dynamic> N;
	Y.resize(Eigen::NoChange,d_nr_data/stepy);
	N.resize(Eigen::NoChange,d_nr_data/stepy);
	unsigned int ycols = Y.cols();

	for(unsigned int i = 0; i < d_nr_data/stepy; i++){
		Y(0,i)	= dst->data(0,i*stepy);
		Y(1,i)	= dst->data(1,i*stepy);
		Y(2,i)	= dst->data(2,i*stepy);
		N(0,i)	= dst->normals(0,i*stepy);
		N(1,i)	= dst->normals(1,i*stepy);
		N(2,i)	= dst->normals(2,i*stepy);
	}

	/// Build kd-tree
	//nanoflann::KDTreeAdaptor<Eigen::Matrix3Xd, 3, nanoflann::metric_L2_Simple>                  kdtree(Y);

	Eigen::VectorXd DST_INORMATION = Eigen::VectorXd::Zero(Y.cols());
	for(unsigned int i = 0; i < d_nr_data/stepy; i++){DST_INORMATION(i) = dst->information(0,i*stepy);}

	double s_mean_x = 0;
	double s_mean_y = 0;
	double s_mean_z = 0;
	for(unsigned int i = 0; i < s_nr_data; i++){
		s_mean_x += src->data(0,i);
		s_mean_y += src->data(1,i);
		s_mean_z += src->data(2,i);
	}
	s_mean_x /= double(s_nr_data);
	s_mean_y /= double(s_nr_data);
	s_mean_z /= double(s_nr_data);

	double d_mean_x = 0;
	double d_mean_y = 0;
	double d_mean_z = 0;
	for(unsigned int i = 0; i < d_nr_data; i++){
        d_mean_x += dst->data(0,i);
		d_mean_y += dst->data(1,i);
		d_mean_z += dst->data(2,i);
	}
	d_mean_x /= double(d_nr_data);
	d_mean_y /= double(d_nr_data);
	d_mean_z /= double(d_nr_data);

	Eigen::Affine3d Ymean = Eigen::Affine3d::Identity();
	Ymean(0,3) = d_mean_x;
	Ymean(1,3) = d_mean_y;
	Ymean(2,3) = d_mean_z;

	Eigen::Affine3d Xmean = Eigen::Affine3d::Identity();
	Xmean(0,3) = s_mean_x;
	Xmean(1,3) = s_mean_y;
	Xmean(2,3) = s_mean_z;


	int stepxsmall = std::max(1,int(s_nr_data)/250);
	Eigen::VectorXd Wsmall (s_nr_data/stepxsmall);
	for(unsigned int i = 0; i < s_nr_data/stepxsmall; i++){Wsmall(i) = src->information(0,i*stepxsmall);}


	double sumtime = 0;
	double sumtimeSum = 0;
	double sumtimeOK = 0;

    refinement->viewer = viewer;
	refinement->visualizationLvl = 0;

	std::vector< double > rxs;
	std::vector< double > rys;
	std::vector< double > rzs;
	double step = 0.1+2.0*M_PI/5;
	for(double rx = 0; rx < 2.0*M_PI; rx += step){
		for(double ry = 0; ry < 2.0*M_PI; ry += step){
			for(double rz = 0; rz < 2.0*M_PI; rz += step){
				rxs.push_back(rx);
				rys.push_back(ry);
				rzs.push_back(rz);
			}
		}
	}


	refinement->target_points = 250;
	unsigned int nr_r = rxs.size();

	std::vector<FusionResults> fr_X;
	fr_X.resize(nr_r);

	#pragma omp parallel for num_threads(8)
	for(unsigned int r = 0; r < nr_r; r++){
		double start = getTime();

		double meantime = 999999999999;
		if(sumtimeOK != 0){meantime = sumtimeSum/double(sumtimeOK+1.0);}
        refinement->maxtime = std::min(0.5,3*meantime);

		Eigen::Affine3d randomrot = Eigen::Affine3d::Identity();
		randomrot =	Eigen::AngleAxisd(rxs[r], Eigen::Vector3d::UnitX()) *
					Eigen::AngleAxisd(rys[r], Eigen::Vector3d::UnitY()) *
					Eigen::AngleAxisd(rzs[r], Eigen::Vector3d::UnitZ());

		Eigen::Affine3d current_guess = Ymean*randomrot*Xmean.inverse();//*Ymean;

		FusionResults fr = refinement->getTransform(current_guess.matrix());
		//fr_X[r] = refinement->getTransform(current_guess.matrix());

		#pragma omp critical
		{
			fr_X[r] = fr;

			double stoptime = getTime();
			sumtime += stoptime-start;
			if(!fr_X[r].timeout){
				sumtimeSum += stoptime-start;
				sumtimeOK++;
			}
	//		bool exists2 = false;
	//		for(unsigned int ax = 0; ax < fr_X.size(); ax++){
	//			if(issame(fr, fr_X[ax],stepxsmall)){exists2 = true; break;}
	//		}
	//		if(!exists2){fr_X.push_back(fr);}
		}
	}



	FusionResults fr = FusionResults();
    refinement->allow_regularization = false;

	int tpbef = refinement->target_points;


	for(int tp = 500; tp <= 1000; tp *= 2){
		refinement->target_points = tp;

		unsigned int nr_X = fr_X.size();
		#pragma omp parallel for num_threads(8)
		for(unsigned int ax = 0; ax < nr_X; ax++){
			fr_X[ax] = refinement->getTransform(fr_X[ax].guess);
		}

		for(unsigned int ax = 0; ax < fr_X.size(); ax++){
			for(unsigned int bx = ax+1; bx < fr_X.size(); bx++){
				if(issame(fr_X[bx], fr_X[ax],stepxsmall)){
					fr_X[bx] = fr_X.back();
					fr_X.pop_back();
					bx--;
				}
			}
		}
	}

//	for(int tp = 500; tp <= 1000; tp *= 2){
//		refinement->target_points = tp;

//		unsigned int nr_X = fr_X.size();
//		//#pragma omp parallel for num_threads(8)
//		for(unsigned int ax = 0; ax < nr_X; ax++){
//			fr_X[ax] = refinement->getTransform(fr_X[ax].guess);
//		}

//		for(unsigned int ax = 0; ax < fr_X.size(); ax++){
//			for(unsigned int bx = ax+1; bx < fr_X.size(); bx++){
//				if(issame(fr_X[bx], fr_X[ax],stepxsmall)){
//					fr_X[bx] = fr_X.back();
//					fr_X.pop_back();
//					bx--;
//				}
//			}
//		}
//	}

	refinement->visualizationLvl = visualizationLvl;
	std::sort (fr_X.begin(), fr_X.end(), compareFusionResults);
	for(unsigned int ax = 0; ax < fr_X.size() && ax < 500; ax++){
		fr.candidates.push_back(fr_X[ax].guess);
		fr.counts.push_back(1);
		fr.scores.push_back(fr_X[ax].score);
	}


	refinement->visualizationLvl = 2;
	refinement->target_points = 30000;
	for(unsigned int ax = 0; ax < fr_X.size() && ax < 30; ax++){
		printf("%i -> %f\n",ax,fr_X[ax].score);
		refinement->getTransform(fr_X[ax].guess);
	}

	refinement->visualizationLvl = 0;
	refinement->target_points = tpbef;
	return fr;
}

}
