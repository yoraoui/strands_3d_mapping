#include "registration/RegistrationRandom.h"
#include <iostream>
#include <fstream>
#include <omp.h>
#include <algorithm>

namespace reglib
{

RegistrationRandom::RegistrationRandom(unsigned int steps){
	only_initial_guess		= false;
	visualizationLvl		= 0;
	refinement				= new RegistrationRefinement();

	steprx		= stepry	= steprz	= steps;
	start_rx	= start_ry	= start_rz	= 0;
	stop_rx		= stop_ry	= stop_rz	= 2.0 * M_PI * double(steps)/double(steps+1);

	steptx		= stepty	= steptz	= 1;
	start_tx	= start_ty	= start_tz	= 0;
	stop_tx		= stop_ty	= stop_tz	= 0;

	src_meantype	= 0;
	dst_meantype	= 0;
}

RegistrationRandom::~RegistrationRandom(){
	delete refinement;
}

void RegistrationRandom::setSrc(std::vector<superpoint> & src_){
	src = src_;
	refinement->setSrc(src_);
}
void RegistrationRandom::setDst(std::vector<superpoint> & dst_){
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
	unsigned int s_nr_data = src.size();
	for(unsigned int i = 0; i < s_nr_data/stepxsmall; i++){
		float x		= src[i*stepxsmall].x;//->data(0,i*stepxsmall);
		float y		= src[i*stepxsmall].y;//src->data(1,i*stepxsmall);
		float z		= src[i*stepxsmall].z;//src->data(2,i*stepxsmall);
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

double getPscore(double p, double x, double y,double z, double r, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud){
	double score = 0;
	unsigned int nrp = cloud->points.size();
	for(unsigned int i = 0; i < nrp; i++){
		pcl::PointXYZRGBNormal po = cloud->points[i];
		double dx = po.x-x;
		double dy = po.y-y;
		double dz = po.z-z;
		double dist = sqrt(dx*dx+dy*dy+dz*dz);
		score += pow(fabs(r-dist),p);
	}
	return score;
}


double getPsphere(double p, double & x, double & y,double & z, double & r, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud){
	double h = 0.00001;


	double score = getPscore(p,x,y,z,r,cloud);
	double step = 0.01;
	while(step > 0.0001){
		double next_score = getPscore(p,x,y,z,r+step,cloud);
		while(score > next_score){
			score = next_score;
			r = r+step;
			next_score = getPscore(p,x,y,z,r+step,cloud);
		}
		step *= 0.1;
	}



	for(unsigned int it = 0; it < 1000; it++){
		double score_start = score;
        //printf("it: %10.10i -> %10.10f pos: %5.5f %5.5f %5.5f %5.5f d: ",it,score,x,y,z,r);
		double dx = -(getPscore(p,x+h,y,z,r,cloud) - getPscore(p,x-h,y,z,r,cloud))/(2*h);
		double dy = -(getPscore(p,x,y+h,z,r,cloud) - getPscore(p,x,y-h,z,r,cloud))/(2*h);
		double dz = -(getPscore(p,x,y,z+h,r,cloud) - getPscore(p,x,y,z-h,r,cloud))/(2*h);
		double dr = -(getPscore(p,x,y,z,r+h,cloud) - getPscore(p,x,y,z,r-h,cloud))/(2*h);

        //printf("%5.5f %5.5f %5.5f %5.5f\n",dx,dy,dz,dr);

		double step = 0.001;
		while(step > 0.00000001){
			double next_score = getPscore(p,x+step*dx,y+step*dy,z+step*dz,r+step*dr,cloud);
			while(score > next_score){
				score = next_score;
				x = x+step*dx;
				y = y+step*dy;
				z = z+step*dz;
				r = r+step*dr;
				next_score = getPscore(p,x+step*dx,y+step*dy,z+step*dz,r+step*dr,cloud);
			}
			step *= 0.1;
		}
		double ratio = score/score_start;
		if(ratio > 0.999){break;}
	}
	return score;
}

Eigen::Affine3d RegistrationRandom::getMean(std::vector<superpoint> & data, int type){
	unsigned int nr_data = data.size();

	double mean_x = 0;
	double mean_y = 0;
	double mean_z = 0;

	if(type == 0 || type == 2){
		for(unsigned int i = 0; i < nr_data; i++){
			mean_x += data[i].x;//data->data(0,i);
			mean_y += data[i].y;//data->data(1,i);
			mean_z += data[i].z;//data->data(2,i);
		}
		mean_x /= double(nr_data);
		mean_y /= double(nr_data);
		mean_z /= double(nr_data);

		if(type == 2){
			pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
			cloud->points.clear();
			for(unsigned int i = 0; i < nr_data; i++){pcl::PointXYZRGBNormal p;p.x = data[i].x  ;p.y = data[i].y;p.z = data[i].z;p.b = 0;p.g = 255;p.r = 0;cloud->points.push_back(p);}

			double sphere_r = 0;
			double score = getPsphere(1.0, mean_x,mean_y,mean_z,sphere_r,cloud);
		}
	}else if(type == 1){
		std::vector<double> xvec;
		std::vector<double> yvec;
		std::vector<double> zvec;
		xvec.resize(nr_data);
		yvec.resize(nr_data);
		zvec.resize(nr_data);
		for(unsigned int i = 0; i < nr_data; i++){
			xvec[i] = data[i].x;//data->data(0,i);
			yvec[i] = data[i].y;//data->data(1,i);
			zvec[i] = data[i].z;//data->data(2,i);
		}

		std::sort (xvec.begin(), xvec.end());
		std::sort (yvec.begin(), yvec.end());
		std::sort (zvec.begin(), zvec.end());

		int mid_ind = nr_data/2;

		mean_x = xvec[mid_ind];
		mean_y = xvec[mid_ind];
		mean_z = xvec[mid_ind];
	}

	Eigen::Affine3d mean = Eigen::Affine3d::Identity();
	mean(0,3) = mean_x;
	mean(1,3) = mean_y;
	mean(2,3) = mean_z;
	return mean;
}

FusionResults RegistrationRandom::getTransform(Eigen::MatrixXd guess){
double startTime = getTime();
	std::vector< double > rxs;
	std::vector< double > rys;
	std::vector< double > rzs;

	std::vector< double > txs;
	std::vector< double > tys;
	std::vector< double > tzs;

	for(double rx = 0; rx < steprx; rx++){
		for(double ry = 0; ry < stepry; ry++){
			for(double rz = 0; rz < steprz; rz++){
				for(double tx = 0; tx < steptx; tx++){
					for(double ty = 0; ty < stepty; ty++){
						for(double tz = 0; tz < steptz; tz++){
							rxs.push_back(start_rx + rx*(stop_rx-start_rx));
							rys.push_back(start_ry + ry*(stop_ry-start_ry));
							rzs.push_back(start_rz + rz*(stop_rz-start_rz));
							txs.push_back(start_tx + tx*(stop_tx-start_tx));
							tys.push_back(start_ty + ty*(stop_ty-start_ty));
							tzs.push_back(start_tz + tz*(stop_tz-start_tz));
						}
					}
				}
			}
		}
	}

	unsigned int nr_r = steprx*stepry*steprz*steptx*stepty*steptz;

	refinement->allow_regularization = true;

	unsigned int s_nr_data = src.size();

	Eigen::Affine3d Xmean = getMean(src,src_meantype);
	Eigen::Affine3d Ymean = getMean(dst,dst_meantype);

	double sumtime = 0;
	double sumtimeSum = 0;
	double sumtimeOK = 0;

	refinement->viewer = viewer;
	refinement->visualizationLvl = 0;
	refinement->target_points = 250;
	int stepxsmall = std::max(1,int(s_nr_data)/refinement->target_points);

	std::vector<FusionResults> fr_X;
	fr_X.resize(nr_r);

	refinement->visualizationLvl = visualizationLvl;
	#pragma omp parallel for num_threads(8) schedule(dynamic)
	for(unsigned int r = 0; r < nr_r; r++){
		double start = getTime();

		double meantime = 999999999999;
		if(sumtimeOK != 0){meantime = sumtimeSum/double(sumtimeOK+1.0);}
		refinement->maxtime = 5.0;//std::min(0.5,3*meantime);
		//refinement->maxtime = std::min(0.5,3*meantime);
		if(refinement->visualizationLvl > 0){
			refinement->maxtime = 99999;
		}

		Eigen::Affine3d randomrot = Eigen::Affine3d::Identity();
		randomrot =	Eigen::AngleAxisd(rxs[r], Eigen::Vector3d::UnitX()) *
				Eigen::AngleAxisd(rys[r], Eigen::Vector3d::UnitY()) *
				Eigen::AngleAxisd(rzs[r], Eigen::Vector3d::UnitZ());

		Eigen::Affine3d current_guess = Ymean*randomrot*Xmean.inverse();//*Ymean;

		FusionResults fr = refinement->getTransform(current_guess.matrix());

#pragma omp critical
		{
			fr_X[r] = fr;

			double stoptime = getTime();
			sumtime += stoptime-start;
			if(!fr_X[r].timeout){
				sumtimeSum += stoptime-start;
				sumtimeOK++;
			}
		}
	}

//	printf("%s::%5.5fs\n",__PRETTY_FUNCTION__,getTime()-startTime);
//	refinement->printDebuggTimes();

	//printf("meantime: %f\n",sumtimeSum/double(sumtimeOK+1.0));


	FusionResults fr = FusionResults();
	refinement->allow_regularization = false;

	int tpbef = refinement->target_points;

	int mul = 4;
	for(int tp = 500; tp <= 1000; tp *= 2){
		//printf("------------------------");
		std::sort (fr_X.begin(), fr_X.end(), compareFusionResults);
		refinement->target_points = tp;

		unsigned int nr_X = fr_X.size()/mul;
		#pragma omp parallel for num_threads(8) schedule(dynamic)
		for(unsigned int ax = 0; ax < nr_X; ax++){
			//printf("%5.5i score: %10.10f ",ax,fr_X[ax].score);
			fr_X[ax] = refinement->getTransform(fr_X[ax].guess);
			//printf("-> score: %10.10f\n",fr_X[ax].score);
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
		mul *= 2;
	}

	refinement->visualizationLvl = visualizationLvl;
	std::sort (fr_X.begin(), fr_X.end(), compareFusionResults);
	for(unsigned int ax = 0; ax < fr_X.size() && ax < 500; ax++){
		fr.candidates.push_back(fr_X[ax].guess);
		fr.counts.push_back(1);
		fr.scores.push_back(fr_X[ax].score);
	}

	if(visualizationLvl > 0){
		refinement->allow_regularization = true;
		refinement->visualizationLvl = visualizationLvl;
		refinement->target_points = 1000000;
		refinement->maxtime = 10000;
		for(unsigned int ax = 0; ax < fr_X.size() && ax < 5; ax++){
			printf("%i -> %f\n",ax,fr_X[ax].score);
			std::cout << fr_X[ax].guess << std::endl << std::endl;
			refinement->getTransform(fr_X[ax].guess);
		}
		refinement->visualizationLvl = 0;
	}

	refinement->target_points = tpbef;

	printf("%s::%5.5fs\n",__PRETTY_FUNCTION__,getTime()-startTime);
	return fr;
}

}
