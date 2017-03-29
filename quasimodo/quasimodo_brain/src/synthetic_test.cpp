#include "ros/ros.h"
#include "std_msgs/String.h"
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>

// PCL specific includes
#include <pcl/ros/conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

//#include "modelupdater/ModelUpdater.h"
#include "core/RGBDFrame.h"

#include <string.h>

#include "Util/Util.h"

#include "core/DescriptorExtractor.h"


#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <sys/time.h>
#include "modelupdater/ModelUpdater.h"
#include "core/RGBDFrame.h"
#include "Util/Util.h"
#include <random>
#include <omp.h>
#include <pcl/common/transformation_from_correspondences.h>

#include "ros/ros.h"
#include "std_msgs/String.h"
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>

// PCL specific includes
#include <pcl/ros/conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

//#include "modelupdater/ModelUpdater.h"
#include "core/RGBDFrame.h"

#include <string.h>

#include "Util/Util.h"

#include "core/DescriptorExtractor.h"

#define chosen_threads 8

using namespace std;
using namespace Eigen;
//using namespace reglib;

default_random_engine generator;

class TransformationFromCorrespondences2
{
   public:
      //-----CONSTRUCTOR&DESTRUCTOR-----
      /** Constructor - dimension gives the size of the vectors to work with. */
      TransformationFromCorrespondences2 () :
        no_of_samples_ (0), accumulated_weight_ (0),
        mean1_ (Eigen::Vector3d::Identity ()),
        mean2_ (Eigen::Vector3d::Identity ()),
        covariance_ (Eigen::Matrix<double, 3, 3>::Identity ())
      { reset (); }


      ~TransformationFromCorrespondences2 () { };

      inline void reset ();

      inline double getAccumulatedWeight () const { return accumulated_weight_;}

      inline unsigned int getNoOfSamples () { return no_of_samples_;}

      inline void add (const Eigen::Vector3d& point, const Eigen::Vector3d& corresponding_point, double weight=1.0);

      inline Eigen::Affine3d getTransformation ();

   protected:
      unsigned int no_of_samples_;
      double accumulated_weight_;
      Eigen::Vector3d mean1_, mean2_;
      Eigen::Matrix<double, 3, 3> covariance_;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
inline void
TransformationFromCorrespondences2::reset ()
{
  no_of_samples_ = 0;
  accumulated_weight_ = 0.0;
  mean1_.fill(0);
  mean2_.fill(0);
  covariance_.fill(0);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
inline void
TransformationFromCorrespondences2::add (const Eigen::Vector3d& point, const Eigen::Vector3d& corresponding_point,
                                             double weight)
{
  if (weight==0.0)
    return;

  ++no_of_samples_;
  accumulated_weight_ += weight;
  double alpha = weight/accumulated_weight_;

  Eigen::Vector3d diff1 = point - mean1_, diff2 = corresponding_point - mean2_;
  covariance_ = (1.0f-alpha)*(covariance_ + alpha * (diff2 * diff1.transpose()));

  mean1_ += alpha*(diff1);
  mean2_ += alpha*(diff2);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
inline Eigen::Affine3d
TransformationFromCorrespondences2::getTransformation ()
{
  //Eigen::JacobiSVD<Eigen::Matrix<float, 3, 3> > svd (covariance_, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::JacobiSVD<Eigen::Matrix<double, 3, 3> > svd (covariance_, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Eigen::Matrix<double, 3, 3>& u = svd.matrixU(),
                                   & v = svd.matrixV();
  Eigen::Matrix<double, 3, 3> s;
  s.setIdentity();
  if (u.determinant()*v.determinant() < 0.0f)
    s(2,2) = -1.0f;

  Eigen::Matrix<double, 3, 3> r = u * s * v.transpose();
  Eigen::Vector3d t = mean2_ - r*mean1_;

  Eigen::Affine3d ret;
  ret(0,0)=r(0,0); ret(0,1)=r(0,1); ret(0,2)=r(0,2); ret(0,3)=t(0);
  ret(1,0)=r(1,0); ret(1,1)=r(1,1); ret(1,2)=r(1,2); ret(1,3)=t(1);
  ret(2,0)=r(2,0); ret(2,1)=r(2,1); ret(2,2)=r(2,2); ret(2,3)=t(2);
  ret(3,0)=0.0f;   ret(3,1)=0.0f;   ret(3,2)=0.0f;   ret(3,3)=1.0f;

  return (ret);
}

unsigned long getMaxRnd(){
    unsigned long rm = RAND_MAX;
    return rm*rm+rm;
}

unsigned long getRnd(){
    unsigned long rm = RAND_MAX;
    unsigned long r0 = rand();
    unsigned long r1 = rand();
    return rm*r0+r1;
}

double getR01(){
    return double(getRnd())/double(getMaxRnd());
}

double getRandVal(){
    return 2.0*getR01() - 1.0;
}

MatrixXd conc(MatrixXd A, MatrixXd B){
	Eigen::MatrixXd C	= Eigen::MatrixXd::Zero(A.rows(),	A.cols()+B.cols());
	C << A, B;
	return C;
}

double sampleGGD(double sigma, double power){
    while(true){
        double rv = 40.0*getRandVal();
        double prob = exp(-0.5*pow(fabs(rv),power));
        if(getR01() < prob){
            return rv*sigma;
        }
    }
}


Matrix3Xd getPoints(int cols, int rows = 3){
	Eigen::Matrix3Xd points	= Eigen::Matrix3Xd::Zero(rows,cols);
	for(int i = 0; i < rows; i++){
		for(int j = 0; j < cols; j++){
			points(i,j) = (rand()%10000)*0.0001;
			//points(i,j) = getRandVal();
		}
	}
	return points;
}


Matrix3Xd getOutlierPoints(Matrix3Xd src){
	Eigen::Matrix3Xd points	= src;
	int cols = src.cols();
	for(int j = 0; j < cols; j++){
		double x = getRandVal();
		double y = getRandVal();
		double z = getRandVal();
//        double norm = sqrt(x*x+y*y+z*z);

//        double len = 1.0*getRandVal();
//        x *= len/norm;
//        y *= len/norm;
//        z *= len/norm;
		points(0,j) += x;
		points(1,j) += y;
		points(2,j) += z;
	}
	return points;
}

Matrix3Xd getMeasurements(Matrix3Xd gt, double noise, double power = 2){
	int cols = gt.cols();

	normal_distribution<double> distribution(0,noise);
	Eigen::Matrix3Xd points	= Eigen::Matrix3Xd::Zero(3,cols);
	for(int i = 0; i < 3; i++){
		for(int j = 0; j < cols; j++){
            points(i,j) = gt(i,j) + sampleGGD(noise,power);//distribution(generator);
		}
	}
	return points;
}

Matrix3Xd transform_points(Matrix3Xd points, Matrix4d transform){
	int cols = points.cols();

	float m00 = transform(0,0); float m01 = transform(0,1); float m02 = transform(0,2); float m03 = transform(0,3);
	float m10 = transform(1,0); float m11 = transform(1,1); float m12 = transform(1,2); float m13 = transform(1,3);
	float m20 = transform(2,0); float m21 = transform(2,1); float m22 = transform(2,2); float m23 = transform(2,3);

	Eigen::Matrix3Xd points2	= Eigen::Matrix3Xd::Zero(3,cols);
	for(int i = 0; i < cols; i++){
		float x	= points(0,i);
		float y	= points(1,i);
		float z	= points(2,i);
		points2(0,i)	= m00*x + m01*y + m02*z + m03;
		points2(1,i)	= m10*x + m11*y + m12*z + m13;
		points2(2,i)	= m20*x + m21*y + m22*z + m23;
	}
	return points2;
}
double get_reconst_rms(Matrix3Xd gt, Matrix3Xd points, int nr_points, double power = 2){
	double rms = 0;
	for(int i = 0; i < nr_points; i++){
		float dx	= gt(0,i) - points(0,i);
		float dy	= gt(1,i) - points(1,i);
		float dz	= gt(2,i) - points(2,i);
        double de = sqrt(dx*dx+dy*dy+dz*dz);
        double rs = pow(de,power);
        rms += rs;

//        if(i % 100 == 0){
//            printf("%i de: %15.15f rs: %15.15f\n",i,de,rs);
//        }
	}
	rms/= double(nr_points);
//    printf("mean %15.15f\n",rms);
    rms = pow(rms,1.0/power)/sqrt(3);
//    printf("final rms: %15.15f\n",rms);

	return rms;
}

Matrix4d getMat( double angle1, double angle2, double angle3, double t1, double t2,	double t3){
	Matrix3d rotmat;
	rotmat = AngleAxisd(angle1, Vector3d::UnitZ()) * AngleAxisd(angle2, Vector3d::UnitY()) * AngleAxisd(angle3, Vector3d::UnitZ());
	Matrix4d transformation = Matrix4d::Identity();
	transformation.block(0,0,3,3) = rotmat;
	transformation(0,3) = t1; transformation(1,3) = t2; transformation(2,3) = t3;
	return transformation;
}

Matrix3Xd align (Matrix3Xd A, Matrix3Xd B, VectorXd W, double power = 2){
    for(int it = 0; it < 100; it++){
        TransformationFromCorrespondences2 tfc;
        for(int i = 0; i < A.cols(); i++){
            double de = (A.col(i)-B.col(i)).norm();
            double we = pow(std::max(0.00001,de),power-2);//pow(std::de,power-2);//1.0;//1.0/pow(de,2-power);
            tfc.add(Eigen::Vector3d(A(0,i),A(1,i),A(2,i)),Eigen::Vector3d(B(0,i),B(1,i),B(2,i)),we*W(i));
        }
        Matrix4d T = tfc.getTransformation().matrix();//.cast<double>();
		A = transform_points(A, T);
	}
	return A;
}

std::vector<double> W_prev;

Matrix3Xd refine (double & match_acc, Matrix3Xd A, Matrix3Xd B, reglib::DistanceWeightFunction2 * func, int index = -1, int nr_correct_matches = 1000, double maxtime = 20.0){
	double stop = 0;

	double startTime = reglib::getTime();

	int cols = A.cols();

	std::vector<double > invstdvec;
	invstdvec.resize(cols);
	for(int i = 0; i < cols; i++){invstdvec[i] = 1;}

	Matrix4d T_tot = Matrix4d::Identity();

	int tot_inner = 0;
	int tot_outer = 0;
	int tot_func = 0;

	//reglib::DistanceWeightFunction2 * funcc2 = fun//c;//c = func->clone();
	reglib::DistanceWeightFunction2 * funcc = func->clone();//c = func->clone();
	funcc->reset();


    //funcc->setDebugg(index == 12);


    bool timebreak = false;
    bool tuned = false;
    int maxouter = 3;
	if(funcc->regularization == 0){
        maxouter = 100;
	}
    //printf("updating = [");
    VectorXd W;
    //for(int funcupdate=0; funcupdate < 1; ++funcupdate) {
    for(int funcupdate=0; funcupdate < 100; ++funcupdate) {
        tot_func++;
        for(int outer=0; outer < maxouter || (tuned && outer < 15); ++outer) {
        //for(int outer=0; outer < 1; ++outer) {
			tot_outer++;
			Matrix3Xd residuals = A-B;
            funcc->computeModel(residuals);
			Matrix3Xd A_old2 = A;

            if(reglib::getTime() - startTime  > maxtime && !funcc->debugg_print){ printf("%i timebreak\n",index); timebreak = true; break; }
            for(int inner=0; inner < 5; ++inner) {
            //for(int inner=0; inner < 1; ++inner) {
                if(reglib::getTime() - startTime  > maxtime && !funcc->debugg_print){ printf("%i timebreak\n",index); timebreak = true; break; }

                //if(index == 1){printf("%i %i %i\n",funcupdate,outer,inner);}
				tot_inner++;
				//VectorXd W = funcc->getProbs(A-B);
                W = funcc->getWeights(invstdvec, A-B);

//                std::vector<double> W_current;
//                W_current.resize(cols);
//                for(int i = 0; i < cols; i++){W_current[i] = W(i);}
//                if(W_prev.size() > 0){
//                    double sum_diff = 0;
//                    double max_diff = 0;
//                    for(int i = 0; i < cols; i++){
//                        sum_diff += fabs(W_current[i] - W_prev[i]);
//                        max_diff  = std::max(max_diff,fabs(W_current[i] - W_prev[i]));
//                    }
//                    printf("max_diff: %7.7f sum_diff: %7.7f mean_diff: %7.7f\n",max_diff,sum_diff,sum_diff/double(cols));
//                    for(int i = 0; i < cols; i+= cols){
//                        printf("%i -> %7.7f and %7.7f -> diff %7.7f\n",i,W_current[i],W_prev[i],W_current[i] - W_prev[i]);
//                    }
//                }else{
//                    W_prev = W_current;
//                }

                if(false && inner == 0 && funcc->debugg_print){
                    printf("Noise: %f regularization: %f\n",funcc->getNoise(),funcc->regularization);
                    double tp = 0;
                    double fp = 0;
                    double tn = 0;
                    double fn = 0;
                    for(int i = 0; i < cols; i++){
                        if(i < nr_correct_matches){
                            tp += W(i);
                            fn += 1.0-W(i);
                        }else{
                            fp += W(i);
                            tn += 1.0-W(i);
                        }
                    }
                    printf("true possitive(good): %5.5f false positive(bad) %5.5f true negative(good): %5.5f false negative(bad): %5.5f\n",tp,fp,tn,fn);
                    cv::Mat wimg;
                    double maxw = W(0);
                    for(int i = 0; i < cols; i++){maxw = std::max(double(W(i)),maxw);}
                    wimg.create(cols/100,100,CV_32FC1);
                    float * wdata = (float *)wimg.data;
                    for(int i = 0; i < cols; i++){
                        wdata[i] = W(i)/maxw;
                    }
                    cv::namedWindow( "wimg", cv::WINDOW_AUTOSIZE );			cv::imshow( "wimg", wimg );
                    cv::waitKey(0);
                }
				//VectorXd W = func->getWeights(invstdvec,A-B);
                if(tuned){
                    stop = 0.000001;
                }else{
                    stop = 0.01*func->getNoise();
                }
				Matrix3Xd A_old = A;
				//A = align (A,B,W);


                TransformationFromCorrespondences2 tfc;
                for(int i = 0; i < A.cols(); i++){
                    tfc.add(Eigen::Vector3d(A(0,i),A(1,i),A(2,i)),Eigen::Vector3d(B(0,i),B(1,i),B(2,i)),W(i));
                }
                Matrix4d T = tfc.getTransformation().matrix();//.cast<double>();

				A = transform_points(A, T);

				T_tot = T * T_tot;

				double stop2 = (A-A_old).colwise().norm().mean();
				A_old = A;

				//printf("%i %i %i -> %f (noise: %f)\n",funcupdate,outer,inner,stop2,func->getNoise());
				if(stop2 < stop) {break;};
			}

			double stop3 = (A-A_old2).colwise().norm().mean();
			//printf("stop3: %f stop: %f\n",stop3,stop);
			A_old2 = A;
			if(stop3 < stop) {break;}
		}
		double noise_before = funcc->getNoise();
		funcc->update();
		double noise_after = funcc->getNoise();


        if(fabs(1.0 - noise_after/noise_before) < 0.01){
            //printf("noise_before: %10.10f noise_after: %10.10f reg: %10.10f\n",noise_before,noise_after,(noise_before-noise_after)/(1-0.8));
            funcc->setTune();
            if(tuned){break;}
            tuned = true;
        }
	}

    //printf("];\n");
    //if(index == 1){printf("func: %i outer: %i inner: %i\n",tot_func,tot_outer,tot_inner);}

    if(timebreak){printf("%i func: %i outer: %i inner: %i\n",index,tot_func,tot_outer,tot_inner);}

//if(tot_func == 100){exit(0);}
    //VectorXd W = funcc->getProbs(A-B);

	match_acc = 0;
	for(int i = 0; i < cols; i++){
		match_acc += fabs(W(i)-double(i < nr_correct_matches));
	}
	match_acc /= double(cols);

	//    //printf("match_acc: %f\n",match_acc);
	//    func->print();
	//    func->getWeights(invstdvec,A-B,true);
	delete funcc;
	return A;
}

void analyzeResults(double & mean, double & failsratio, vector< double > res_vec, float threshold = 0.02){
	mean = 0;
	failsratio = 0;
	for(unsigned int i = 0; i < res_vec.size(); i++){
		if(res_vec[i] < threshold){
			mean += res_vec[i];
			//printf("%i -> %f = ok\n",i,res_vec[i]);
		}else{
			failsratio++;
			//printf("%i -> %f = fail\n",i,res_vec[i]);
		}
	}

	mean /= double(res_vec.size()-failsratio);
	failsratio /= double(res_vec.size());
}

void test(std::string testname, int nr_frames, int nr_correct_matches, int nr_outlier_matches, double noise, double power,
		  vector<Matrix4d> translation_transformations, vector<Matrix4d> angle_transformations, vector< reglib::DistanceWeightFunction2 * > funcs){

	printf("%% %s -> nr_frames: %i, nr_correct_matches: %i, nr_outlier_matches: %i, noise: %f\n",testname.c_str(), nr_frames, nr_correct_matches, nr_outlier_matches, noise );

	vector< double    > optimal;
	vector< Matrix3Xd > inlier_backgrounds;
	vector< Matrix3Xd > inlier_sampled;

	vector< Matrix3Xd > outlier_backgrounds;
	vector< Matrix3Xd > outlier_sampled;

	vector< Matrix3Xd > backgrounds;

	for(unsigned int i = 0; i < nr_frames; i++){
		inlier_backgrounds.push_back(getPoints(nr_correct_matches));
        inlier_sampled.push_back(getMeasurements(inlier_backgrounds.back(),noise,power));
		outlier_backgrounds.push_back(getPoints(nr_outlier_matches));
		outlier_sampled.push_back(getOutlierPoints(outlier_backgrounds.back()));
		VectorXd W (nr_correct_matches);
		for(int i = 0; i < nr_correct_matches; i++){W(i)= 1;}
        double opt = get_reconst_rms(inlier_backgrounds.back(), align (inlier_sampled.back(), inlier_backgrounds.back(),W,power),nr_correct_matches,power);
		backgrounds.push_back(conc(inlier_backgrounds.back(),outlier_backgrounds.back()));
		optimal.push_back(opt);
	}

	double optimal_mean_error;
	double optimal_failsratio;
	analyzeResults(optimal_mean_error,optimal_failsratio,optimal,noise);
    printf("%% optimal_mean_error: %7.7f\n",optimal_mean_error);

	unsigned int nr_funcs = funcs.size();
//#pragma omp parallel for num_threads(11)
	for(unsigned int f = 0; f < nr_funcs; f++){
		reglib::DistanceWeightFunction2 * func = funcs[f];

		unsigned int nr_translation_transformations = translation_transformations.size();
		vector< vector< double > > translation_res_vec;
		vector< double > translation_meantime_vec;
		vector< double > translation_matchacc_vec;
		for(unsigned int k = 0; k < translation_transformations.size(); k++){
			double total_match_acc = 0;
			double total_time = 0;
			int done = 0;
			unsigned int nr_sampled = backgrounds.size();
			double startTime1 = reglib::getTime();

			vector< double > res_vec;
			res_vec.resize(nr_sampled);

            #pragma omp parallel for num_threads(chosen_threads)
			for(unsigned int i = 0; i < nr_sampled; i++){
				Matrix3Xd measurements_full_trans = conc(transform_points(inlier_sampled[i], translation_transformations[k]),outlier_sampled[i]);
				double match_acc;
				double startTime = reglib::getTime();
                Matrix3Xd result = refine (match_acc,measurements_full_trans, backgrounds[i],func,i, nr_correct_matches);
				double endTime = reglib::getTime();
                double opt = get_reconst_rms(backgrounds[i], result, nr_correct_matches,power)-optimal[i];
				if(opt < 0){opt = 0.00000000000000000000000000001;}
				res_vec[i] = opt;

				//exit(0);
				#pragma omp critical
				{
					//printf("%i -> opt: %15.15f %5.5f\n",i,opt,log(opt));
					total_match_acc += match_acc;
					total_time += endTime-startTime;
					done++;
					//printf("\r %% %s Translation %4i / %4i -> %4i / %4i -> %4i / %4i",func->name.c_str(), f+1,funcs.size(),k+1,translation_transformations.size(),done,nr_sampled);
                  //printf("%% Translation %4i / %4i -> %4i / %4i -> %4i / %4i -> %5.5f\n",f+1,funcs.size(),k+1,translation_transformations.size(),done,nr_sampled,log(opt));
				}

			}

			total_match_acc /= double(nr_sampled);
			translation_meantime_vec.push_back(total_time/ double(nr_sampled));
			translation_matchacc_vec.push_back(total_match_acc);
			translation_res_vec.push_back(res_vec);


			double mean_error = 0;
			double failsratio = 0;
			analyzeResults(mean_error,failsratio,translation_res_vec[k],noise);

////			printf("%% IMPORTANT!!!! ---> %4i/%4i -> %4f / %4f\n",k+1,translation_transformations.size(),log(mean_error),failsratio);

//            printf("%s_%s_translation_mean = [",testname.c_str(),func->name.c_str());
//            for(unsigned int k = 0; k < translation_res_vec.size(); k++){
//                double mean_error = 0;
//                double failsratio = 0;
//                analyzeResults(mean_error,failsratio,translation_res_vec[k],noise);
//                printf("%4.4f ",log(mean_error));
//            }
//            printf("];\n");

//            printf("%s_%s_translation_fail = [",testname.c_str(),func->name.c_str());
//            for(unsigned int k = 0; k < translation_res_vec.size(); k++){
//                double mean_error = 0;
//                double failsratio = 0;
//                analyzeResults(mean_error,failsratio,translation_res_vec[k],noise);
//                printf("%4.4f ",failsratio);
//            }
//            printf("];\n");

//            printf("%s_%s_translation_time = [",testname.c_str(),func->name.c_str());
//            for(unsigned int k = 0; k < translation_meantime_vec.size(); k++){printf("%4.4f ",translation_meantime_vec[k]);}
//            printf("];\n");

//            printf("%s_%s_translation_match = [",testname.c_str(),func->name.c_str());
//            for(unsigned int k = 0; k < translation_matchacc_vec.size(); k++){printf("%4.4f ",translation_matchacc_vec[k]);}
//            printf("];\n");
		}

//#pragma omp critical
//        {
            //printf("\r");
			if(translation_transformations.size() > 0){
				printf("%s_%s_translation_mean = [",testname.c_str(),func->name.c_str());
				for(unsigned int k = 0; k < translation_transformations.size(); k++){
					double mean_error = 0;
					double failsratio = 0;
					analyzeResults(mean_error,failsratio,translation_res_vec[k],noise);
					printf("%4.4f ",log(mean_error));
				}
				printf("];\n");

				printf("%s_%s_translation_fail = [",testname.c_str(),func->name.c_str());
				for(unsigned int k = 0; k < translation_transformations.size(); k++){
					double mean_error = 0;
					double failsratio = 0;
					analyzeResults(mean_error,failsratio,translation_res_vec[k],noise);
					printf("%4.4f ",failsratio);
				}
				printf("];\n");

				printf("%s_%s_translation_time = [",testname.c_str(),func->name.c_str());
				for(unsigned int k = 0; k < translation_transformations.size(); k++){printf("%4.4f ",translation_meantime_vec[k]);}
				printf("];\n");

				printf("%s_%s_translation_match = [",testname.c_str(),func->name.c_str());
				for(unsigned int k = 0; k < translation_transformations.size(); k++){printf("%4.4f ",translation_matchacc_vec[k]);}
				printf("];\n");
			}
//        }

		vector< double > angle_meantime_vec;
		vector< double > angle_matchacc_vec;
		vector< vector< double > > angle_res_vec;
		for(unsigned int k = 0; k < angle_transformations.size(); k++){
			double total_match_acc = 0;
			double total_time = 0;
			unsigned int nr_sampled = backgrounds.size();
			int done = 0;
			vector< double > res_vec;
			res_vec.resize(nr_sampled);

            #pragma omp parallel for num_threads(chosen_threads)
			for(unsigned int i = 0; i < nr_sampled; i++){
				Matrix3Xd measurements_full_trans = conc(transform_points(inlier_sampled[i], angle_transformations[k]),outlier_sampled[i]);
				double match_acc;
				double startTime = reglib::getTime();
                Matrix3Xd result = refine (match_acc,measurements_full_trans, backgrounds[i],func,i, nr_correct_matches);
				double endTime = reglib::getTime();
                double opt = get_reconst_rms(backgrounds[i], result, nr_correct_matches,power)-optimal[i];
                if(opt < 0){opt = 0.00000000000000000000000000001;}
				res_vec[i] = opt;
				#pragma omp critical
				{
					total_match_acc += match_acc;
					total_time += endTime-startTime;
					done++;
//                    //printf("\r %% %s Translation %4i / %4i -> %4i / %4i -> %4i / %4i",func->name.c_str(), f+1,funcs.size(),k+1,angle_transformations.size(),done,nr_sampled);
                    //printf("%% Angle %4i / %4i -> %4i / %4i -> %4i / %4i -> %20.20f -> %5.5f from %i\n",f+1,funcs.size(),k+1,angle_transformations.size(),done,nr_sampled,opt,log(opt),i);
//                    if(true || opt > 0.00001){
//                        //func->debugg_print = true;
//                        for(unsigned int test = 0; test < 100; test++){
//                            Matrix3Xd result2 = refine (match_acc,measurements_full_trans, backgrounds[i],func, nr_correct_matches);
//                            double opt2 = get_reconst_rms(backgrounds[i], result2, nr_correct_matches)-optimal[i];
//                            printf("%% Angle %4i / %4i -> %4i / %4i -> %4i / %4i -> %20.20f -> %5.5f\n",f+1,funcs.size(),k+1,angle_transformations.size(),done,nr_sampled,opt2,log(opt2));
//                        }
//                        exit(0);
//                    }
				}

			}

            //printf("%% Angle %4i / %4i -> %4i / %4i -> %4i / %4i\n",f+1,funcs.size(),k+1,angle_transformations.size(),done,nr_sampled);


			total_match_acc /= double(nr_sampled);
			angle_meantime_vec.push_back(total_time/ double(nr_sampled));
			angle_matchacc_vec.push_back(total_match_acc);
			angle_res_vec.push_back(res_vec);

//	#pragma omp critical
//			{
//				printf("\r");
//				if(angle_transformations.size() > 0){
//					printf("%s_%s_angle_mean = [",testname.c_str(),func->name.c_str());
//					for(unsigned int k = 0; k < angle_res_vec.size(); k++){
//						double mean_error = 0;
//						double failsratio = 0;
//						analyzeResults(mean_error,failsratio,angle_res_vec[k],noise);
//						printf("%4.4f ",log(mean_error));
//					}
//					printf("];\n");

//					printf("%s_%s_angle_fail = [",testname.c_str(),func->name.c_str());
//					for(unsigned int k = 0; k < angle_res_vec.size(); k++){
//						double mean_error = 0;
//						double failsratio = 0;
//						analyzeResults(mean_error,failsratio,angle_res_vec[k],noise);
//						printf("%4.4f ",failsratio);
//					}
//					printf("];\n");

//					printf("%s_%s_angle_time = [",testname.c_str(),func->name.c_str());
//					for(unsigned int k = 0; k < angle_meantime_vec.size(); k++){printf("%4.4f ",angle_meantime_vec[k]);}
//					printf("];\n");

//					printf("%s_%s_angle_match = [",testname.c_str(),func->name.c_str());
//					for(unsigned int k = 0; k < angle_matchacc_vec.size(); k++){printf("%4.4f ",angle_matchacc_vec[k]);}
//					printf("];\n");
//				}
//			}

		}
#pragma omp critical
		{
            //printf("\r");
			if(angle_transformations.size() > 0){
				printf("%s_%s_angle_mean = [",testname.c_str(),func->name.c_str());
				for(unsigned int k = 0; k < angle_transformations.size(); k++){
					double mean_error = 0;
					double failsratio = 0;
					analyzeResults(mean_error,failsratio,angle_res_vec[k],noise);
					printf("%4.4f ",log(mean_error));
				}
				printf("];\n");

				printf("%s_%s_angle_fail = [",testname.c_str(),func->name.c_str());
				for(unsigned int k = 0; k < angle_transformations.size(); k++){
					double mean_error = 0;
					double failsratio = 0;
					analyzeResults(mean_error,failsratio,angle_res_vec[k],noise);
					printf("%4.4f ",failsratio);
				}
				printf("];\n");

				printf("%s_%s_angle_time = [",testname.c_str(),func->name.c_str());
				for(unsigned int k = 0; k < angle_transformations.size(); k++){printf("%4.4f ",angle_meantime_vec[k]);}
				printf("];\n");

				printf("%s_%s_angle_match = [",testname.c_str(),func->name.c_str());
				for(unsigned int k = 0; k < angle_transformations.size(); k++){printf("%4.4f ",angle_matchacc_vec[k]);}
				printf("];\n");
			}
		}
	}

}

Eigen::Matrix4d getTransform(double step){
	double angle = double(step)*M_PI;
	double randX = getRandVal();
	double randY = getRandVal();
	double randZ = getRandVal();

	double de = sqrt(randX*randX + randY*randY + randZ*randZ);

	return getMat(angle, angle, angle, sqrt(3.0)*step*randX/de, sqrt(3.0)*step*randY/de, sqrt(3.0)*step*randZ/de);
}

int main(int argc, char **argv){

    double noise = 0.01;
    double power = 2.0;
	reglib::MassRegistrationPPR3 * massreg = new reglib::MassRegistrationPPR3();

    vector<Matrix4d> translation_transformations;
    for(int i = 0; i <= 100; i+=1){ translation_transformations.push_back(getMat(0,0,0,double(i*0.01),0,0)); }
    //for(int i = 0; i <= 100; i+=25){ translation_transformations.push_back(getMat(0.01*double(i)*M_PI,0.01*double(i)*M_PI*0.1,0.01*double(i)*M_PI*0.1,double(i*0.01),double(i*0.01)*0.5,double(i*0.01)*0.25)); }
	vector<Matrix4d> angle_transformations;
    for(int i = 0; i <= 100; i+=1){ angle_transformations.push_back(getMat(0.01*double(i)*M_PI,0,0,0,0,0)); }
    //for(int i = 0; i <= 100; i+=25){ angle_transformations.push_back(getMat(0.01*double(i)*M_PI,0,0,0,0,0)); }

	char buf [1024];

	vector< reglib::DistanceWeightFunction2 * > funcs;

    if(true){
        reglib::DistanceWeightFunction2JointDist * jdsfunc = new reglib::DistanceWeightFunction2JointDist(0);
        jdsfunc->name = "ppr0";
        jdsfunc->multidim = false;
        jdsfunc->setDebugg(false);
        funcs.push_back(jdsfunc);
    }

    if(true){
        reglib::DistanceWeightFunction2JointDist * jdsfunc = new reglib::DistanceWeightFunction2JointDist(1);
        jdsfunc->name = "ppr1";
        jdsfunc->multidim = false;
        jdsfunc->setDebugg(false);
        funcs.push_back(jdsfunc);
    }

    if(true){
        reglib::DistanceWeightFunction2JointDist * jdsfunc = new reglib::DistanceWeightFunction2JointDist(2);
        jdsfunc->name = "ppr2";
        jdsfunc->multidim = false;
        jdsfunc->setDebugg(false);
        funcs.push_back(jdsfunc);
    }

    if(false){
        double mul = 3.5;
        reglib::DistanceWeightFunction2 * tfunc = new reglib::DistanceWeightFunction2();
        tfunc->f                        = reglib::THRESHOLD;
        tfunc->p                        = mul*noise;
        sprintf(buf,"threshold_%i",int(100.0*mul));
        tfunc->name                     = string(buf);
        tfunc->debugg_print             = false;
        funcs.push_back(tfunc);
    }

    if(false){
        reglib::DistanceWeightFunction2 * tfunc = new reglib::DistanceWeightFunction2();
        tfunc->f                        = reglib::PNORM;
        tfunc->p                        = 0.1;
        tfunc->name                     = "pnorm01";
        tfunc->debugg_print             = false;
        funcs.push_back(tfunc);
    }

    if(false){
        reglib::DistanceWeightFunction2 * tfunc = new reglib::DistanceWeightFunction2();
        tfunc->f                        = reglib::PNORM;
        tfunc->p                        = 1.0;
        tfunc->name                     = "pnorm10";
        tfunc->debugg_print             = false;
        funcs.push_back(tfunc);
    }


    if(false){
        reglib::DistanceWeightFunction2Tdist * tfunc = new reglib::DistanceWeightFunction2Tdist();
        tfunc->name                     = "Tdist";
        tfunc->debugg_print             = false;
        funcs.push_back(tfunc);
    }

    test("verylow_power2", 100, 1000, 100  , noise,power,translation_transformations,angle_transformations,funcs);
    test("low_power2",     100, 1000, 1000 , noise,power,translation_transformations,angle_transformations,funcs);
    test("high_power2",    100, 1000, 10000, noise,power,translation_transformations,angle_transformations,funcs);
	exit(0);
}
