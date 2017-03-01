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

using namespace std;
using namespace Eigen;
//using namespace reglib;

default_random_engine generator;

double getRandVal(){
    return 2.0*(double(rand()%10000)/10000.0) - 1.0;
}

Matrix3Xd conc(Matrix3Xd A, Matrix3Xd B){
    Eigen::Matrix3Xd C	= Eigen::Matrix3Xd::Zero(3,	A.cols()+B.cols());
    C << A, B;
    return C;
}



Matrix3Xd getPoints(int cols){
    Eigen::Matrix3Xd points	= Eigen::Matrix3Xd::Zero(3,cols);
    for(int i = 0; i < 3; i++){
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

Matrix3Xd getMeasurements(Matrix3Xd gt, double noise){
    int cols = gt.cols();

    normal_distribution<double> distribution(0,noise);
    Eigen::Matrix3Xd points	= Eigen::Matrix3Xd::Zero(3,cols);
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < cols; j++){
            points(i,j) = gt(i,j) + distribution(generator);
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
double get_reconst_rms(Matrix3Xd gt, Matrix3Xd points, int nr_points){
    double rms = 0;
    for(int i = 0; i < nr_points; i++){
        float dx	= gt(0,i) - points(0,i);
        float dy	= gt(1,i) - points(1,i);
        float dz	= gt(2,i) - points(2,i);
        rms += dx*dx+dy*dy+dz*dz;

        //printf("%i rms: %f\n",i,rms);
    }
    rms/= double(nr_points);
    rms = sqrt(rms)/sqrt(3);
    //printf("final rms: %f\n",rms);
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

Matrix3Xd align (Matrix3Xd A, Matrix3Xd B, VectorXd W){
    for(int it = 0; it < 1; it++){
        pcl::TransformationFromCorrespondences tfc;
        for(int i = 0; i < A.cols(); i++){
            tfc.add(Eigen::Vector3f(A(0,i),A(1,i),A(2,i)),Eigen::Vector3f(B(0,i),B(1,i),B(2,i)),W(i));
        }
        Matrix4d T = tfc.getTransformation().matrix().cast<double>();
        //std::cout << T << std::endl << std::endl;
        A = transform_points(A, T);
    }
    return A;
}


Matrix3Xd refine (double & match_acc, Matrix3Xd A, Matrix3Xd B, reglib::DistanceWeightFunction2 * func, int nr_correct_matches = 1000, double maxtime = 2.0){
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
    for(int funcupdate=0; funcupdate < 100; ++funcupdate) {
        tot_func++;
        for(int outer=0; outer < 100; ++outer) {
            tot_outer++;
            Matrix3Xd residuals = A-B;
            funcc->computeModel(residuals);
            Matrix3Xd A_old2 = A;

            if(reglib::getTime() - startTime  > maxtime){ break; }
            for(int inner=0; inner < 100; ++inner) {
                if(reglib::getTime() - startTime  > maxtime){ break; }
                tot_inner++;
                //VectorXd W = funcc->getProbs(A-B);
                VectorXd W = funcc->getWeights(invstdvec, A-B);


                if(funcc->debugg_print){

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
                stop = 0.00001;// * func->getNoise();
                Matrix3Xd A_old = A;
                //A = align (A,B,W);


                pcl::TransformationFromCorrespondences tfc;
                for(int i = 0; i < A.cols(); i++){
                    tfc.add(Eigen::Vector3f(A(0,i),A(1,i),A(2,i)),Eigen::Vector3f(B(0,i),B(1,i),B(2,i)),W(i));
                }
                Matrix4d T = tfc.getTransformation().matrix().cast<double>();
                A = transform_points(A, T);

                T_tot = T * T_tot;

                double stop2 = (A-A_old).colwise().norm().mean();
                A_old = A;

                //printf("%i %i %i -> %f (noise: %f)\n",funcupdate,outer,inner,stop2,func->getNoise());
                if(stop2 < stop) {break;};
            }

            double stop3 = (A-A_old2).colwise().norm().mean();
            A_old2 = A;
            if(stop3 < stop) {break;}
        }
        double noise_before = funcc->getNoise();
        funcc->update();
        double noise_after = funcc->getNoise();

        if(fabs(1.0 - noise_after/noise_before) < 0.001){break;}
    }

    //printf("func: %i outer: %i inner: %i\n",tot_func,tot_outer,tot_inner);


    VectorXd W = funcc->getProbs(A-B);

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
        }else{
            failsratio++;
        }
    }

    mean /= double(res_vec.size()-failsratio);
    failsratio /= double(res_vec.size());
}

void test(std::string testname, int nr_frames, int nr_correct_matches, int nr_outlier_matches, double noise,
          vector<Matrix4d> translation_transformations, vector<Matrix4d> angle_transformations, vector< reglib::DistanceWeightFunction2 * > funcs){

    printf("%% %s -> nr_frames: %i, nr_correct_matches: %i, nr_outlier_matches: %i, noise: %f\n",testname.c_str(), nr_frames, nr_correct_matches, nr_outlier_matches, noise );

//    VectorXd W (nr_correct_matches+nr_outlier_matches);
//    for(int i = 0; i < nr_correct_matches+nr_outlier_matches; i++){
//        W(i)= i < nr_correct_matches;
//    }
//    vector< double    > optimal;
//    vector< Matrix3Xd > backgrounds;
//    vector< Matrix3Xd > sampled;
//    for(unsigned int i = 0; i < nr_frames; i++){

//        //backgrounds.push_back(inlier_backgrounds.back());
//        //sampled.push_back(inlier_sampled.back);

////        Matrix3Xd bgoutliers = getPoints(nr_outlier_matches);
////        Matrix3Xd sampleoutliers = getOutlierPoints(bgoutliers);

////        backgrounds.back()  = conc(backgrounds.back(),  bgoutliers);
////        sampled.back()      = conc(sampled.back(),      sampleoutliers);

//        //backgrounds.back()  = conc(backgrounds.back(),  getPointsOutlier(nr_outlier_matches));
//        //sampled.back()      = conc(sampled.back(),      getPointsOutlier(nr_outlier_matches));


//        //double opt = get_reconst_rms(backgrounds.back(), align (sampled.back(), backgrounds.back(),W),nr_correct_matches);

//        optimal.push_back(opt);
//    }


    vector< double    > optimal;
    vector< Matrix3Xd > inlier_backgrounds;
    vector< Matrix3Xd > inlier_sampled;

    vector< Matrix3Xd > outlier_backgrounds;
    vector< Matrix3Xd > outlier_sampled;

    vector< Matrix3Xd > backgrounds;

    for(unsigned int i = 0; i < nr_frames; i++){
        inlier_backgrounds.push_back(getPoints(nr_correct_matches));
        inlier_sampled.push_back(getMeasurements(inlier_backgrounds.back(),noise));
        outlier_backgrounds.push_back(getPoints(nr_outlier_matches));
        outlier_sampled.push_back(getOutlierPoints(outlier_backgrounds.back()));
        VectorXd W (nr_correct_matches);
        for(int i = 0; i < nr_correct_matches; i++){W(i)= 1;}
        double opt = get_reconst_rms(inlier_backgrounds.back(), align (inlier_sampled.back(), inlier_backgrounds.back(),W),nr_correct_matches);
        backgrounds.push_back(conc(inlier_backgrounds.back(),outlier_backgrounds.back()));
        optimal.push_back(opt);
    }

    double optimal_mean_error;
    double optimal_failsratio;
    analyzeResults(optimal_mean_error,optimal_failsratio,optimal,noise);
    printf("optimal_mean_error: %7.7f\n",optimal_mean_error);

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

            #pragma omp parallel for num_threads(11)
            for(unsigned int i = 0; i < nr_sampled; i++){
                Matrix3Xd measurements_full_trans = conc(transform_points(inlier_sampled[i], translation_transformations[k]),outlier_sampled[i]);
                double match_acc;
                double startTime = reglib::getTime();
                Matrix3Xd result = refine (match_acc,measurements_full_trans, backgrounds[i],func, nr_correct_matches);
                double endTime = reglib::getTime();
                double opt = get_reconst_rms(backgrounds[i], result, nr_correct_matches)-optimal[i];
                res_vec[i] = opt;
                #pragma omp critical
                {
                    total_match_acc += match_acc;
                    total_time += endTime-startTime;
                    done++;
//                    //printf("\r %% %s Translation %4i / %4i -> %4i / %4i -> %4i / %4i",func->name.c_str(), f+1,funcs.size(),k+1,translation_transformations.size(),done,nr_sampled);
//                  printf("%% Translation %4i / %4i -> %4i / %4i -> %4i / %4i\n",f+1,funcs.size(),k+1,translation_transformations.size(),done,nr_sampled);
                }

            }

            total_match_acc /= double(nr_sampled);
            translation_meantime_vec.push_back(total_time/ double(nr_sampled));
            translation_matchacc_vec.push_back(total_match_acc);
            translation_res_vec.push_back(res_vec);
        }

//#pragma omp critical
//        {
            printf("\r");
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

            #pragma omp parallel for num_threads(11)
            for(unsigned int i = 0; i < nr_sampled; i++){
                Matrix3Xd measurements_full_trans = conc(transform_points(inlier_sampled[i], angle_transformations[k]),outlier_sampled[i]);
                double match_acc;
                double startTime = reglib::getTime();
                Matrix3Xd result = refine (match_acc,measurements_full_trans, backgrounds[i],func, nr_correct_matches);
                double endTime = reglib::getTime();
                double opt = get_reconst_rms(backgrounds[i], result, nr_correct_matches)-optimal[i];
                res_vec[i] = opt;
                #pragma omp critical
                {
                    total_match_acc += match_acc;
                    total_time += endTime-startTime;
                    done++;
//                    //printf("\r %% %s Translation %4i / %4i -> %4i / %4i -> %4i / %4i",func->name.c_str(), f+1,funcs.size(),k+1,angle_transformations.size(),done,nr_sampled);
//                  printf("%% Angle %4i / %4i -> %4i / %4i -> %4i / %4i\n",f+1,funcs.size(),k+1,angle_transformations.size(),done,nr_sampled);
                }

            }

            total_match_acc /= double(nr_sampled);
            angle_meantime_vec.push_back(total_time/ double(nr_sampled));
            angle_matchacc_vec.push_back(total_match_acc);
            angle_res_vec.push_back(res_vec);
        }
#pragma omp critical
        {
            printf("\r");
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



Eigen::Matrix4d getTransform(double step){
    double angle = double(step)*M_PI;
    double randX = getRandVal();
    double randY = getRandVal();
    double randZ = getRandVal();

    double de = sqrt(randX*randX + randY*randY + randZ*randZ);

    return getMat(angle, angle, angle, sqrt(3.0)*step*randX/de, sqrt(3.0)*step*randY/de, sqrt(3.0)*step*randZ/de);
}

void showTest2d(reglib::DistanceWeightFunction2 * func){

}

int main(int argc, char **argv){
    reglib::MassRegistrationPPR3 * massreg = new reglib::MassRegistrationPPR3();

    int nr_frames = 100;
    int nr_correct_matches = 1000;
    int nr_outlier_matches = 10000;
    double noise = 0.01;

    vector<Matrix4d> translation_transformations;
    //for(double i = 0; i <= 0.2; i+=0.01){ translation_transformations.push_back(getTransform(i)); }
    for(int i = 0; i <= 100; i+=1){ translation_transformations.push_back(getMat(0,0,0,double(i*0.01),0,0)); }
    vector<Matrix4d> angle_transformations;
    for(int i = 0; i <= 100; i+=1){ angle_transformations.push_back(getMat(0.01*double(i)*M_PI,0,0,0,0,0)); }

    char buf [1024];

    vector< reglib::DistanceWeightFunction2 * > funcs;

    for(unsigned int setup = 0; setup <= 0b1; setup++){
        for(double sr = 0.0; sr <= 0.01; sr += 0.01){
            //for(double cp = 0.01; cp <= 0.05; cp += 0.005){
                reglib::GeneralizedGaussianDistribution * gd = new reglib::GeneralizedGaussianDistribution(true,(setup & 0b1) > 0,true);
                gd->nr_refineiters = 6;
                gd->costpen = 5;
                gd->ratio_costpen = 0;
                gd->debugg_print = false;
                reglib::DistanceWeightFunction2PPR3 * sfunc = new reglib::DistanceWeightFunction2PPR3(gd);
                sfunc->startreg                             = sr;
                sfunc->blur                                 = 0.02;//0.03;
                sfunc->data_per_bin                         = 80;
                sfunc->debugg_print                         = false;
                sfunc->threshold                            = false;//(setup & 0b1) > 0;
                sfunc->useIRLSreweight                      = (setup & 0b1) > 0;
                if(sfunc->useIRLSreweight){
                    sprintf(buf,"ppr_%i_ggd",int(100.0*sr));
                }else{
                    sprintf(buf,"ppr_%i_gausian",int(100.0*sr));
                }
                sfunc->name = string(buf);
                funcs.push_back(sfunc);
            //}
        }
    }


    //    for(unsigned int setup = 0; setup <= 0b1; setup++){
    //        for(double sr = 0.0; sr <= 0.1; sr += 0.1){
    //            reglib::GeneralizedGaussianDistribution * gd = new reglib::GeneralizedGaussianDistribution(true,true);
    //            gd->nr_refineiters = 6;
    //            gd->costpen = 3;
    //            gd->ratio_costpen = 0;
    //            gd->debugg_print = false;
    //            reglib::DistanceWeightFunction2PPR3 * sfunc = new reglib::DistanceWeightFunction2PPR3(gd);
    //            sfunc->startreg                             = sr;
    //            sfunc->blur                                 = 0.03;
    //            sfunc->data_per_bin                         = 20;
    //            sfunc->debugg_print                         = false;
    //            sfunc->threshold                            = false;//(setup & 0b1) > 0;
    //            sfunc->useIRLSreweight                      = (setup & 0b1) > 0;
    //            if(sfunc->useIRLSreweight){
    //                sprintf(buf,"ppr_%i_ggd",int(100.0*sr));
    //            }else{
    //                sprintf(buf,"ppr_%i_gausian",int(100.0*sr));
    //            }
    //            sfunc->name = string(buf);
    //            funcs.push_back(sfunc);
    //        }
    //    }

    //    test("verylow", 11, 1000, 100,     0.01,translation_transformations,angle_transformations,funcs);
    //    test("low",     11, 1000, 1000,    0.01,translation_transformations,angle_transformations,funcs);
    //    test("medium",  11, 1000, 3000,    0.01,translation_transformations,angle_transformations,funcs);
    //    test("high",    11, 1000, 10000,   0.01,translation_transformations,angle_transformations,funcs);
    //    exit(0);


//    for(double mul = 3; mul <= 4; mul += 0.5)
//    {
//        reglib::DistanceWeightFunction2 * tfunc = new reglib::DistanceWeightFunction2();
//        tfunc->f                     = reglib::THRESHOLD;
//        tfunc->p                     = mul*noise;
//        sprintf(buf,"threshold_%i",int(100.0*mul));
//        tfunc->name = string(buf);
//        funcs.push_back(tfunc);
//    }
/*
    for(unsigned int setup = 0; setup <= 0b0; setup++){
        for(double sr = 0.0; sr <= 0.1; sr += 0.1){
            //for(double cp = 0.01; cp <= 0.05; cp += 0.005){
                reglib::GeneralizedGaussianDistribution * gd = new reglib::GeneralizedGaussianDistribution(true,(setup & 0b1) > 0,false);
                gd->nr_refineiters = 6;
                gd->costpen = 5;
                gd->ratio_costpen = 0;
                gd->debugg_print = false;
                reglib::DistanceWeightFunction2PPR3 * sfunc = new reglib::DistanceWeightFunction2PPR3(gd);
                sfunc->startreg                             = sr;
                sfunc->blur                                 = 0.02;//0.03;
                sfunc->data_per_bin                         = 80;
                sfunc->debugg_print                         = false;
                sfunc->threshold                            = false;//(setup & 0b1) > 0;
                sfunc->useIRLSreweight                      = (setup & 0b1) > 0;
                if(sfunc->useIRLSreweight){
                    sprintf(buf,"ppr_%i_ggd",int(100.0*sr));
                }else{
                    sprintf(buf,"ppr_%i_gausian",int(100.0*sr));
                }
                sfunc->name = string(buf);
                funcs.push_back(sfunc);
            //}
        }
    }
    */

//    for(double mul = 3; mul <= 4; mul += 0.5){
//        reglib::DistanceWeightFunction2 * tfunc = new reglib::DistanceWeightFunction2();
//        tfunc->f                     = reglib::THRESHOLD;
//        tfunc->p                     = mul*noise;
//        sprintf(buf,"threshold_%i",int(100.0*mul));
//        tfunc->name = string(buf);
//        funcs.push_back(tfunc);
//    }

//    {
//        reglib::DistanceWeightFunction2 * tfunc = new reglib::DistanceWeightFunction2();
//        tfunc->f                        = reglib::PNORM;
//        tfunc->p                        = 0.1;
//        tfunc->name                     = "pnorm01";
//        funcs.push_back(tfunc);
//    }

//    {
//        reglib::DistanceWeightFunction2 * tfunc = new reglib::DistanceWeightFunction2();
//        tfunc->f                     = reglib::PNORM;
//        tfunc->p                     = 1.0;
//        //tfunc->debugg_print                         = true;
//        tfunc->name                     = "pnorm10";
//        funcs.push_back(tfunc);
//    }


//    test("high", 100, 1000, 10000, 0.01,translation_transformations,angle_transformations,funcs);
//    exit(0);


    test("verylow", 100, 1000, 100, 0.01,translation_transformations,angle_transformations,funcs);
    test("low", 100, 1000, 1000, 0.01,translation_transformations,angle_transformations,funcs);
    test("medium", 100, 1000, 3000, 0.01,translation_transformations,angle_transformations,funcs);
    test("high", 100, 1000, 10000, 0.01,translation_transformations,angle_transformations,funcs);
    exit(0);

    VectorXd W (nr_correct_matches+nr_outlier_matches);
    for(int i = 0; i < nr_correct_matches+nr_outlier_matches; i++){
        W(i)= i < nr_correct_matches;
    }
    vector< double    > optimal;
    vector< Matrix3Xd > backgrounds;
    vector< Matrix3Xd > sampled;
    for(unsigned int i = 0; i < nr_frames; i++){
        backgrounds.push_back(getPoints(nr_correct_matches));
        sampled.push_back(getMeasurements(backgrounds.back(),noise));


        backgrounds.back()  = conc(backgrounds.back(),  getPoints(nr_outlier_matches));
        sampled.back()      = conc(sampled.back(),      getPoints(nr_outlier_matches));


        double opt = get_reconst_rms(backgrounds.back(), align (sampled.back(), backgrounds.back(),W),nr_correct_matches);
        optimal.push_back(opt);
    }

    double optimal_mean_error;
    double optimal_failsratio;
    analyzeResults(optimal_mean_error,optimal_failsratio,optimal,noise);
    printf("optimal_mean_error: %7.7f\n",optimal_mean_error);

    unsigned int nr_funcs = funcs.size();

    //    vector< vector< vector< double > > > all_results;
    //    vector< vector< vector< double > > > all_times;
    //    all_results.resize(funcs.size());
    //    all_times.resize(funcs.size());
    //    for(unsigned int j = 0; j < funcs.size(); j++){
    //        all_results[j].resize(transformations.size());
    //        all_times[j].resize(transformations.size());
    //        for(unsigned int k = 0; k < transformations.size(); k++){
    //            all_results[j][k].resize(size);
    //            all_times[j][k].resize(size);
    //            for(int i = 0; i < size; i++){
    //                all_results[j][k][i] = 0;
    //                all_times[j][k][i] = 0;
    //            }
    //        }
    //    }

    for(unsigned int f = 0; f < nr_funcs; f++){
        reglib::DistanceWeightFunction2 * func = funcs[f];

        vector< vector< double > > translation_res_vec;
        for(unsigned int k = 0; k < translation_transformations.size(); k++){
            vector< double > res_vec;
            double startTime = reglib::getTime();
            double total_match_acc = 0;
            for(unsigned int i = 0; i < sampled.size(); i++){
                Matrix3Xd measurements_full_trans = transform_points(sampled[i], translation_transformations[k]);
                double match_acc;
                Matrix3Xd result = refine (match_acc,measurements_full_trans, backgrounds[i],func, nr_correct_matches);
                double opt = get_reconst_rms(backgrounds[i], result, nr_correct_matches)-optimal[i];
                res_vec.push_back(opt);
                total_match_acc += match_acc;
            }
            double endTime = reglib::getTime();
            total_match_acc /= double(sampled.size());

            translation_res_vec.push_back(res_vec);
        }

        vector< vector< double > > angle_res_vec;
        for(unsigned int k = 0; k < angle_transformations.size(); k++){
            vector< double > res_vec;
            double startTime = reglib::getTime();
            double total_match_acc = 0;
            for(unsigned int i = 0; i < sampled.size(); i++){
                Matrix3Xd measurements_full_trans = transform_points(sampled[i], angle_transformations[k]);
                double match_acc;
                Matrix3Xd result = refine (match_acc,measurements_full_trans, backgrounds[i],func, nr_correct_matches);
                double opt = get_reconst_rms(backgrounds[i], result, nr_correct_matches)-optimal[i];
                res_vec.push_back(opt);
                total_match_acc += match_acc;
            }
            double endTime = reglib::getTime();
            total_match_acc /= double(sampled.size());

            angle_res_vec.push_back(res_vec);
        }

        printf("%s_translation_mean = [",func->name.c_str());
        for(unsigned int k = 0; k < translation_transformations.size(); k++){
            double mean_error = 0;
            double failsratio = 0;
            analyzeResults(mean_error,failsratio,translation_res_vec[k],noise);
            printf("%4.4f ",log(mean_error));
        }
        printf("];\n");

        printf("%s_translation_fail = [",func->name.c_str());
        for(unsigned int k = 0; k < translation_transformations.size(); k++){
            double mean_error = 0;
            double failsratio = 0;
            analyzeResults(mean_error,failsratio,translation_res_vec[k],noise);
            printf("%4.4f ",failsratio);
        }

        printf("];\n");

        printf("%s_angle_mean = [",func->name.c_str());
        for(unsigned int k = 0; k < angle_transformations.size(); k++){
            double mean_error = 0;
            double failsratio = 0;
            analyzeResults(mean_error,failsratio,angle_res_vec[k],noise);
            printf("%4.4f ",log(mean_error));
        }
        printf("];\n");

        printf("%s_angle_fail = [",func->name.c_str());
        for(unsigned int k = 0; k < angle_transformations.size(); k++){
            double mean_error = 0;
            double failsratio = 0;
            analyzeResults(mean_error,failsratio,angle_res_vec[k],noise);
            printf("%4.4f ",failsratio);
        }
        printf("];\n");

    }

    //
    //    funcs.push_back(sfunc);

    //    int size = 100;
    //    int overall_scale = 1.0;


    //    double optimal = 0;
    //    vector< vector< double > > results;
    //    vector< vector< double > > times;
    //    results.resize(funcs.size());
    //    times.resize(funcs.size());
    //    for(unsigned int j = 0; j < funcs.size(); j++){
    //        results[j].resize(transformations.size());
    //        times[j].resize(transformations.size());
    //        for(unsigned int k = 0; k < transformations.size(); k++){
    //            results[j][k] = 0;
    //            times[j][k] = 0;
    //        }1 / 1 -> mean_error: 0.0002009964 failsratio: 0.0000000 meantime: 0.0005957s
    //    }





    //        for(int i = 0; i < size; i++){
    //            //printf("\%\% %i / %i\n",i,size);
    //            Matrix3Xd measurements_tmp	= overall_scale * measurements[i];
    //            Matrix3Xd gt_tmp			= overall_scale * gt[i];
    //            //printf("start L2:\n");
    //            align(L2, measurements_tmp, gt_tmp);
    //            //printf("stop  L2:\n");
    //            double opt_rms = get_reconst_rms(gt_tmp, measurements_tmp, cols)/sqrt(3.0) / overall_scale;
    //            //printf("opt rms: %6.6f\n",opt_rms);
    //            optimal += opt_rms;

    //            Matrix3Xd gt_outliers			= getPoints(nr_outliers);
    //            Matrix3Xd measurements_outliers	= getMeasurementsOffseted(gt_outliers, noise*sample_area);

    //            //Matrix3Xd gt_full             = gt_outliers;//conc(gt[i],gt_outliers);
    //            //Matrix3Xd measurements_full	= measurements_outliers;//conc(measurements[i],measurements_outliers);
    //            Matrix3Xd gt_full               = overall_scale*conc(gt[i],gt_outliers);
    //            Matrix3Xd measurements_full     = conc(measurements[i],measurements_outliers);

    //

    //			//#pragma omp parallel for num_threads(7)
    //			for(int j = 0; j < nr_funcs; j++){
    //				//if(debugg){if(k < 1 || i != 8){continue;}}
    //				//if(i < 2){continue;}
    //				//if(i < 91){continue;}
    //				//if(i > 2){continue;}
    //				Matrix3Xd measurements_full_tmp = overall_scale*measurements_full_trans;

    //				printf("%i %i %i / %i %i %i --> ",i,k,j,size,transformations.size(),funcs.size());
    //				//cout << measurements_full_tmp << endl << endl;
    //				double start = getTime();
    //				//if(i == 93){
    //				//	funcs[j]->debugg_print = true;
    //				//	align(funcs[j], measurements_full_tmp, gt_full,0.00001,true);}
    //				//else{align(funcs[j], measurements_full_tmp, gt_full,0.00001,debugg);}
    //				align(funcs[j], measurements_full_tmp, gt_full,0.00001,debugg);
    //				double stop = getTime()-start;
    //				//cout << measurements_full_tmp << endl << endl;

    //				double rms = get_reconst_rms(gt_full, measurements_full_tmp, cols)/sqrt(3.0) / overall_scale;
    //				results[j][k] += rms-opt_rms;
    //				times[j][k] += stop;//-start;
    //				all_results[j][k][i] = rms-opt_rms;
    //				all_times[j][k][i] = stop;//-start;

    //				printf("rms: %12.12f\n",10000.0*(rms-opt_rms));
    ///*
    //				if(rms-opt_rms > 0.000001){
    //					measurements_full_tmp = overall_scale*measurements_full_trans;
    //					align(funcs[j], measurements_full_tmp, gt_full,0.00001,true);
    //					exit(0);
    //				}
    //*/
    //				//if(stop > 1){exit(0);}

    //				//printf("getNoise: %f\n",funcs[j]->getNoise());
    //				//printf("%f += %f - %f (%f)\n",times[j],stop,start,stop-start);
    //				//printf("%s diff from opt rms: %6.6f\n",funcs[j]->getString().c_str(),rms-opt_rms);
    //			}
    //		}
    //	}

    //	for(unsigned int j = 0; j < funcs.size(); j++){
    //		printf("%s_rms =[",funcs[j]->getString().c_str());
    //		for(unsigned int k = 0; k < transformations.size(); k++){
    //			printf("%4.4f ",10000.0*(results[j][k]));
    //		}printf("];\n");
    //	}

    //	for(unsigned int j = 0; j < funcs.size(); j++){
    //		printf("\%\% %i -> %s\n",j,funcs[j]->getString().c_str());
    //	}
    ///*
    //	printf("\%\% all_rms(funcs,transformations,testcase)\n");
    //	printf("all_rms = zeros(%i,%i,%i);\n",funcs.size(),transformations.size(),size);
    //	for(unsigned int j = 0; j < funcs.size(); j++){
    //		for(unsigned int k = 0; k < transformations.size(); k++){
    //			printf("all_rms(%i,%i,:) = [",j,k);//buf,funcs.size(),transformations.size(),size);
    //			for(int i = 0; i < size; i++){
    //				printf("%3.4f ",10000.0*(all_results[j][k][i]));
    //			}printf("];\n");
    //		}
    //	}
    //*/

    //	for(unsigned int j = 0; j < funcs.size(); j++){
    //		printf("%s_stab =[",funcs[j]->getString().c_str());
    //		for(unsigned int k = 0; k < transformations.size(); k++){
    //			double inl = 0;
    //			for(int i = 0; i < size; i++){
    //				inl += all_results[j][k][i] < 0.01;
    //			}

    //			printf("%5.5f ",inl/double(size));
    //		}printf("];\n");
    //	}


    //	printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
    //	for(unsigned int j = 0; j < funcs.size(); j++){
    //		printf("%s_time =[",funcs[j]->getString().c_str());
    //		for(unsigned int k = 0; k < transformations.size(); k++){
    //			printf("%3.3f ",times[j][k]);
    //		}printf("];\n");
    //	}


    return 0;
}
