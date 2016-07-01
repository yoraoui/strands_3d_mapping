#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <sys/time.h>

#include "../../quasimodo_models/include/modelupdater/ModelUpdater.h"
#include "core/RGBDFrame.h"
#include "core/Util.h"
#include <random>

#include <omp.h>

#include "ceres/ceres.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "ceres/rotation.h"
#include "ceres/iteration_callback.h"

using namespace std;
using namespace Eigen;
using namespace reglib;

using ceres::NumericDiffCostFunction;
using ceres::SizedCostFunction;
using ceres::CENTRAL;
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;
using ceres::Solve;

default_random_engine generator;
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
		}
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
	}
	rms/= double(nr_points);
	rms = sqrt(rms);
	return rms;
}


#include <pcl/common/transformation_from_correspondences.h>

Matrix4d align1(Matrix3Xd gt, Matrix3Xd points){
	pcl::TransformationFromCorrespondences tfc;
	for(int i = 0; i < gt.cols(); i++){
		tfc.add(Eigen::Vector3f(gt(0,i),gt(1,i),gt(2,i)),Eigen::Vector3f(points(0,i),points(1,i),points(2,i)));
	}
	return tfc.getTransformation().matrix().cast<double>().inverse();
}


Eigen::Matrix4d constructTransformationMatrixtest (const double & alpha, const double & beta, const double & gamma, const double & tx,    const double & ty,   const double & tz){
	// Construct the transformation matrix from rotation and translation
	Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Zero ();
	transformation_matrix (0, 0) =  cos (gamma) * cos (beta);
	transformation_matrix (0, 1) = -sin (gamma) * cos (alpha) + cos (gamma) * sin (beta) * sin (alpha);
	transformation_matrix (0, 2) =  sin (gamma) * sin (alpha) + cos (gamma) * sin (beta) * cos (alpha);
	transformation_matrix (1, 0) =  sin (gamma) * cos (beta);
	transformation_matrix (1, 1) =  cos (gamma) * cos (alpha) + sin (gamma) * sin (beta) * sin (alpha);
	transformation_matrix (1, 2) = -cos (gamma) * sin (alpha) + sin (gamma) * sin (beta) * cos (alpha);
	transformation_matrix (2, 0) = -sin (beta);
	transformation_matrix (2, 1) =  cos (beta) * sin (alpha);
	transformation_matrix (2, 2) =  cos (beta) * cos (alpha);

	transformation_matrix (0, 3) = tx;
	transformation_matrix (1, 3) = ty;
	transformation_matrix (2, 3) = tz;
	transformation_matrix (3, 3) = 1;
	return transformation_matrix;
}


Matrix4d align2(Matrix3Xd gt, Matrix3Xd points){
	Affine3d p = Affine3d::Identity();
	for(int l = 0; l < 25; l++){


		typedef Eigen::Matrix<double, 6, 1> Vector6d;
		typedef Eigen::Matrix<double, 6, 6> Matrix6d;

		Matrix6d ATA;
		Vector6d ATb;
		ATA.setZero ();
		ATb.setZero ();

		const double & m00 = p(0,0); const double & m01 = p(0,1); const double & m02 = p(0,2); const double & m03 = p(0,3);
		const double & m10 = p(1,0); const double & m11 = p(1,1); const double & m12 = p(1,2); const double & m13 = p(1,3);
		const double & m20 = p(2,0); const double & m21 = p(2,1); const double & m22 = p(2,2); const double & m23 = p(2,3);

		double scoreX = 0;
		double scoreY = 0;
		double scoreZ = 0;
		double scoreALL = 0;

		double wsum = 0;
		double wsx = 0;
		double wsy = 0;
		double wsz = 0;
		for(int i = 0; i < gt.cols(); i++){
			const double & src_x = gt(0,i);
			const double & src_y = gt(1,i);
			const double & src_z = gt(2,i);

			const double & dx = points(0,i);
			const double & dy = points(1,i);
			const double & dz = points(2,i);

			const double & sx = m00*src_x + m01*src_y + m02*src_z + m03;
			const double & sy = m10*src_x + m11*src_y + m12*src_z + m13;
			const double & sz = m20*src_x + m21*src_y + m22*src_z + m23;

			const double diffX = dx-sx;
			const double diffY = dy-sy;
			const double diffZ = dz-sz;

			scoreX += diffX*diffX;
			scoreY += diffY*diffY;
			scoreZ += diffZ*diffZ;
			scoreALL += diffX*diffX+diffY*diffY+diffZ*diffZ;

			double weight = 1;
			wsum += weight;

			wsx += weight * sx;
			wsy += weight * sy;
			wsz += weight * sz;

			double wsxsx = weight * sx*sx;
			double wsysy = weight * sy*sy;
			double wszsz = weight * sz*sz;

			ATA.coeffRef (0)  += wsysy + wszsz;//a0 * a0;
			ATA.coeffRef (1)  -= weight * sx*sy;//a0 * a1;
			ATA.coeffRef (2)  -= weight * sz*sx;//a0 * a2;


			ATA.coeffRef (7)  += wsxsx + wszsz;//a1 * a1;
			ATA.coeffRef (8)  -= weight * sy*sz;//a1 * a2;

			ATA.coeffRef (14) += wsxsx + wsysy;//a2 * a2;

			ATb.coeffRef (0) += weight * (sy*diffZ -sz*diffY);//a0 * b;
			ATb.coeffRef (1) += weight * (-sx*diffZ + sz*diffX);//a1 * b;
			ATb.coeffRef (2) += weight * (sx*diffY -sy*diffX);//a2 * b;
			ATb.coeffRef (3) += weight * diffX;//a3 * b;
			ATb.coeffRef (4) += weight * diffY;//a4 * b;
			ATb.coeffRef (5) += weight * diffZ;//a5 * b;
		}


		ATA.coeffRef (4)  -= wsz;//a0 * a4;
		ATA.coeffRef (9)  += wsz;//a1 * a3;

		ATA.coeffRef (5)  += wsy;//a0 * a5;
		ATA.coeffRef (15) -= wsy;//a2 * a3;

		ATA.coeffRef (11) -= wsx;//a1 * a5;
		ATA.coeffRef (16) += wsx;//a2 * a4;

		ATA.coeffRef (21) += wsum;//a3 * a3;

		ATA.coeffRef (28) += wsum;//a4 * a4;

		ATA.coeffRef (35) += wsum;//a5 * a5;

		ATA.coeffRef (6) = ATA.coeff (1);
		ATA.coeffRef (12) = ATA.coeff (2);
		ATA.coeffRef (13) = ATA.coeff (8);
		ATA.coeffRef (18) = ATA.coeff (3);
		ATA.coeffRef (19) = ATA.coeff (9);
		ATA.coeffRef (20) = ATA.coeff (15);
		ATA.coeffRef (24) = ATA.coeff (4);
		ATA.coeffRef (25) = ATA.coeff (10);
		ATA.coeffRef (26) = ATA.coeff (16);
		ATA.coeffRef (27) = ATA.coeff (22);
		ATA.coeffRef (30) = ATA.coeff (5);
		ATA.coeffRef (31) = ATA.coeff (11);
		ATA.coeffRef (32) = ATA.coeff (17);
		ATA.coeffRef (33) = ATA.coeff (23);
		ATA.coeffRef (34) = ATA.coeff (29);

		ATA(0,0) += wsum*0.000001;
		ATA(1,1) += wsum*0.000001;
		ATA(2,2) += wsum*0.000001;
		ATA(3,3) += wsum*0.000001;
		ATA(4,4) += wsum*0.000001;
		ATA(5,5) += wsum*0.000001;

		//printf("--------------------\n");
		printf("scoreX:   %5.5f ",scoreX);
		printf("scoreY:   %5.5f ",scoreY);
		printf("scoreZ:   %5.5f ",scoreZ);
		printf("scoreALL: %5.5f\n",scoreALL);

		// Solve A*x = b
		Vector6d x = static_cast<Vector6d> (ATA.inverse () * ATb);

		Eigen::Affine3d transformation = Eigen::Affine3d(constructTransformationMatrixtest(x(0,0),x(1,0),x(2,0),x(3,0),x(4,0),x(5,0)));
		//		std::cout << ATA << std::endl << std::endl;
		//		std::cout << ATb.transpose() << std::endl << std::endl;
		//		std::cout << ATA.inverse () << std::endl << std::endl;
		//		std::cout << x.transpose() << std::endl << std::endl;
		//		std::cout << transformation.matrix() << std::endl << std::endl;
		p = transformation*p;

	}
	return p.matrix().inverse();
}

int main(int argc, char **argv){
	int nr_tests = 1;
	int nr_frames = 100;
	int points_per_match = 100;
	double overlap_prob = 0.1;
	int nr_points = 5000;
	double noise = 0.01;

	for(int nt = 0; nt < nr_tests; nt++){
		Matrix3Xd gt = getPoints(nr_points);
		Matrix3Xd measurements = getMeasurements(gt,noise);

		double angle1 = double(0.1);double angle2 = 0.1;double angle3 = 0.1;
		double t1 = double(0.1);double t2 = 0;	double t3 = 0;

		//printf("\%\% transformation %i -> angle: %f %f %f translation: %f %f %f\n",i,angle1,angle2,angle3,t1,t2,t3);
		Matrix3d rotmat;
		rotmat = AngleAxisd(angle1, Vector3d::UnitZ()) * AngleAxisd(angle2, Vector3d::UnitY()) * AngleAxisd(angle3, Vector3d::UnitZ());
		Matrix4d transformation = Matrix4d::Identity();
		transformation.block(0,0,3,3) = rotmat;
		transformation(0,3) = t1; transformation(1,3) = t2; transformation(2,3) = t3;

		Matrix3Xd measurements_trans = transform_points(measurements, transformation);

		Matrix4d transformation_est1 = align1(gt,measurements_trans);
		Matrix3Xd measurements_trans_est1 = transform_points(measurements_trans, transformation_est1);

		Matrix4d transformation_est2 = align2(gt,measurements_trans);
		Matrix3Xd measurements_trans_est2 = transform_points(measurements_trans, transformation_est2);

		double opt_rms = get_reconst_rms(gt, measurements, nr_points)/sqrt(3.0);
		double rms1 = get_reconst_rms(gt, measurements_trans_est1, nr_points)/sqrt(3.0);
		double rms2 = get_reconst_rms(gt, measurements_trans_est2, nr_points)/sqrt(3.0);
		printf("rms1: %12.12f\n",10000.0*(rms1-opt_rms));
		printf("rms2: %12.12f\n",10000.0*(rms2-opt_rms));


	}
	return 0;
}
