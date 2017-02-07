#ifndef Registration_H
#define Registration_H

#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h> 
#include <chrono>

#include <Eigen/Dense>

// PCL specific includes
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transformation_from_correspondences.h>
#include <pcl_ros/transforms.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/ros/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "../weightfunctions/DistanceWeightFunction2.h"
#include "../core/RGBDFrame.h"
#include "../core/Util.h"
#include "../model/Model.h"
#include "../model/ModelMask.h"

namespace reglib
{

	enum MatchType { PointToPoint, PointToPlane };

	class FusionResults{
		public:

		Eigen::MatrixXd guess;
		double score;
		double stop;
		bool timeout;

		std::vector< Eigen::MatrixXd > candidates;
		std::vector< int > counts;
		std::vector< double > scores;

		FusionResults(){score = -1;}
		FusionResults(Eigen::MatrixXd guess_, double score_){
			guess = guess_;
			score = score_;
		}
		~FusionResults(){}
	};

	class Registration{
		public:
		
		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;

		bool only_initial_guess;

        std::vector<superpoint> src;
        std::vector<superpoint> dst;

        std::vector<KeyPoint> src_kp;
        std::vector<KeyPoint> dst_kp;

		unsigned int visualizationLvl;
		int target_points;
		int dst_points;
        bool allow_regularization;
        double maxtime;

        double convergence;
        double regularization;

		std::map<std::string,double> debugg_times;

		Registration();
		~Registration();

        virtual std::string getString();

        virtual void setSrc(std::vector<superpoint> & src_);
        virtual void setDst(std::vector<superpoint> & dst_);

        virtual void setSrc(Model * src, bool recompute = false);
        virtual void setDst(Model * dst, bool recompute = false);

		void setVisualizationLvl(unsigned int lvl);

		void addTime(std::string key, double time);
        virtual void printDebuggTimes();

		virtual FusionResults getTransform(Eigen::MatrixXd guess);

        virtual void show(double * X, unsigned int nr_X, double * Y, unsigned int nr_Y, bool stop = true, std::vector<double> weights = std::vector<double>());
		virtual void show(Eigen::MatrixXd X, Eigen::MatrixXd Y, bool stop = true);
		virtual void show(Eigen::MatrixXd X, Eigen::MatrixXd Y, Eigen::VectorXd W);
		virtual void show(Eigen::MatrixXd X, Eigen::MatrixXd Xn, Eigen::MatrixXd Y, Eigen::MatrixXd Yn);
        virtual void show(std::vector<superpoint> & X, std::vector<superpoint> & Y, Eigen::Matrix4d p, std::vector<double> weights = std::vector<double>(), bool stop = true );
        virtual void show(std::vector<KeyPoint> & X, std::vector<KeyPoint> & Y, Eigen::Matrix4d p, std::vector<int> matches = std::vector<int>(), std::vector<double> weights = std::vector<double>(), bool stop = true);
        virtual void show(double * sp,unsigned int nr_sp,  double * dp, unsigned int nr_dp, Eigen::Matrix4d p, bool stop = true, std::vector<double> weights = std::vector<double>());
    };

}

#include "RegistrationRefinement.h"
#include "RegistrationRefinement2.h"
#include "RegistrationRefinement3.h"
#include "RegistrationRandom.h"
#endif // Registration_H
