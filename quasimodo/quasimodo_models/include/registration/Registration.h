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

#include "../core/Util.h"
#include "../weightfunctions/DistanceWeightFunction2.h"

namespace reglib
{

	enum MatchType { PointToPoint, PointToPlane };

	class CloudData{
		public:
		Eigen::MatrixXd information;
		Eigen::MatrixXd data;
		Eigen::MatrixXd normals;
	
		CloudData();
		~CloudData();
	};

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

		unsigned int visualizationLvl;
		int target_points;
		int dst_points;
        bool allow_regularization;
        double maxtime;

		std::map<std::string,double> debugg_times;

		Registration();
		~Registration();

		virtual void setSrc(std::vector<superpoint> & src_);
		virtual void setDst(std::vector<superpoint> & dst_);
		void setVisualizationLvl(unsigned int lvl);

		void addTime(std::string key, double time);
		void printDebuggTimes();

		virtual FusionResults getTransform(Eigen::MatrixXd guess);
		virtual void show(Eigen::MatrixXd X, Eigen::MatrixXd Y, bool stop = true);
		virtual void show(Eigen::MatrixXd X, Eigen::MatrixXd Y, Eigen::VectorXd W);

		virtual void show(Eigen::MatrixXd X, Eigen::MatrixXd Xn, Eigen::MatrixXd Y, Eigen::MatrixXd Yn);
	};

}

#include "RegistrationRefinement.h"
#include "RegistrationRandom.h"
#endif // Registration_H
