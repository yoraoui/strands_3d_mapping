#ifndef MassRegistration_H
#define MassRegistration_H

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
//#include <pcl_ros/transforms.h>
//#include <pcl_ros/point_cloud.h>
//#include <pcl/ros/conversions.h>
//#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/pcl_visualizer.h>

//#include "ICP.h"
//#include "nanoflann.hpp"

#include "../weightfunctions/DistanceWeightFunction2.h"
#include "../core/RGBDFrame.h"
#include "../core/Util.h"
#include "../model/Model.h"

namespace reglib
{
/*
	enum MatchType { PointToPoint, PointToPlane };

	class CloudData{
		public:
		Eigen::MatrixXd information;
		Eigen::MatrixXd data;
		Eigen::MatrixXd normals;
	
		CloudData();
		~CloudData();
	};
*/
	class MassFusionResults{
		public:

		std::vector<Eigen::Matrix4d> poses;
		double score;

		MassFusionResults(){score = -1;}
		MassFusionResults(std::vector<Eigen::Matrix4d> poses_, double score_){
			poses = poses_;
			score = score_;
		}
		~MassFusionResults(){}
	};

	class MassRegistration{
		public:
        double timeout;
		bool nomask;
		int maskstep;
		int nomaskstep;

		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;

		unsigned int visualizationLvl;


		std::string savePath;

		std::vector<RGBDFrame *> frames;
		std::vector<ModelMask *> mmasks;

		MassRegistration();
		~MassRegistration();

		bool okVal(double v);
		bool isValidPoint(pcl::PointXYZRGBNormal p);


		virtual void clearData();
		virtual void addData(RGBDFrame* frame, ModelMask * mmask);
		virtual void addModelData(Model * model, bool submodels = true);
		virtual void addData(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud);

		virtual void setData(std::vector<RGBDFrame*> frames_, std::vector<ModelMask *> mmasks);
		virtual void setData(std::vector< pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr > all_clouds);
		void setVisualizationLvl(unsigned int lvl);
		virtual MassFusionResults getTransforms(std::vector<Eigen::Matrix4d> guess);
		virtual void show(Eigen::MatrixXd X, Eigen::MatrixXd Y);
		virtual void show(std::vector<Eigen::MatrixXd> Xv, bool save = false, std::string filename = "", bool stop = true);
		Eigen::MatrixXd getMat(int rows, int cols, double * datas);


		virtual void savePCD(std::vector<Eigen::MatrixXd> Xv, std::string path);

	};

}

#include "MassRegistrationPPR.h"
#include "MassRegistrationPPR2.h"

#endif // MassRegistration_H
