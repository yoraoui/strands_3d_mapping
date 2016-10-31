#ifndef reglibModel_H
#define reglibModel_H

#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h> 
#include <chrono>

#include <Eigen/Dense>

// PCL specific includes
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "../core/RGBDFrame.h"
#include "../registration/Registration.h"

#include "ModelMask.h"

namespace reglib
{
using namespace std;
using namespace Eigen;


	class Model{
		public:

        double score;
		unsigned long id;

		int last_changed;

		std::string savePath;

		std::vector<superpoint> points;

		std::vector< std::vector<cv::KeyPoint> >	all_keypoints;
		std::vector< cv::Mat >						all_descriptors;
		std::vector<Eigen::Matrix4d>				relativeposes;
		std::vector<RGBDFrame*>						frames;
		std::vector<ModelMask*>						modelmasks;

		std::vector<Eigen::Matrix4d>	rep_relativeposes;
		std::vector<RGBDFrame*>			rep_frames;
		std::vector<ModelMask*>			rep_modelmasks;

		double total_scores;
		std::vector<std::vector < float > > scores;

		std::vector<Model *>				submodels;
		std::vector<Eigen::Matrix4d>		submodels_relativeposes;
		std::vector<std::vector < float > > submodels_scores;

		Model();
		Model(RGBDFrame * frame_, cv::Mat mask, Eigen::Matrix4d pose = Eigen::Matrix4d::Identity());
		~Model();

		void fullDelete();
		
		void merge(Model * model, Eigen::Matrix4d p);

		void showHistory(boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer);

		std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> getHistory();

		void addSuperPoints(	vector<superpoint> & spvec, Matrix4d p, RGBDFrame* frame, ModelMask* modelmask, boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = 0);
		void addAllSuperPoints(	vector<superpoint> & spvec, Eigen::Matrix4d pose,								boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = 0);
		void recomputeModelPoints(Eigen::Matrix4d pose = Eigen::Matrix4d::Identity(),							boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = 0);
		void addPointsToModel(RGBDFrame * frame, ModelMask * modelmask, Eigen::Matrix4d p);

		//void addFrameToModel(RGBDFrame * frame, cv::Mat mask, Eigen::Matrix4d p);
		void addFrameToModel(RGBDFrame * frame, ModelMask * modelmask, Eigen::Matrix4d p);
		CloudData * getCD(unsigned int target_points = 2000);
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr getPCLcloud(int step = 5, bool color = true);
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr getPCLnormalcloud(int step = 5, bool color = true);
		void print();
		void save(std::string path = "");
		static Model * load(Camera * cam, std::string path);
		bool testFrame(int ind = 0);

		void getData(std::vector<Eigen::Matrix4d> & po, std::vector<RGBDFrame*> & fr, std::vector<ModelMask*> & mm, Eigen::Matrix4d p = Eigen::Matrix4d::Identity());

		//void getData(std::vector<Eigen::Matrix4d> & po, std::vector<RGBDFrame*> & fr, std::vector<ModelMask*> & mm, Eigen::Matrix4d p = Eigen::Matrix4d::Identity());
	};

}

#endif // reglibModel_H
