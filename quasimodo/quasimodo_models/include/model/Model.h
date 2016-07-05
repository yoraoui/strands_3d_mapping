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
class superpoint{
	public:
	Eigen::Vector3f point;
	Eigen::Vector3f normal;
	Eigen::VectorXf feature;
	double point_information;
	double feature_information;
	int last_update_frame_id;

	superpoint(Eigen::Vector3f p, Eigen::Vector3f n, Eigen::VectorXf f, double pi = 1, double fi = 1, int id = 0){
		point = p;
		normal = n;
		feature = f;
		point_information = pi;
		feature_information = fi;
		last_update_frame_id = id;
	}

	~superpoint(){}

	void merge(superpoint p, double weight = 1){
		double newpweight = weight*p.point_information		+ point_information;
		double newfweight = weight*p.feature_information	+ feature_information;
//printf("before: (%3.3f)(%3.3f) + (%3.3f)(%3.3f) -> (%3.3f)(%3.3f)\n",
//feature_information,feature(0),p.feature_information,p.feature(0),(feature_information*feature(0)+p.feature_information*p.feature(0))/(feature_information+p.feature_information));

		point	= weight*p.point_information*p.point		+ point_information*point;
		normal	= weight*p.point_information*p.normal		+ point_information*normal;
        //feature = weight*p.feature_information*p.feature	+ feature_information*feature;


		normal.normalize();

		point /= newpweight;
		point_information = newpweight;

        //feature /= newfweight;
        //feature_information = newfweight;

		last_update_frame_id = std::max(p.last_update_frame_id,last_update_frame_id);
	}
};

	class Model{
		public:

        double score;
		unsigned long id;

		int last_changed;

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
		
		void merge(Model * model, Eigen::Matrix4d p);

		void showHistory(boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer);

		std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> getHistory();

		void addSuperPoints(vector<superpoint> & spvec, Matrix4d p, RGBDFrame* frame, ModelMask* modelmask);
		void addAllSuperPoints(vector<superpoint> & spvec, Eigen::Matrix4d pose);
		void recomputeModelPoints(Eigen::Matrix4d pose = Eigen::Matrix4d::Identity());
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
	};

}

#endif // reglibModel_H
