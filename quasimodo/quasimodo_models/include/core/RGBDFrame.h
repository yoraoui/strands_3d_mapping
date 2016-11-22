#ifndef reglibRGBDFrame_H
#define reglibRGBDFrame_H

#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h> 
#include <chrono>

#include <Eigen/Dense>

// PCL specific includes
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/integral_image_normal.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "superpoint.h"
#include "Util.h"
#include "Camera.h"
#include "../weightfunctions/DistanceWeightFunction2.h"

namespace reglib
{
	class RGBDFrame{
		public:
		static int saveId;

		std::string keyval;
		Camera * camera;
		unsigned long id;
		double capturetime;
		Eigen::Matrix4d pose;
		int sweepid;
		std::string soma_id;

		//float * rgbdata;
		cv::Mat det_dilate;
		cv::Mat rgb;
		cv::Mat depth;
		cv::Mat normals;
		cv::Mat depthedges;

		cv::Mat de;
		cv::Mat ce;
		int * labels;
		int nr_labels;

		std::vector< std::vector<double> > connections;
		std::vector< std::vector<double> > intersections;

		RGBDFrame();
		RGBDFrame(Camera * camera_,cv::Mat rgb_, cv::Mat depth_, double capturetime_ = 0, Eigen::Matrix4d pose_ = Eigen::Matrix4d::Identity(), bool compute_normals = true, std::string savePath = "", bool compute_imgedges = true);
		~RGBDFrame();

		void show(bool stop = false);
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr getPCLcloud();
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr getSmallPCLcloud();
        void savePCD(std::string path = "cloud.pcd", Eigen::Matrix4d pose = Eigen::Matrix4d::Identity());

        void save(std::string path = "");
		RGBDFrame * clone();
		static RGBDFrame * load(Camera * cam, std::string path);
		std::vector<ReprojectionResult> getReprojections(std::vector<superpoint> & spvec, Eigen::Matrix4d cp, bool * maskvec,  bool useDet = true);

		std::vector<superpoint> getSuperPoints(Eigen::Matrix4d cp = Eigen::Matrix4d::Identity(), unsigned int step = 1, bool zeroinclude = true);

		std::vector< std::vector<float> > getImageProbs(bool depthonly = false);

        void saveCombinedImages(std::string path);
        void saveCombinedProcessedImages(std::string path);
	};
}

#endif // reglibRGBDFrame_H
