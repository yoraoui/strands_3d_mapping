#ifndef reglibModelMask_H
#define reglibModelMask_H

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

namespace reglib
{
	class ModelMask{
		public:
		std::string savepath;
		int id;
		std::string label;
		cv::Mat mask;
		int width;
		int height;
		bool * maskvec;
		std::vector<int> testw;
		std::vector<int> testh;
		int sweepid;


		ModelMask();
		ModelMask(cv::Mat mask_, std::string label_ = "");
		~ModelMask();
		cv::Mat getMask();
		void saveFast(std::string path);
		static ModelMask * loadFast(std::string path);
	};

}

#endif // reglibModelMask_H
