#ifndef reglibsuperpoint_H
#define reglibsuperpoint_H

#include <Eigen/Dense>


namespace reglib
{
class superpoint{
public:
	Eigen::Vector3f point;
	Eigen::Vector3f normal;
	Eigen::VectorXf feature;
	double point_information;
	double normal_information;
	double feature_information;
	int last_update_frame_id;

	superpoint(Eigen::Vector3f p = Eigen::Vector3f(0,0,0), Eigen::Vector3f n = Eigen::Vector3f(0,0,0), Eigen::VectorXf f = Eigen::VectorXf(3), double pi = 1, double fi = 1, int id = 0);
	~superpoint();
	void merge(superpoint p, double weight = 1);
};
}

#endif
