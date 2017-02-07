#ifndef reglibsuperpoint_H
#define reglibsuperpoint_H

#include <Eigen/Dense>


namespace reglib
{
class superpoint{
public:
//	Eigen::Vector3f point;
//	Eigen::Vector3f normal;
//	Eigen::VectorXf feature;
	double x;
	double y;
	double z;

	double nx;
	double ny;
	double nz;

	double r;
	double g;
	double b;

	double point_information;
	double normal_information;
	double colour_information;
	//double feature_information;
	int last_update_frame_id;
    bool is_boundry;

    superpoint(Eigen::Vector3f p = Eigen::Vector3f(0,0,0), Eigen::Vector3f n = Eigen::Vector3f(0,0,0), Eigen::VectorXf f = Eigen::VectorXf(3), double pi = 1, double fi = 1, int id = 0, bool is_boundry_ = false);
	~superpoint();
	void merge(superpoint p, double weight = 1);
    void print();
};
}

#endif
