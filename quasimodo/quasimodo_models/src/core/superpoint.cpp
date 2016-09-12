#include "../../include/core/superpoint.h"

namespace reglib
{

superpoint::superpoint(Eigen::Vector3f p, Eigen::Vector3f n, Eigen::VectorXf f, double pi, double fi, int id){
	point = p;
	normal = n;
	feature = f;
	point_information = pi;
	feature_information = fi;
	last_update_frame_id = id;
}

superpoint::~superpoint(){}

void superpoint::merge(superpoint p, double weight){
	double newpweight = weight*p.point_information		+ point_information;
	double newfweight = weight*p.feature_information	+ feature_information;
	point	= weight*p.point_information*p.point		+ point_information*point;
	normal	= weight*p.point_information*p.normal		+ point_information*normal;
	normal.normalize();
	point /= newpweight;
	point_information = newpweight;
	feature_information = newfweight;
	last_update_frame_id = std::max(p.last_update_frame_id,last_update_frame_id);
}
}

