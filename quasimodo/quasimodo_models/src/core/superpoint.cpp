#include "../../include/core/superpoint.h"

namespace reglib
{

superpoint::superpoint(Eigen::Vector3f p, Eigen::Vector3f n, Eigen::VectorXf f, double pi, double fi, int id, bool is_boundry_){
    x	= p(0);
    y	= p(1);
    z	= p(2);

    nx	= n(0);
    ny	= n(1);
    nz	= n(2);

    r	= f(0);
    g	= f(1);
    b	= f(2);

    point_information = pi;
    normal_information = pi;
    colour_information = fi;
    last_update_frame_id = id;

    is_boundry = is_boundry_;
}

superpoint::~superpoint(){}

void superpoint::merge(superpoint p, double weight){

    double newpweight = weight*p.point_information	+ point_information;
    x	= weight*p.point_information*p.x		+ point_information*x;
    y	= weight*p.point_information*p.y		+ point_information*y;
    z	= weight*p.point_information*p.z		+ point_information*z;
    x	/= newpweight;
    y	/= newpweight;
    z	/= newpweight;
    point_information = newpweight;

    double newnweight = weight*p.point_information	+ point_information;

    nx	= weight*p.normal_information*p.nx		+ normal_information*nx;
    ny	= weight*p.normal_information*p.ny		+ normal_information*ny;
    nz	= weight*p.normal_information*p.nz		+ normal_information*nz;
    double nnorm = sqrt(nx*nx+ny*ny+nz*nz);
    if(nnorm != 0){
        nx /= nnorm;
        ny /= nnorm;
        nz /= nnorm;
    }
    normal_information = newnweight;


    double newcweight = weight*p.colour_information	+ colour_information;
    r	= weight*p.colour_information*p.r		+ colour_information*r;
    g	= weight*p.colour_information*p.g		+ colour_information*g;
    b	= weight*p.colour_information*p.b		+ colour_information*b;
    r	/= newcweight;
    g	/= newcweight;
    b	/= newcweight;
    colour_information = newcweight;

    last_update_frame_id = std::max(p.last_update_frame_id,last_update_frame_id);

    if(!p.is_boundry){is_boundry = false;}
}

void superpoint::print(){
    printf("point: %5.5f %5.5f %5.5f Colour: %5.5f %5.5f %5.5f Infos: %5.5f %5.5f %5.5f\n",x,y,z,r,g,b,point_information,normal_information,colour_information);
}

}

