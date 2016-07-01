
#include "ros/ros.h"
#include "std_msgs/String.h"
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>

// PCL specific includes
#include <pcl/ros/conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "modelupdater/ModelUpdater.h"
#include "core/RGBDFrame.h"


#include <iostream>

#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>

#include <string.h>

using namespace std;

boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;

int counterr = 0;
int counterl = 0;
std::string path = "./";

//From crossproduct
void getNormal(float & nx, float & ny, float & nz, float x1, float y1, float z1,float x2, float y2, float z2,float x3, float y3, float z3){
	float u1 = x1-x2;
	float u2 = y1-y2;
	float u3 = z1-z2;
	//	float unorm = sqrt(u1*u1+u2*u2+u3*u3);
	//	u1 /= unorm;
	//	u2 /= unorm;
	//	u3 /= unorm;

	float v1 = x1-x3;
	float v2 = y1-y3;
	float v3 = z1-z3;
	//	float vnorm = sqrt(v1*v1+v2*v2+v3*v3);
	//	v1 /= vnorm;
	//	v2 /= vnorm;
	//	v3 /= vnorm;


	//Corssprod u x v
	nx = u2*v3-u3*v2;
	ny = u3*v1-u1*v3;
	nz = u1*v2-u2*v1;

	float nnorm = sqrt(nx*nx+ny*ny+nz*nz);
	nx /= nnorm;
	ny /= nnorm;
	nz /= nnorm;

	//printf("%f %f %f\n",nx,ny,nz);
	//Flip direction if point + normal further away than point
	if(((x1+nx)*(x1+nx) + (y1+ny)*(y1+ny) + (z1+nz)*(z1+nz)) > (x1*x1 + y1*y1 + z1*z1)){
		nx = -nx;
		ny = -ny;
		nz = -nz;
	}


}

double scoren (Eigen::Vector3d & a, Eigen::Vector3d & b){
	double v = a.dot(b);
	if(std::isnan(v)){return 0;}
	return v*v;
}

unsigned int text_id = 0;
bool gonext = false;
void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event,  void* viewer_void) {
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer> *> (viewer_void);
	if (event.getKeySym () == "n" && event.keyDown ()){
		gonext = true;
		std::cout << "n was pressed => removing all text" << std::endl;
	}
}

void fillNormals(pcl::PointCloud<pcl::PointXYZRGBNormal> & cloud, int normaltype = 0){
	viewer->registerKeyboardCallback (keyboardEventOccurred, (void*)&viewer);
	int stepw = 2*16;

	unsigned int nrp = cloud.points.size();
	for(int i = 0; i < cloud.points.size(); i++){
		pcl::PointXYZRGBNormal & np = cloud.points[i];

		pcl::PointXYZRGBNormal & right = cloud.points[(i+stepw)%nrp];
		pcl::PointXYZRGBNormal & left = cloud.points[(nrp+i-stepw)%nrp];
		pcl::PointXYZRGBNormal up;
		if(i%16 != 0){up = cloud.points[i-1];}

		pcl::PointXYZRGBNormal down;
		if(i%16 != 15){down = cloud.points[i+1];}

		Eigen::Vector3d npv (np.x,np.y,np.z);
		Eigen::Vector3d rv (right.x,right.y,right.z);
		Eigen::Vector3d uv (up.x,up.y,up.z);
		Eigen::Vector3d lv (left.x,left.y,left.z);
		Eigen::Vector3d dv (down.x,down.y,down.z);

		Eigen::Vector3d n1 = (npv-rv).cross(npv-uv);
		Eigen::Vector3d n2 = (npv-rv).cross(npv-dv);
		Eigen::Vector3d n3 = (npv-lv).cross(npv-uv);
		Eigen::Vector3d n4 = (npv-lv).cross(npv-dv);

		n1.normalize();
		n2.normalize();
		n3.normalize();
		n4.normalize();

		double s1 = scoren(n1,n1)+scoren(n1,n2)+scoren(n1,n3)+scoren(n1,n4);
		double s2 = scoren(n2,n1)+scoren(n2,n2)+scoren(n2,n3)+scoren(n2,n4);
		double s3 = scoren(n3,n1)+scoren(n3,n2)+scoren(n3,n3)+scoren(n3,n4);
		double s4 = scoren(n4,n1)+scoren(n4,n2)+scoren(n4,n3)+scoren(n4,n4);

		Eigen::Vector3d n = n1;
		double score = s1;
		if(s2 > score){score = s2; n = n2;}
		if(s3 > score){score = s3; n = n3;}
		if(s4 > score){score = s4; n = n4;}

		if((npv+n).norm() > npv.norm()){
			n = -n;
		}

		np.normal_x = n(0);
		np.normal_y = n(1);
		np.normal_z = n(2);
	}
}

pcl::PointCloud<pcl::PointXYZRGBNormal> getCloudWithNormals(pcl::PointCloud<pcl::PointXYZRGB> & cloud, int normaltype = 0){
	pcl::PointCloud<pcl::PointXYZRGBNormal> normalcloud;
	normalcloud.points.resize(cloud.points.size());
	for(int i = 0; i < cloud.points.size(); i++){
		pcl::PointXYZRGB & op = cloud.points[i];
		pcl::PointXYZRGBNormal & np = normalcloud.points[i];
		np.r = 0;//op.r;
		np.g = 255;//op.g;
		np.b = 0;//op.b;
		np.x = op.x;
		np.y = op.y;
		np.z = op.z;
	}

	//	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	//	*cloud_ptr = normalcloud;
	//	viewer->removeAllPointClouds();
	//	viewer->addPointCloud<pcl::PointXYZRGBNormal> (cloud_ptr, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(cloud_ptr), "cloud");
	//	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");
	//	//viewer->addPointCloudNormals<pcl::PointXYZRGBNormal> (cloud_ptr, 17, 1.5, "normals");
	//	viewer->spin();
	fillNormals(normalcloud,normaltype);
	return normalcloud;
}


pcl::PointCloud<pcl::PointXYZRGBNormal> getSparsifyCloud(pcl::PointCloud<pcl::PointXYZRGBNormal> & cloud, int stepw){
    pcl::PointCloud<pcl::PointXYZRGBNormal> sparsecloud;
    sparsecloud.reserve(cloud.points.size()/stepw);
    for(int i = 0; i < cloud.points.size(); i+= 16*5){//16*(stepw-1)){
        int starti = i;
        for(;i < starti+16; i++){
            sparsecloud.push_back(cloud.points[i]);
        }
    }
    return sparsecloud;
}

pcl::PointCloud<pcl::PointXYZRGBNormal> getCloudFromParts(pcl::PointCloud<pcl::PointXYZRGB> prev, pcl::PointCloud<pcl::PointXYZRGB> curr, Eigen::Matrix4d motion = Eigen::Matrix4d::Identity()){
    pcl::PointCloud<pcl::PointXYZRGB> prev_cloud;
    prev_cloud.resize(prev.points.size()/2);
    for(int i = 0; i < prev.points.size(); i+=2){
        prev_cloud.points[i/2] = prev.points[i];
    }

    pcl::PointCloud<pcl::PointXYZRGB> curr_cloud;
    curr_cloud.resize(curr.points.size()/2);
    for(int i = 0; i < prev.points.size(); i+=2){
        curr_cloud.points[i/2] = curr.points[i];
    }


    pcl::PointCloud<pcl::PointXYZRGB> fullcloud = prev_cloud+curr_cloud;

    Eigen::Matrix4d invmotion = motion.inverse();
    Eigen::Matrix3d rot = invmotion.block(0,0,3,3);
    Vector3d ea = rot.eulerAngles(2, 0, 2);

    for(int i = 0; i < fullcloud.points.size(); i+=16){
        double part = double(i)/double(fullcloud.points.size());


        Eigen::Affine3d mat;
        mat = Eigen::AngleAxisd(part*ea(0), Eigen::Vector3d::UnitZ())*Eigen::AngleAxisd(part*ea(1), Eigen::Vector3d::UnitX())*Eigen::AngleAxisd(part*ea(2), Eigen::Vector3d::UnitZ());

        const double & m00 = mat(0,0); const double & m01 = mat(0,1); const double & m02 = mat(0,2); const double & m03 = part*invmotion(0,3);
        const double & m10 = mat(1,0); const double & m11 = mat(1,1); const double & m12 = mat(1,2); const double & m13 = part*invmotion(1,3);
        const double & m20 = mat(2,0); const double & m21 = mat(2,1); const double & m22 = mat(2,2); const double & m23 = part*invmotion(2,3);

        for(int j = i; j < i+16; j++){//TODO set zeros to zero still
            pcl::PointXYZRGB & p = fullcloud.points[j];
            const double & src_x = p.x;
            const double & src_y = p.y;
            const double & src_z = p.z;
            p.x = m00*src_x + m01*src_y + m02*src_z + m03;
            p.y = m10*src_x + m11*src_y + m12*src_z + m13;
            p.z = m20*src_x + m21*src_y + m22*src_z + m23;
        }
    }

    pcl::PointCloud<pcl::PointXYZRGBNormal> fullcloudnormals = getCloudWithNormals(fullcloud,0);
    return fullcloudnormals;
}

pcl::PointCloud<pcl::PointXYZRGB> prevl;
std::vector< pcl::PointCloud<pcl::PointXYZRGBNormal> > all_cloudsl;
std::vector< Eigen::Matrix4d > all_posesl;
void  cloud_cb_l(const sensor_msgs::PointCloud2ConstPtr& input){
    ROS_INFO("l pointcloud in %i",counterl);

    pcl::PointCloud<pcl::PointXYZRGB> tmpcloud;
    pcl::fromROSMsg (*input, tmpcloud);

    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    cloud.resize(tmpcloud.points.size()/2);
    for(int i = 0; i < tmpcloud.points.size(); i+=2){cloud.points[i/2] = tmpcloud.points[i];}


    char buf[1024];
    if(counterl % 2 == 1){
        pcl::PointCloud<pcl::PointXYZRGBNormal> fullcloudnormals = getCloudFromParts(prevl,cloud, Eigen::Matrix4d::Identity());
        pcl::PointCloud<pcl::PointXYZRGBNormal> sparsecloudnormals = getSparsifyCloud(fullcloudnormals,6);
//        all_clouds.push_back(fullcloudnormals);
//        if(all_poses.size()==0){all_poses.push_back(Eigen::Matrix4d::Identity());
//        }else if(all_poses.size()==1){all_poses.push_back(all_poses.back());
//        }else{
//            all_poses.push_back(all_poses.back()*all_poses[all_poses.size()].inverse()*all_poses.back());
//        }

//        reglib::MassRegistrationPPR * massreg = new reglib::MassRegistrationPPR(0.1);
//        massreg->timeout = 60;
//        massreg->viewer = viewer;
//        massreg->visualizationLvl = 1;

////        massreg->setData(clouds);

//        massreg->stopval = 0.001;
//        massreg->steps = 10;

////      reglib::MassFusionResults mfr = massreg->getTransforms(relativeposes);
//        delete massreg;

//		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
//		*cloud_ptr = fullcloudnormals;
//		viewer->removeAllPointClouds();
//		viewer->addPointCloud<pcl::PointXYZRGBNormal> (cloud_ptr, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(cloud_ptr), "cloud");
//		viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");
//		viewer->addPointCloudNormals<pcl::PointXYZRGBNormal> (cloud_ptr, 17, 0.5, "normals");
//		viewer->spinOnce();

        sprintf(buf,"%s/normalsfullleft_%.10i.pcd",path.c_str(),counterl/2);
        pcl::io::savePCDFileBinary (string(buf), fullcloudnormals);
        sprintf(buf,"%s/normalssparseleft_%.10i.pcd",path.c_str(),counterl/2);
        pcl::io::savePCDFileBinary (string(buf), sparsecloudnormals);
    }

    prevl = cloud;
    counterl++;
}

pcl::PointCloud<pcl::PointXYZRGB> prevr;
std::vector< pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr > all_clouds;
std::vector< Eigen::Matrix4d > all_poses;

void  cloud_cb_r(const sensor_msgs::PointCloud2ConstPtr& input){
    ROS_INFO("r pointcloud in %i",counterr);

    pcl::PointCloud<pcl::PointXYZRGB> tmpcloud;
    pcl::fromROSMsg (*input, tmpcloud);

    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    cloud.resize(tmpcloud.points.size()/2);
    for(int i = 0; i < tmpcloud.points.size(); i+=2){cloud.points[i/2] = tmpcloud.points[i];}


    char buf[1024];
    if(counterr % 2 == 1){
        pcl::PointCloud<pcl::PointXYZRGBNormal> fullcloudnormals = getCloudFromParts(prevr,cloud, Eigen::Matrix4d::Identity());
        pcl::PointCloud<pcl::PointXYZRGBNormal> sparsecloudnormals = getSparsifyCloud(fullcloudnormals,6);

        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        *cloud_ptr = fullcloudnormals;
        all_clouds.push_back(cloud_ptr);
        if(all_poses.size()<=1){all_poses.push_back(Eigen::Matrix4d::Identity());
        }else{
            Eigen::Matrix4d v = all_poses[all_poses.size()-2].inverse()*all_poses.back();
            all_poses.push_back(v*all_poses.back());
        }
        if(all_poses.size()>2){

            std::vector< pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr > ac;
            ac.push_back(all_clouds.front());
            ac.push_back(all_clouds.back());

            std::vector< Eigen::Matrix4d > ap;
            ap.push_back(all_poses.front());
            ap.push_back(all_poses.back());
            reglib::MassRegistrationPPR * massreg = new reglib::MassRegistrationPPR(0.1);
            massreg->timeout = 60;
            massreg->viewer = viewer;
            massreg->visualizationLvl = 0;
            if(all_poses.size()%10 == 0){massreg->visualizationLvl = 1;}


            massreg->setData(ac);

            massreg->stopval = 0.001;
            massreg->steps = 10;

            reglib::MassFusionResults mfr = massreg->getTransforms(ap);
            all_poses.back() = mfr.poses.back();
            delete massreg;
        }



//		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
//		*cloud_ptr = fullcloudnormals;
//		viewer->removeAllPointClouds();
//		viewer->addPointCloud<pcl::PointXYZRGBNormal> (cloud_ptr, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(cloud_ptr), "cloud");
//		viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");
//		viewer->addPointCloudNormals<pcl::PointXYZRGBNormal> (cloud_ptr, 17, 0.5, "normals");
//		viewer->spinOnce();

        sprintf(buf,"%s/normalssparseright_%.10i.pcd",path.c_str(),counterr/2);
        pcl::io::savePCDFileBinary (string(buf), sparsecloudnormals);
        sprintf(buf,"%s/normalsfullright_%.10i.pcd",path.c_str(),counterr/2);
        pcl::io::savePCDFileBinary (string(buf), fullcloudnormals);
    }

    prevr = cloud;
    counterr++;
}

int main(int argc, char **argv){
	viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer ("viewer"));
	viewer->addCoordinateSystem();
	viewer->setBackgroundColor(0.8,0.8,0.8);
	viewer->initCameraParameters ();

	string pathr = "/velodyne_points_right";
	string pathl = "/velodyne_points_left";
	if(argc > 1){	pathr = string(argv[1]);}
	if(argc > 2){	pathl = string(argv[2]);}

	ros::init(argc, argv, "massregVelodyneNode");
	ros::NodeHandle n;
	ros::Subscriber subr = n.subscribe (pathr, 0, cloud_cb_r);
	ros::Subscriber subl = n.subscribe (pathl, 0, cloud_cb_l);
	ros::spin();

	return 0;
}
