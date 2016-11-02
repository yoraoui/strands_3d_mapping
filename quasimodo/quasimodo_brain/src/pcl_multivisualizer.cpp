#include "ros/ros.h"
#include "std_msgs/String.h"
#include <sensor_msgs/PointCloud2.h>
#include <string.h>

#include <pcl_ros/point_cloud.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_ros/transforms.h>

#include <pcl_ros/point_cloud.h>
#include <cv_bridge/cv_bridge.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_ros/transforms.h>

#include <sys/time.h>
#include <sys/resource.h>

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>




int main(int argc, char** argv){
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
	viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer ("3D Viewer"));
	viewer->setBackgroundColor (1.0, 1.0, 1.0);
	//viewer->addCoordinateSystem (1.0);
	viewer->initCameraParameters ();

	std::vector< std::vector< pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr > > clouds;
	for(int i = 1; i < argc;i++){
		printf("input: %s\n",argv[i]);
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);

		if (pcl::io::loadPCDFile<pcl::PointXYZRGBNormal> (argv[i], *cloud) != -1){
			printf("%i\n",cloud->points.size());
			clouds.push_back(std::vector< pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr >());
			clouds.back().push_back(cloud);
		}
	}
	printf("nr: %i ->",clouds.size());
	double square = sqrt(double(clouds.size())/1.5);
	double wgrid = std::ceil(square*1.5);
	double hgrid = std::ceil(double(clouds.size())/double(wgrid));
	printf("square: %f ->  w: %f h: %f\n",square,wgrid,hgrid);

	std::vector<int> inds;
	for( double w = 0; w+0.5 < wgrid; w++ ){
		for( double h = 0; h+0.5 < hgrid; h++ ){
			int v1(0);
			viewer->createViewPort ( double(w+0)/wgrid, double(h+0)/hgrid, double(w+1)/wgrid,  double(h+1)/hgrid, v1);
			inds.push_back(v1);
		}
	}

	for( int i = 0; i < clouds.size(); i++ ){
		for( int j = 0; j < clouds[i].size(); j++ ){
			viewer->addPointCloud<pcl::PointXYZRGBNormal> (clouds[i][j], pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(clouds[i][j]), std::to_string(i)+"_"+std::to_string(j),inds[i]);
		}

		viewer->addText (argv[1+i], 10, 10,0,0,0, std::to_string(i), inds[i]);
		printf("%i -> %i\n",i,inds[i]);
		viewer->setBackgroundColor (1.0,1.0,1.0, inds[i]+1);

	}

	viewer->spin();


	for( int i = 0; i < clouds.size(); i++ ){
		viewer->removeAllPointClouds(inds[i]);
	}

	int v1(0);
	viewer->createViewPort ( 0, 0, 1, 1, v1);
	viewer->setBackgroundColor (255.0, 255.0, 255.0,v1+1);

	for( int i = 0; i < clouds.size(); i++ ){
		for( int j = 0; j < clouds[i].size(); j++ ){
			viewer->addPointCloud<pcl::PointXYZRGBNormal> (clouds[i][j], pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(clouds[i][j]), std::to_string(i)+"_"+std::to_string(j),v1);
			viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3,  std::to_string(i)+"_"+std::to_string(j),v1);
		}
		viewer->spinOnce();
		std::string path = std::string(argv[1+i]);
		path.pop_back();
		path.pop_back();
		path.pop_back();
		path += "png";
		printf("path: %s\n",path.c_str());

		viewer->saveScreenshot(path);
		viewer->spinOnce();
		//viewer->spin();
		viewer->removeAllPointClouds(v1);
	}
	viewer->spin();


	printf("done...\n");
	return 0;
}
