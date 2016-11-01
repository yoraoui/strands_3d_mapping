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
			double bgcr = 1;//0.1*double(rand()%10);
			double bgcg = 1;//0.1*double(rand()%10);
			double bgcb = 1;//0.1*double(rand()%10);

//			printf("%3.3f %3.3f %3.3f %3.3f -> %3.3f %3.3f %3.3f\n", double(w+0)/wgrid, double(h+0)/hgrid, double(w+1)/wgrid,  double(h+1)/hgrid, bgcr,bgcg,bgcb);

			viewer->setBackgroundColor (bgcr,bgcg,bgcb, v1);
			inds.push_back(v1);
//			viewer->spin();
		}
	}

	for( int i = 0; i < clouds.size(); i++ ){
		for( int j = 0; j < clouds[i].size(); j++ ){
			viewer->addPointCloud<pcl::PointXYZRGBNormal> (clouds[i][j], pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(clouds[i][j]), std::to_string(i)+"_"+std::to_string(j),inds[i]);
		}

		viewer->addText (argv[1+i], 10, 10,0,0,0, std::to_string(i), inds[i]);
		printf("%i -> %i\n",i,inds[i]);
		viewer->spin();

	}

	viewer->spin();

	for( int i = 0; i < clouds.size(); i++ ){
		viewer->removeAllPointClouds(inds[i]);

	}

	int v1(0);
	viewer->createViewPort ( 0, 0, 1, 1, v1);

	for( int i = 0; i < clouds.size(); i++ ){
		for( int j = 0; j < clouds[i].size(); j++ ){
			viewer->addPointCloud<pcl::PointXYZRGBNormal> (clouds[i][j], pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(clouds[i][j]), std::to_string(i)+"_"+std::to_string(j),v1);
		}
		//viewer->spinOnce();
		//	}
		//	//void pcl::visualization::PCLVisualizerInteractorStyle::saveScreenshot	(	const std::string & 	file	)
		//	if(save){
		//		printf("saving: %s\n",filename.c_str());
		//		viewer->saveScreenshot(filename);
		viewer->spin();
		viewer->removeAllPointClouds(v1);
	}



//	viewer->addText ("Radius: 0.01", 10, 10, "v1 text", v1);
//	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb (cloud);
//	viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud1", v1);

//	int v2(0);
//	viewer->createViewPort (0.5, 0.0, 1.0, 1.0, v2);
//	viewer->setBackgroundColor (0.3, 0.3, 0.3, v2);
//	viewer->addText ("Radius: 0.1", 10, 10, "v2 text", v2);
//	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> single_color (cloud, 0, 255, 0);
//	viewer->addPointCloud<pcl::PointXYZRGB> (cloud, single_color, "sample cloud2", v2);



	viewer->spin();

//	if(!save){

//	}else{
//
//	}
//	viewer->removeAllPointClouds();

	printf("done...\n");
	return 0;
}
