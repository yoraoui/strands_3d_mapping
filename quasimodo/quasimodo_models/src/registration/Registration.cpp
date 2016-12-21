#include "registration/Registration.h"
//#include "ICP.h"
namespace reglib
{

CloudData::CloudData(){}
CloudData::~CloudData(){}

Registration::Registration(){
	only_initial_guess = false;
}
Registration::~Registration(){}

void Registration::setSrc(std::vector<superpoint> & src_){src = src_;}
void Registration::setDst(std::vector<superpoint> & dst_){dst = dst_;}

FusionResults Registration::getTransform(Eigen::MatrixXd guess){
	std::cout << guess << std::endl;
	return FusionResults(guess,0);
}

void Registration::show(Eigen::MatrixXd X, Eigen::MatrixXd Y, bool stop){
	unsigned int s_nr_data = X.cols();
	unsigned int d_nr_data = Y.cols();

	viewer->removeAllPointClouds();
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr scloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr dcloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);

	scloud->points.clear();
	dcloud->points.clear();
	for(unsigned int i = 0; i < s_nr_data; i++){pcl::PointXYZRGBNormal p;p.x = X(0,i);p.y = X(1,i);p.z = X(2,i);p.b = 0;p.g = 255;p.r = 0;scloud->points.push_back(p);}
	for(unsigned int i = 0; i < d_nr_data; i++){pcl::PointXYZRGBNormal p;p.x = Y(0,i);p.y = Y(1,i);p.z = Y(2,i);p.b = 0;p.g = 0;p.r = 255;dcloud->points.push_back(p);}		
	viewer->addPointCloud<pcl::PointXYZRGBNormal> (scloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(scloud), "scloud");
	viewer->addPointCloud<pcl::PointXYZRGBNormal> (dcloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(dcloud), "dcloud");
	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "scloud");
	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "dcloud");

	if(stop){	viewer->spin();}
	else{		viewer->spinOnce();}

	viewer->removeAllPointClouds();
}

void Registration::addTime(std::string key, double time){
	if (debugg_times.count(key) == 0){debugg_times[key] = 0;}
	debugg_times[key] += time;
}

void Registration::printDebuggTimes(){
	printf("====================printDebuggTimes====================\n");
	double sum = 0;
	for (auto it=debugg_times.begin(); it!=debugg_times.end(); ++it){sum += it->second;}
	for (auto it=debugg_times.begin(); it!=debugg_times.end(); ++it){
		printf("%25s :: %15.15f s (%5.5f %%)\n",it->first.c_str(),it->second,it->second/sum);
	}
	printf("========================================================\n");
}

void Registration::show(Eigen::MatrixXd X, Eigen::MatrixXd Xn, Eigen::MatrixXd Y, Eigen::MatrixXd Yn){

	unsigned int s_nr_data = X.cols();
	unsigned int d_nr_data = Y.cols();

	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr scloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr dcloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);

	pcl::PointCloud<pcl::Normal>::Ptr sNcloud (new pcl::PointCloud<pcl::Normal>);
	pcl::PointCloud<pcl::Normal>::Ptr dNcloud (new pcl::PointCloud<pcl::Normal>);

	scloud->points.clear();
	dcloud->points.clear();

	sNcloud->points.clear();
	dNcloud->points.clear();

	for(unsigned int i = 0; i < s_nr_data; i++){
		pcl::PointXYZRGBNormal p;
		p.x = X(0,i);
		p.y = X(1,i);
		p.z = X(2,i);
		p.b = 0;p.g = 255;p.r = 0;
		scloud->points.push_back(p);
	}
	for(unsigned int i = 0; i < d_nr_data; i++){pcl::PointXYZRGBNormal p;p.x = Y(0,i);p.y = Y(1,i);p.z = Y(2,i);p.b = 0;p.g = 0;p.r = 255;dcloud->points.push_back(p);}

	viewer->addPointCloud<pcl::PointXYZRGBNormal> (scloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(scloud), "scloud");
	viewer->addPointCloud<pcl::PointXYZRGBNormal> (dcloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(dcloud), "dcloud");
	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "scloud");
	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "dcloud");
	viewer->spin();
	viewer->removeAllPointClouds();
}

void Registration::setVisualizationLvl(unsigned int lvl){visualizationLvl = lvl;}

void Registration::show(Eigen::MatrixXd X, Eigen::MatrixXd Y, Eigen::VectorXd W){
	show(X,Y);
	double mw = W.maxCoeff();
	W = W/mw;
	//std::cout << W << std::endl;
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr scloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr dcloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	unsigned int s_nr_data = X.cols();
	unsigned int d_nr_data = Y.cols();

	//printf("nr datas: %i %i\n",s_nr_data,d_nr_data);
	scloud->points.clear();
	dcloud->points.clear();
	//for(unsigned int i = 0; i < s_nr_data; i++){pcl::PointXYZRGBNormal p;p.x = X(0,i);p.y = X(1,i);p.z = X(2,i);p.b = 0;p.g = 255;p.r = 0;scloud->points.push_back(p);}
	//for(unsigned int i = 0; i < d_nr_data; i++){pcl::PointXYZRGBNormal p;p.x = Y(0,i);p.y = Y(1,i);p.z = Y(2,i);p.b = 0;p.g = 0;p.r = 255;dcloud->points.push_back(p);}
	for(unsigned int i = 0; i < s_nr_data; i++){
		pcl::PointXYZRGBNormal p;p.x = X(0,i);p.y = X(1,i);p.z = X(2,i);p.b = 255*W(i);	p.g = 255*W(i);	p.r = 255*W(i);	scloud->points.push_back(p);
		//if(W(i) > 0.001){	pcl::PointXYZRGBNormal p;p.x = X(0,i);p.y = X(1,i);p.z = X(2,i);p.b = 0;p.g = 255;	p.r = 0;	scloud->points.push_back(p);}
		//else{				pcl::PointXYZRGBNormal p;p.x = X(0,i);p.y = X(1,i);p.z = X(2,i);p.b = 0;p.g = 0;	p.r = 255;	scloud->points.push_back(p);}
	}
	for(unsigned int i = 0; i < d_nr_data; i+=1){pcl::PointXYZRGBNormal p;p.x = Y(0,i);p.y = Y(1,i);p.z = Y(2,i);p.b = 255;p.g = 0;p.r = 0;dcloud->points.push_back(p);}
	viewer->addPointCloud<pcl::PointXYZRGBNormal> (scloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(scloud), "scloud");
	viewer->addPointCloud<pcl::PointXYZRGBNormal> (dcloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(dcloud), "dcloud");
	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "scloud");
	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "dcloud");
	//printf("pre spin\n");
    viewer->spin();
	//printf("post spin\n");
	viewer->removeAllPointClouds();
}

}
