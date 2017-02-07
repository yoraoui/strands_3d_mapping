#include "registration/Registration.h"
//#include "ICP.h"
namespace reglib
{

Registration::Registration(){
	only_initial_guess = false;
}
Registration::~Registration(){}

void Registration::setSrc(std::vector<superpoint> & src_){src = src_;}
void Registration::setDst(std::vector<superpoint> & dst_){dst = dst_;}

void Registration::setSrc(Model * src, bool recompute){
    src_kp = src->keypoints;
    setSrc(src->points);
}
void Registration::setDst(Model * dst, bool recompute){
    dst_kp = dst->keypoints;
    setDst(dst->points);
}

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
    printf("TOTAL TIME: %5.5fs\n",sum);
	printf("========================================================\n");
}

void Registration::show(double * X, unsigned int nr_X, double * Y, unsigned int nr_Y, bool stop, std::vector<double> weights){
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr dcloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    scloud->points.resize(nr_X);
    dcloud->points.resize(nr_Y);

    for(unsigned int i = 0; i < nr_X; i++){
        scloud->points[i].x = X[3*i+0];
        scloud->points[i].y = X[3*i+1];
        scloud->points[i].z = X[3*i+2];
        scloud->points[i].b = 0;
        scloud->points[i].g = 255;
        scloud->points[i].r = 0;

        if(weights.size() == nr_X){
            scloud->points[i].b = 255.0*weights[i];
            scloud->points[i].g = 255.0*weights[i];
            scloud->points[i].r = 255.0*weights[i];
        }
    }

    for(unsigned int i = 0; i < nr_Y; i++){
        dcloud->points[i].x = Y[3*i+0];
        dcloud->points[i].y = Y[3*i+1];
        dcloud->points[i].z = Y[3*i+2];
        dcloud->points[i].b = 0;
        dcloud->points[i].g = 0;
        dcloud->points[i].r = 255;
    }

    viewer->addPointCloud<pcl::PointXYZRGB> (scloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(scloud), "scloud");
    viewer->addPointCloud<pcl::PointXYZRGB> (dcloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(dcloud), "dcloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "scloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "dcloud");
    viewer->spin();
    viewer->removeAllPointClouds();
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


void Registration::show(std::vector<superpoint> & X, std::vector<superpoint> & Y, Eigen::Matrix4d p,std::vector<double> weights, bool stop){
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr dcloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    unsigned int s_nr_data = X.size();
    unsigned int d_nr_data = Y.size();
    scloud->points.clear();
    dcloud->points.clear();

    const double & m00 = p(0,0); const double & m01 = p(0,1); const double & m02 = p(0,2); const double & m03 = p(0,3);
    const double & m10 = p(1,0); const double & m11 = p(1,1); const double & m12 = p(1,2); const double & m13 = p(1,3);
    const double & m20 = p(2,0); const double & m21 = p(2,1); const double & m22 = p(2,2); const double & m23 = p(2,3);
    for(unsigned int i = 0; i < s_nr_data; i++){
        superpoint & sp = X[i];
        double x  = sp.x;
        double y  = sp.y;
        double z  = sp.z;
        pcl::PointXYZRGB p;
        p.x = m00*x + m01*y + m02*z + m03;
        p.y = m10*x + m11*y + m12*z + m13;
        p.z = m20*x + m21*y + m22*z + m23;
        p.b = 0;
        p.g = 255;
        p.r = 0;
//        if(weights.size() == s_nr_data){
//            p.b = 0;
//            p.g = 255.0*weights[i];
//            p.r = 255.0*(1-weights[i]);
//        }
        scloud->points.push_back(p);
    }

    for(unsigned int i = 0; i < d_nr_data; i++){
        superpoint & sp = Y[i];
        pcl::PointXYZRGB p;
        p.x = sp.x;
        p.y = sp.y;
        p.z = sp.z;
        p.b = 255;
        p.g = 0;
        p.r = 0;
        dcloud->points.push_back(p);
    }

    viewer->addPointCloud<pcl::PointXYZRGB> (scloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(scloud), "scloud");
    viewer->addPointCloud<pcl::PointXYZRGB> (dcloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(dcloud), "dcloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "scloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "dcloud");
    if(stop){viewer->spin();}
    viewer->removeAllPointClouds();
}

void Registration::show(std::vector<KeyPoint> & X, std::vector<KeyPoint> & Y, Eigen::Matrix4d p, std::vector<int> matches, std::vector<double> weights, bool stop){
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr dcloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    unsigned int s_nr_data = X.size();
    unsigned int d_nr_data = Y.size();
    scloud->points.clear();
    dcloud->points.clear();

    const double & m00 = p(0,0); const double & m01 = p(0,1); const double & m02 = p(0,2); const double & m03 = p(0,3);
    const double & m10 = p(1,0); const double & m11 = p(1,1); const double & m12 = p(1,2); const double & m13 = p(1,3);
    const double & m20 = p(2,0); const double & m21 = p(2,1); const double & m22 = p(2,2); const double & m23 = p(2,3);


    for(unsigned int i = 0; i < s_nr_data; i++){
        superpoint & sp = X[i].point;
        double x  = sp.x;
        double y  = sp.y;
        double z  = sp.z;
        pcl::PointXYZRGB p;
        p.x = m00*x + m01*y + m02*z + m03;
        p.y = m10*x + m11*y + m12*z + m13;
        p.z = m20*x + m21*y + m22*z + m23;
        p.b = 255;
        p.g = 255;
        p.r = 255;
        scloud->points.push_back(p);
    }

    for(unsigned int i = 0; i < d_nr_data; i++){
        superpoint & sp = Y[i].point;
        pcl::PointXYZRGB p;
        p.x = sp.x;
        p.y = sp.y;
        p.z = sp.z;
        p.b = 0;
        p.g = 0;
        p.r = 255;
        dcloud->points.push_back(p);
    }

    if(matches.size() == s_nr_data){
        viewer->removeAllShapes();
        char buf [1024];

        for(unsigned int i = 0; i < s_nr_data; i++){

            pcl::PointXYZRGB & p = scloud->points[i];
            p.b = 0;
            p.g = 255;
            p.r = 0;

            int id = matches[i];
            if(id < 0){
                p.b = 0;
                p.g = 0;
                p.r = 255;
            }else{
                pcl::PointXYZRGB p2 = dcloud->points[id];
                p2.b = 0;
                p2.g = 255;
                p2.r = 0;
            }
        }

        for(unsigned int i = 0; i < s_nr_data; i++){
            int id = matches[i];
            if(id < 0){continue;}
            double w = 1;//
            if(weights.size() == s_nr_data){w = weights[i];}
            pcl::PointXYZRGB p1 = scloud->points[i];
            pcl::PointXYZRGB p2 = dcloud->points[id];
            sprintf(buf,"line%i",i);
            //viewer->addLine<pcl::PointXYZRGB> (p1,p2,255.0*(1-w),255*w,0,buf);
            if(w > 0.000001){
                viewer->addLine<pcl::PointXYZRGB> (p1,p2,w,w,w,buf);
            }
        }
    }


    viewer->addPointCloud<pcl::PointXYZRGB> (scloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(scloud), "scloud");
    viewer->addPointCloud<pcl::PointXYZRGB> (dcloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(dcloud), "dcloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "scloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "dcloud");
    if(stop){viewer->spin();}
    viewer->removeAllPointClouds();
}


void Registration::show(double * sp, unsigned int nr_sp, double * dp,unsigned int nr_dp,  Eigen::Matrix4d p, bool stop, std::vector<double> weights){
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr dcloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    scloud->points.clear();
    dcloud->points.clear();

    const double & m00 = p(0,0); const double & m01 = p(0,1); const double & m02 = p(0,2); const double & m03 = p(0,3);
    const double & m10 = p(1,0); const double & m11 = p(1,1); const double & m12 = p(1,2); const double & m13 = p(1,3);
    const double & m20 = p(2,0); const double & m21 = p(2,1); const double & m22 = p(2,2); const double & m23 = p(2,3);


    for(unsigned int i = 0; i < nr_sp; i++){
        double x  = sp[3*i+0];
        double y  = sp[3*i+1];
        double z  = sp[3*i+2];
        pcl::PointXYZRGB p;
        p.x = m00*x + m01*y + m02*z + m03;
        p.y = m10*x + m11*y + m12*z + m13;
        p.z = m20*x + m21*y + m22*z + m23;
        p.b = 255*weights[i];
        p.g = 255*weights[i];
        p.r = 255*weights[i];
        scloud->points.push_back(p);
    }

    for(unsigned int i = 0; i < nr_dp; i++){
        pcl::PointXYZRGB p;
        p.x = dp[3*i+0];
        p.y = dp[3*i+1];
        p.z = dp[3*i+2];
        p.b = 0;
        p.g = 0;
        p.r = 255;
        dcloud->points.push_back(p);
    }

//    if(matches.size() == s_nr_data){
//        viewer->removeAllShapes();
//        char buf [1024];
//        for(unsigned int i = 0; i < s_nr_data; i++){
//            pcl::PointXYZRGB & p = scloud->points[i];
//            p.b = 0;
//            p.g = 255;
//            p.r = 0;
//            int id = matches[i];
//            if(id < 0){
//                p.b = 0;
//                p.g = 0;
//                p.r = 255;
//            }else{
//                pcl::PointXYZRGB p2 = dcloud->points[id];
//                p2.b = 0;
//                p2.g = 255;
//                p2.r = 0;
//            }
//        }
//        for(unsigned int i = 0; i < s_nr_data; i++){
//            int id = matches[i];
//            if(id < 0){continue;}
//            double w = 1;//
//            if(weights.size() == s_nr_data){w = weights[i];}
//            pcl::PointXYZRGB p1 = scloud->points[i];
//            pcl::PointXYZRGB p2 = dcloud->points[id];
//            sprintf(buf,"line%i",i);
//            //viewer->addLine<pcl::PointXYZRGB> (p1,p2,255.0*(1-w),255*w,0,buf);
//            if(w > 0.000001){
//                viewer->addLine<pcl::PointXYZRGB> (p1,p2,w,w,w,buf);
//            }
//        }
//    }


    viewer->addPointCloud<pcl::PointXYZRGB> (scloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(scloud), "scloud");
    viewer->addPointCloud<pcl::PointXYZRGB> (dcloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(dcloud), "dcloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "scloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "dcloud");
    if(stop){viewer->spin();}
    viewer->removeAllPointClouds();
}

std::string Registration::getString(){
    return "";
}

}
