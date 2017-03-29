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

//#include "modelupdater/ModelUpdater.h"
#include "core/RGBDFrame.h"

#include <string.h>

#include "Util/Util.h"

#include "core/DescriptorExtractor.h"

using namespace std;

int counter = 0;

std::string scriptPath = "test.txt";

reglib::Camera *						camera;
std::vector<reglib::RGBDFrame *>		frames;

std::vector<cv::Mat> rgb_images;
std::vector<string> rgb_images_path;
std::vector<cv::Mat> depth_images;
std::vector<string> depth_images_path;
std::vector<double> timestamps;
std::vector<double> gttimestamps;

std::vector<Eigen::Matrix4d> gt_poses;

void addFBdata(string path, int start, int stop, int skip){

	rgb_images.clear();
	depth_images.clear();
	rgb_images_path.clear();
	depth_images_path.clear();
	timestamps.clear();
	gttimestamps.clear();
	gt_poses.clear();

	string input 		= path;
	string input_rgb	= string(input+"/rgb.txt");
	string input_depth	= string(input+"/depth.txt");
	string input_gt		= string(input+"/groundtruth.txt");
	printf("input:       %s\n",input.c_str());
	//printf("input_rgb:   %s\n",input_rgb.c_str());
	//printf("input_depth: %s\n",input_depth.c_str());


	int camera_id = 0;
	if(input.find("rgbd_dataset_freiburg1") != -1){camera_id = 1;}
	if(input.find("rgbd_dataset_freiburg2") != -1){camera_id = 2;}
	if(input.find("rgbd_dataset_freiburg3") != -1){camera_id = 3;}
	//printf("Camer_id: %i\n",camera_id);

	camera->idepth_scale = 1.0/5000.0;

//    Freiburg 1 Depth 	1.035
//    Freiburg 2 Depth 	1.031
//    Freiburg 3 Depth 	1.000

	if(camera_id == 0){
		camera->fx			= 525.0;				//Focal Length X
		camera->fy			= 525.0;				//Focal Length Y
		camera->cx			= 319.5;				//Center coordinate X
		camera->cy			= 239.5;				//Center coordinate X
		//camera->idepth_scale /= 1.000;
	}else if(camera_id == 1){
		camera->fx			= 517.3;				//Focal Length X
		camera->fy			= 516.5;				//Focal Length Y
		camera->cx			= 318.6;				//Center coordinate X
		camera->cy			= 255.3;				//Center coordinate X
		//camera->idepth_scale /= 1.035;
	}else if(camera_id == 2){
		camera->fx			= 520.9;				//Focal Length X
		camera->fy			= 521.0;				//Focal Length Y
		camera->cx			= 325.1;				//Center coordinate X
		camera->cy			= 249.7;				//Center coordinate X
		//camera->idepth_scale /= 1.031;
	}else if(camera_id == 3){
		camera->fx			= 535.4;				//Focal Length X
		camera->fy			= 539.2;				//Focal Length Y
		camera->cx			= 320.1;				//Center coordinate X
		camera->cy			= 247.6;				//Center coordinate X
		//camera->idepth_scale /= 1.000;
	}else{printf("Error, should not get to here\n");}

	string line;

	ifstream rgb_file (input_rgb.c_str());
	vector<pair<double, string> > rgb_lines;
	if (rgb_file.is_open()){
		while ( rgb_file.good()){
			getline (rgb_file,line);
			if(line[0] != '#'){
				int space1 = line.find(" ");
				if(space1 != -1){
					int dot1		= line.find(".");
					string secs		= line.substr(0,dot1);
					string nsecs	= line.substr(dot1+1,space1-dot1);
					string path		= line.substr(space1+1);
					double timestamp = double(atoi(secs.c_str()))+0.000001*double(atoi(nsecs.c_str()));
					rgb_lines.push_back(make_pair(timestamp,path));
				}
			}
		}
		rgb_file.close();
	}else{cout << "Unable to open " << input;}

	ifstream depth_file (input_depth.c_str());
	vector<pair<double, string> > depth_lines;
	if (depth_file.is_open()){
		while ( depth_file.good()){
			getline (depth_file,line);
			if(line[0] != '#'){
				int space1 = line.find(" ");
				if(space1 != -1){
					int dot1		= line.find(".");
					string secs		= line.substr(0,dot1);
					string nsecs	= line.substr(dot1+1,space1-dot1);
					string path		= line.substr(space1+1);
					double timestamp = double(atoi(secs.c_str()))+0.000001*double(atoi(nsecs.c_str()));
					depth_lines.push_back(make_pair(timestamp,path));
				}
			}
		}
		depth_file.close();
	}else{cout << "Unable to open " << input;}


	ifstream gt_file (input_gt.c_str());
	vector<pair<double, Eigen::Matrix4d> > gt_lines;
	if (gt_file.is_open()){
		while ( gt_file.good()){
			getline (gt_file,line);
			if(line[0] != '#'){
				int space1 = line.find(" ");
				if(space1 != -1){
					int dot1		= line.find(".");
					string secs		= line.substr(0,dot1);
					string nsecs	= line.substr(dot1+1,space1-dot1);
					string path		= line.substr(space1+1);

					std::vector<double> v;
					for(unsigned int i = 0; i < 7; i++){
						int space2 = path.find(" ");
						v.push_back(atof(path.substr(0,space2).c_str()));
						path = path.substr(space2+1);
					}

					Eigen::Quaterniond qr(v[6],v[3],v[4],v[5]);//a.rotation());

					Eigen::Affine3d a(qr);
					a(0,3) = v[0];
					a(1,3) = v[1];
					a(2,3) = v[2];

					double timestamp = atof(line.substr(0,space1).c_str());//double(atoi(secs.c_str()))+0.000001*double(atoi(nsecs.c_str()));
					gt_lines.push_back(make_pair(timestamp,a.matrix()));
				}
			}
		}
		depth_file.close();
	}else{cout << "Unable to open " << input;}

	unsigned int rgb_counter = 0;
	unsigned int depth_counter = 0;
	unsigned int gt_counter = 0;

	vector<int> rgb_indexes;
	vector<int> depth_indexes;
	vector<int> gt_indexes;

	float max_diff = 0.03;

	for(; rgb_counter < rgb_lines.size(); rgb_counter++){
		double rgb_ts		= rgb_lines.at(rgb_counter).first;
		double depth_ts		= depth_lines.at(depth_counter).first;
		double gt_ts		= gt_lines.at(gt_counter).first;
		double diff_best	= fabs(rgb_ts - depth_ts);
		double gtdiff_best	= fabs(rgb_ts - gt_ts);

		for(unsigned int current_counter = depth_counter; current_counter < depth_lines.size(); current_counter++){
			double dts = depth_lines.at(current_counter).first;
			double diff_current = fabs(rgb_ts - dts);
			if(diff_current <= diff_best){
				diff_best = diff_current;
				depth_counter = current_counter;
				depth_ts = dts;
			}else{break;}
		}

		for(unsigned int current_counter = gt_counter; current_counter < gt_lines.size(); current_counter++){
		//for(unsigned int current_counter = 0; current_counter < gt_lines.size(); current_counter++){
			double dts = gt_lines.at(current_counter).first;
			double diff_current = fabs(rgb_ts - dts);
			//if(rgb_indexes.size() < 10){printf("rgb: %5.5fs gt: %5.5fs diff: %5.5fs\n",rgb_ts,dts,diff_current);}
			if(diff_current <= gtdiff_best){
				gtdiff_best = diff_current;
				gt_counter = current_counter;
				gt_ts = dts;
			}else{break;}

		}

		if(diff_best > max_diff){continue;}//Failed to find corresponding depth image
		rgb_indexes.push_back(rgb_counter);
		depth_indexes.push_back(depth_counter);
		gt_indexes.push_back(gt_counter);
	}

	for(unsigned int i = start; i < rgb_indexes.size() && i < stop; i+= skip){
		string rgbpath = input+"/"+rgb_lines.at(rgb_indexes.at(i)).second;
		string depthpath = input+"/"+depth_lines.at(depth_indexes.at(i)).second;

		cv::Mat rgbimage    = cv::imread(rgbpath, CV_LOAD_IMAGE_COLOR);
		cv::Mat depthimage  = cv::imread(depthpath, CV_LOAD_IMAGE_UNCHANGED);

		rgb_images.push_back(rgbimage);
		depth_images.push_back(depthimage);
		timestamps.push_back(depth_lines.at(depth_indexes.at(i)).first);
		gt_poses.push_back(gt_lines.at(gt_indexes.at(i)).second);

		//std::cout << "\r loaded " << i << " " << std::cout.flush();
		//printf("\r loaded %i",i);
		//fflush(stdout);

	}
	//printf("\r                                                   \r");
	//fflush(stdout);
	Eigen::Matrix4d finv = gt_poses.front().inverse();
	for(unsigned int i = 0; i < gt_poses.size(); i++){
		gt_poses[i] = finv*gt_poses[i];
	}
}

void addFBdata2(string path, int start, int stop, int skip){

	rgb_images.clear();
	depth_images.clear();
	rgb_images_path.clear();
	depth_images_path.clear();
	timestamps.clear();
	gttimestamps.clear();
	gt_poses.clear();

	string input 		= path;
	string input_rgb	= string(input+"/rgb.txt");
	string input_depth	= string(input+"/depth.txt");
	string input_gt		= string(input+"/groundtruth.txt");
	printf("input:       %s\n",input.c_str());
	//printf("input_rgb:   %s\n",input_rgb.c_str());
	//printf("input_depth: %s\n",input_depth.c_str());


	int camera_id = 0;
	if(input.find("rgbd_dataset_freiburg1") != -1){camera_id = 1;}
	if(input.find("rgbd_dataset_freiburg2") != -1){camera_id = 2;}
	if(input.find("rgbd_dataset_freiburg3") != -1){camera_id = 3;}
	//printf("Camer_id: %i\n",camera_id);

	camera->idepth_scale = 1.0/5000.0;

//    Freiburg 1 Depth 	1.035
//    Freiburg 2 Depth 	1.031
//    Freiburg 3 Depth 	1.000

	if(camera_id == 0){
		camera->fx			= 525.0;				//Focal Length X
		camera->fy			= 525.0;				//Focal Length Y
		camera->cx			= 319.5;				//Center coordinate X
		camera->cy			= 239.5;				//Center coordinate X
		//camera->idepth_scale /= 1.000;
	}else if(camera_id == 1){
		camera->fx			= 517.3;				//Focal Length X
		camera->fy			= 516.5;				//Focal Length Y
		camera->cx			= 318.6;				//Center coordinate X
		camera->cy			= 255.3;				//Center coordinate X
		//camera->idepth_scale /= 1.035;
	}else if(camera_id == 2){
		camera->fx			= 520.9;				//Focal Length X
		camera->fy			= 521.0;				//Focal Length Y
		camera->cx			= 325.1;				//Center coordinate X
		camera->cy			= 249.7;				//Center coordinate X
		//camera->idepth_scale /= 1.031;
	}else if(camera_id == 3){
		camera->fx			= 535.4;				//Focal Length X
		camera->fy			= 539.2;				//Focal Length Y
		camera->cx			= 320.1;				//Center coordinate X
		camera->cy			= 247.6;				//Center coordinate X
		//camera->idepth_scale /= 1.000;
	}else{printf("Error, should not get to here\n");}

	string line;

	ifstream rgb_file (input_rgb.c_str());
	vector<pair<double, string> > rgb_lines;
	if (rgb_file.is_open()){
		while ( rgb_file.good()){
			getline (rgb_file,line);
			if(line[0] != '#'){
				int space1 = line.find(" ");
				if(space1 != -1){
					int dot1		= line.find(".");
					string secs		= line.substr(0,dot1);
					string nsecs	= line.substr(dot1+1,space1-dot1);
					string path		= line.substr(space1+1);
					double timestamp = double(atoi(secs.c_str()))+0.000001*double(atoi(nsecs.c_str()));
					rgb_lines.push_back(make_pair(timestamp,path));
				}
			}
		}
		rgb_file.close();
	}else{cout << "Unable to open " << input;}

	ifstream depth_file (input_depth.c_str());
	vector<pair<double, string> > depth_lines;
	if (depth_file.is_open()){
		while ( depth_file.good()){
			getline (depth_file,line);
			if(line[0] != '#'){
				int space1 = line.find(" ");
				if(space1 != -1){
					int dot1		= line.find(".");
					string secs		= line.substr(0,dot1);
					string nsecs	= line.substr(dot1+1,space1-dot1);
					string path		= line.substr(space1+1);
					double timestamp = double(atoi(secs.c_str()))+0.000001*double(atoi(nsecs.c_str()));
					depth_lines.push_back(make_pair(timestamp,path));
				}
			}
		}
		depth_file.close();
	}else{cout << "Unable to open " << input;}


	ifstream gt_file (input_gt.c_str());
	vector<pair<double, Eigen::Matrix4d> > gt_lines;
	if (gt_file.is_open()){
		while ( gt_file.good()){
			getline (gt_file,line);
			if(line[0] != '#'){
				int space1 = line.find(" ");
				if(space1 != -1){
					int dot1		= line.find(".");
					string secs		= line.substr(0,dot1);
					string nsecs	= line.substr(dot1+1,space1-dot1);
					string path		= line.substr(space1+1);

					std::vector<double> v;
					for(unsigned int i = 0; i < 7; i++){
						int space2 = path.find(" ");
						v.push_back(atof(path.substr(0,space2).c_str()));
						path = path.substr(space2+1);
					}

					Eigen::Quaterniond qr(v[6],v[3],v[4],v[5]);//a.rotation());

					Eigen::Affine3d a(qr);
					a(0,3) = v[0];
					a(1,3) = v[1];
					a(2,3) = v[2];

					double timestamp = atof(line.substr(0,space1).c_str());//double(atoi(secs.c_str()))+0.000001*double(atoi(nsecs.c_str()));
					gt_lines.push_back(make_pair(timestamp,a.matrix()));
				}
			}
		}
		depth_file.close();
	}else{cout << "Unable to open " << input;}

	unsigned int rgb_counter = 0;
	unsigned int depth_counter = 0;
	unsigned int gt_counter = 0;

	vector<int> rgb_indexes;
	vector<int> depth_indexes;
	vector<int> gt_indexes;

	float max_diff = 0.015;

	for(; depth_counter < depth_lines.size(); depth_counter++){
		double rgb_ts		= rgb_lines.at(rgb_counter).first;
		double depth_ts		= depth_lines.at(depth_counter).first;
		double gt_ts		= gt_lines.at(gt_counter).first;
		double diff_best	= fabs(rgb_ts - depth_ts);
		double gtdiff_best	= fabs(depth_ts - gt_ts);

		//for(unsigned int current_counter = rgb_counter; current_counter < rgb_lines.size(); current_counter++){
		for(unsigned int current_counter = rgb_counter; current_counter < rgb_lines.size(); current_counter++){
			double rgbts = rgb_lines.at(current_counter).first;
			double diff_current = fabs(depth_ts - rgbts);
			//printf("current_counter: %i depth_ts: %7.7fs rgb_ts: %7.7f -> %f\n",current_counter,depth_ts,rgb_ts,diff_current);
			if(diff_current <= diff_best){
				diff_best = diff_current;
				rgb_counter = current_counter;
				rgb_ts = rgbts;
			}else{break;}
		}

		for(unsigned int current_counter = gt_counter; current_counter < gt_lines.size(); current_counter++){
			double dts = gt_lines.at(current_counter).first;
			double diff_current = fabs(depth_ts - dts);
			//printf("current_counter: %i depth_ts: %7.7fs dts_ts: %7.7f -> %f\n",current_counter,dts,rgb_ts,diff_current);
			//if(rgb_indexes.size() < 10){printf("rgb: %5.5fs gt: %5.5fs diff: %5.5fs\n",rgb_ts,dts,diff_current);}
			if(diff_current <= gtdiff_best){
				gtdiff_best = diff_current;
				gt_counter = current_counter;
				gt_ts = dts;
			}else{break;}

		}
//exit(0);
		//printf("%i -> depth_ts: %7.7fs rgb_ts: %7.7f diff: %7.7fs diff2: %7.7fs\n",depth_counter,depth_ts,rgb_ts,fabs(depth_ts-rgb_ts),diff_best);
		rgb_indexes.push_back(rgb_counter);
		depth_indexes.push_back(depth_counter);
		gt_indexes.push_back(gt_counter);
		//printf("%i %i\n",depth_counter,gt_counter);
	}

	for(unsigned int i = start; i < depth_indexes.size() && i < stop; i+= skip){
		string rgbpath = input+"/"+rgb_lines.at(rgb_indexes.at(i)).second;
		string depthpath = input+"/"+depth_lines.at(depth_indexes.at(i)).second;

		rgb_images_path.push_back(rgbpath);
		depth_images_path.push_back(depthpath);

//        cv::Mat rgbimage    = cv::imread(rgbpath, CV_LOAD_IMAGE_COLOR);
//        cv::Mat depthimage  = cv::imread(depthpath, CV_LOAD_IMAGE_UNCHANGED);

//        rgb_images.push_back(rgbimage);
//        depth_images.push_back(depthimage);
		timestamps.push_back(depth_lines.at(depth_indexes.at(i)).first);
		gt_poses.push_back(gt_lines.at(gt_indexes.at(i)).second);

		std::cout << "\r loaded " << i << " " << std::cout.flush();
		printf("\r loaded %i",i);
		fflush(stdout);

	}

	Eigen::Matrix4d finv = gt_poses.front().inverse();
	for(unsigned int i = 0; i < gt_poses.size(); i++){
		gt_poses[i] = finv*gt_poses[i];
	}
	//exit(0);
}

void saveModelToFB(std::vector<Eigen::Matrix4d> poses, std::vector<double> timestamps, string datapath = "./", string name = "testoutput.txt", string benchmarkpath = "./rgbd_benchmark_tools/scripts/"){
	string pathname = datapath+"/"+name;
	string figurename = name;
	figurename.pop_back();
	figurename.pop_back();
	figurename.pop_back();
	figurename.pop_back();
	printf("Saving map in: %s\n",pathname.c_str());


	ofstream myfile;
	myfile.open((pathname).c_str());

	char buf[1024];
	for(unsigned int i = 0; i < poses.size(); i++){
		Matrix4d p = poses[i];
		Eigen::Affine3d a(p.cast<double>());
		Eigen::Quaterniond qr(a.rotation());
		double timestamp = timestamps[i];
		float tx = a(0,3);
		float ty = a(1,3);
		float tz = a(2,3);
		float qx = qr.x();
		float qy = qr.y();
		float qz = qr.z();
		float qw = qr.w();
		int n = sprintf(buf,"%f %f %f %f %f %f %f %f\n",timestamp,tx,ty,tz,qx,qy,qz,qw);
		myfile << buf;
	}
	myfile.close();


	//printf("%s",buf);
	std::ofstream outfile;
	outfile.open(scriptPath, std::ios_base::app);

	sprintf(buf,"echo %s\n",pathname.c_str());
	outfile << std::string(buf);

	sprintf(buf,"%sevaluate_rpe.py %s/groundtruth.txt %s --max_pairs 10000 --fixed_delta  --delta 1 --delta_unit s --plot %s/%s_figure_rpe.png --offset 0 --scale 1 --verbose | grep translational_error.rmse\n",benchmarkpath.c_str(),datapath.c_str(),pathname.c_str(),datapath.c_str(),figurename.c_str());
	//printf("%s",buf);
	//system(buf);
	outfile << std::string(buf);

	sprintf(buf,"%sevaluate_ate.py %s/groundtruth.txt %s --plot %s/%s_figure_ate.png --verbose | grep absolute_translational_error.rmse\n",benchmarkpath.c_str(),datapath.c_str(),pathname.c_str(),datapath.c_str(),figurename.c_str());
	//printf("%s",buf);
	//system(buf);
	outfile << std::string(buf);

}

std::vector<Eigen::Matrix4d> slam_vo2(reglib::DistanceWeightFunction2 * func, reglib::Registration * reg, reglib::DescriptorExtractor * de, string datapath = "./", string name = "testoutput.txt", string benchmarkpath = "./rgbd_benchmark_tools/scripts/", bool fuse_surface = true, bool fuse_kp = true, bool visualize = true){
	printf("slam_vo2\n");
	saveModelToFB(gt_poses,timestamps,datapath,name+"_gtodometry.txt",benchmarkpath);
	unsigned int nr_frames = rgb_images.size();
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
	pcl::PointCloud<pcl::PointXYZ>::Ptr    coordcloud	(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr    gt_coordcloud	(new pcl::PointCloud<pcl::PointXYZ>);

	if(visualize){
		viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer ("viewer"));
		viewer->setBackgroundColor(0.8,0.8,0.8);
		viewer->removeAllPointClouds();
		viewer->removeAllShapes();

		coordcloud->points.resize(4);
		coordcloud->points[0].x = 0.00; coordcloud->points[0].y = 0.00; coordcloud->points[0].z = 0.00;
		coordcloud->points[1].x = 0.02; coordcloud->points[1].y = 0.00; coordcloud->points[1].z = 0.00;
		coordcloud->points[2].x = 0.00; coordcloud->points[2].y = 0.02; coordcloud->points[2].z = 0.00;
		coordcloud->points[3].x = 0.00; coordcloud->points[3].y = 0.00; coordcloud->points[3].z = 0.02;

		for(unsigned int i = 0; i < gt_poses.size(); i++){
			pcl::PointCloud<pcl::PointXYZ>::Ptr    cloudCoord	(new pcl::PointCloud<pcl::PointXYZ>);
			char buf [1024];
			pcl::transformPointCloud (*coordcloud, *cloudCoord, gt_poses[i]);
			if(i > 0){
				pcl::PointXYZ p; p.x = gt_poses[i-1](0,3);p.y = gt_poses[i-1](1,3); p.z = gt_poses[i-1](2,3);
				gt_coordcloud->points.push_back(p);
				sprintf(buf,"gt_%i",i);viewer->addLine<pcl::PointXYZ> (cloudCoord->points[0],p,0,0,255,buf);
			}
		}
		viewer->spinOnce();
	}

	std::vector<Eigen::Matrix4d> poses;

	cv::Mat fullmask;
	fullmask.create(480,640,CV_8UC1);
	unsigned char * maskdata = (unsigned char *)fullmask.data;
	for(unsigned int j = 0; j < 480*640; j++){maskdata[j] = 255;}

	std::vector<reglib::Model * >       keyframes;
	std::vector< int >                  keyframe_ind;
	std::vector<Eigen::Matrix4d>        keyframe_poses;
	std::vector<double>                 keyframe_time;
	std::vector<reglib::RGBDFrame *>    keyframes_frames;

	reglib::Model * kf = new reglib::Model();
	kf->last_changed = 0;

	reglib::Model * current = new reglib::Model();
	current->relativeposes.push_back(Eigen::Matrix4d::Identity());
	current->modelmasks.push_back(new reglib::ModelMask(fullmask));
	current->frames.push_back(0);


	reglib::MassRegistrationPPR3 * massreg = new reglib::MassRegistrationPPR3(func);
	massreg->viewer = viewer;
	massreg->visualizationLvl = 0;
	massreg->convergence_mul = 1.0;
	massreg->func_setup = 0;

	//reglib::MassFusionResults bgmfr3 = bgmassreg3->getTransforms(keyframe_poses);


	//register to last kf
	std::vector<Eigen::Matrix4d> cp;
	cp.push_back(Eigen::Matrix4d::Identity());
	cp.push_back(Eigen::Matrix4d::Identity());

	std::vector<bool> is_kf;

	int kfstep = 1;//10;//std::max(1.0,double(nr_frames)/20.0);
	//Merge into a new kf

	//massreg->visualizationLvl = 0;

	reglib::RGBDFrame * prev = 0;
	for(unsigned int i = 0; i < nr_frames; i++){
//        if(i > 40){
//            massreg->visualizationLvl = 2;
//        }
//		if(i > 0 && i < 110){

//			Eigen::Matrix4d gtcurrPose = gt_poses[i];
//			poses.push_back(gtcurrPose);
//			is_kf.push_back(false);

//					Eigen::Matrix4d currPose = poses.back();


//					if(visualize){
//						char buf [1024];
//						pcl::PointXYZ currp; currp.x = currPose(0,3);currp.y = currPose(1,3); currp.z = currPose(2,3);
//						pcl::PointXYZ gtp; gtp.x = gtcurrPose(0,3);gtp.y = gtcurrPose(1,3); gtp.z = gtcurrPose(2,3);
//						sprintf(buf,"link_%i",i);viewer->addLine<pcl::PointXYZ> (gtp,currp,255,0,255,buf);
//					}

//					if(visualize){
//						pcl::PointCloud<pcl::PointXYZ>::Ptr    cloudCoord	(new pcl::PointCloud<pcl::PointXYZ>);
//						char buf [1024];
//						pcl::transformPointCloud (*coordcloud, *cloudCoord, Eigen::Affine3f(poses.back().cast<float>()));
//						pcl::PointXYZ p; p.x = poses[i-1](0,3);p.y = poses[i-1](1,3); p.z = poses[i-1](2,3);
//						sprintf(buf,"prev_%i",i);viewer->addLine<pcl::PointXYZ> (cloudCoord->points[0],p,0,255,0,buf);

//					}
//continue;
//					//if(!is_kf.back()){delete prev;}
//		}

		reglib::RGBDFrame * frame = new reglib::RGBDFrame(camera,rgb_images[i],depth_images[i],timestamps[i], Eigen::Matrix4d::Identity(),true,"", false);
		//frame->show();


		std::vector< reglib::KeyPoint > kps = de->extract(frame);
		for(unsigned int j = 0; j < kps.size(); j++){ kps[j].point.last_update_frame_id = i; }
		current->frames[0] = frame;
		current->keypoints = kps;
		current->recomputeModelPoints();
		current->points = frame->getSuperPoints(Eigen::Matrix4d::Identity(), 2, false);
		current->color_edgepoints = frame->getEdges(Eigen::Matrix4d::Identity(), 1, 0);

		bool is_keyframe = false;
		if(i == 0){//First frame is always a kf
			kf->frames.push_back(frame);
			kf->relativeposes.push_back(Eigen::Matrix4d::Identity());
			kf->modelmasks.push_back(new reglib::ModelMask(fullmask));
			poses.push_back(Eigen::Matrix4d::Identity());
			is_keyframe = true;
			is_kf.push_back(true);

			reglib::Model * new_kf = new reglib::Model();
			keyframes.push_back(new_kf);
			new_kf->modelmasks.push_back(kf->modelmasks.back());
			new_kf->relativeposes.push_back(Eigen::Matrix4d::Identity());
			kf->addSuperPoints(kf->points,poses.back(),frame,kf->modelmasks.back());
			massreg->addModel(kf);

			keyframe_ind.push_back(i);
			keyframes_frames.push_back(frame);
			keyframe_time.push_back(timestamps[keyframe_ind.back()]);
			keyframe_poses.push_back(poses[keyframe_ind.back()]);
			new_kf->frames.push_back(frame);
			new_kf->recomputeModelPoints();

			if(visualize){
				viewer->removeAllPointClouds();
				pcl::PointCloud<pcl::PointXYZRGB>::Ptr cld = kf->getPCLcloud(1,3);
				viewer->addPointCloud<pcl::PointXYZRGB> (cld, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(cld), "cloud");
				viewer->addCoordinateSystem(0.1,Eigen::Affine3f(keyframe_poses.back().cast<float>()));
				viewer->spinOnce();
			}
		}else if(false && i < 110){

			Eigen::Matrix4d currPose = poses.back();
			Eigen::Matrix4d gtcurrPose = gt_poses[i];
			poses.push_back(gtcurrPose);

			if(visualize){
				char buf [1024];
				pcl::PointXYZ currp; currp.x = currPose(0,3);currp.y = currPose(1,3); currp.z = currPose(2,3);
				pcl::PointXYZ gtp; gtp.x = gtcurrPose(0,3);gtp.y = gtcurrPose(1,3); gtp.z = gtcurrPose(2,3);
				sprintf(buf,"link_%i",i);viewer->addLine<pcl::PointXYZ> (gtp,currp,255,0,255,buf);
				viewer->spinOnce();
			}

			if(visualize){
				pcl::PointCloud<pcl::PointXYZ>::Ptr    cloudCoord	(new pcl::PointCloud<pcl::PointXYZ>);
				char buf [1024];
				pcl::transformPointCloud (*coordcloud, *cloudCoord, Eigen::Affine3f(poses.back().cast<float>()));
				pcl::PointXYZ p; p.x = poses[i-1](0,3);p.y = poses[i-1](1,3); p.z = poses[i-1](2,3);
				sprintf(buf,"prev_%i",i);viewer->addLine<pcl::PointXYZ> (cloudCoord->points[0],p,0,255,0,buf);
				viewer->spinOnce();
			}

			//if(!is_kf.back()){delete prev;}
			prev = frame;
			is_kf.push_back(false);
		}else{
			massreg->addModel(current);
			cp.back() = poses.back();
			//cp.back() = gt_poses[i];

			double regStart2 = quasimodo_brain::getTime();
			reglib::MassFusionResults mfr = massreg->getTransforms(cp);
			massreg->removeLastNode();

			double regtime = quasimodo_brain::getTime()-regStart2;

			//if(mfr.score < 0.0 || keyframe_ind.back() + kfstep < i){//Add keyframe?
			if(keyframe_ind.back() + kfstep < i){
				printf("NEW KEYFRAME\n");
				reglib::Model * new_kf = new reglib::Model();
				keyframes.push_back(new_kf);
				new_kf->modelmasks.push_back(kf->modelmasks.back());
				new_kf->relativeposes.push_back(Eigen::Matrix4d::Identity());
				kf->last_changed = keyframes.size();

				if(is_kf.back()){//If last frame was also a keyframe, we just have to accept the current frame/registration as a new kf
					keyframe_ind.push_back(i);
					keyframes_frames.push_back(frame);
					new_kf->frames.push_back(frame);
					is_kf.push_back(true);
					poses.push_back(mfr.poses.back());

					prev = frame;

					if(visualize){
						pcl::PointCloud<pcl::PointXYZ>::Ptr    cloudCoord	(new pcl::PointCloud<pcl::PointXYZ>);
						char buf [1024];
						pcl::transformPointCloud (*coordcloud, *cloudCoord, Eigen::Affine3f(poses.back().cast<float>()));
						pcl::PointXYZ p; p.x = poses[i-1](0,3);p.y = poses[i-1](1,3); p.z = poses[i-1](2,3);
						sprintf(buf,"prev_%i",i);viewer->addLine<pcl::PointXYZ> (cloudCoord->points[0],p,0,255,0,buf);
						viewer->spinOnce();
					}

				}else{//If last frame was not a keyframe: make it a keyframe and redo current frame
					i--;
					keyframe_ind.push_back(i);
					keyframes_frames.push_back(prev);
					new_kf->frames.push_back(prev);
					is_kf.back() = true;
				}

				keyframe_time.push_back(timestamps[keyframe_ind.back()]);
				keyframe_poses.push_back(poses[keyframe_ind.back()]);
				kf->points.clear();
				kf->addSuperPoints( kf->points,poses.back(),keyframes_frames.back(),kf->modelmasks.back());
				new_kf->recomputeModelPoints();

				int nr_kfp = kf->points.size();
				for(int k = 0; k < nr_kfp; k++){
					if( kf->last_changed - kf->points[k].last_update_frame_id > 20){
						kf->points[k] = kf->points.back();
						kf->points.pop_back();
						k--;
						nr_kfp--;
					}
				}

				massreg->removeLastNode();
				massreg->addModel(kf);

				if(visualize){
					viewer->removeAllPointClouds();
					pcl::PointCloud<pcl::PointXYZRGB>::Ptr cld = kf->getPCLcloud(1,3);
					pcl::PointCloud<pcl::PointXYZRGB>::Ptr edge_cld = kf->getPCLEdgeCloud(1,0);
					viewer->addPointCloud<pcl::PointXYZRGB> (cld, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(cld), "cloud");
					viewer->addPointCloud<pcl::PointXYZRGB> (edge_cld, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(edge_cld), "edge_cloud");
					viewer->addCoordinateSystem(0.1,Eigen::Affine3f(keyframe_poses.back().cast<float>()));
					viewer->spinOnce();
				}

			}else{

				poses.push_back(mfr.poses.back());

				Eigen::Matrix4d currPose = poses.back();
				Eigen::Matrix4d gtcurrPose = gt_poses[i];

				Eigen::Matrix4d regChange = currPose * poses[keyframe_ind.back()].inverse();
				Eigen::Matrix4d gtChange = gtcurrPose * gt_poses[keyframe_ind.back()].inverse();
				double dx = regChange(0,3)-gtChange(0,3);
				double dy = regChange(1,3)-gtChange(1,3);
				double dz = regChange(2,3)-gtChange(2,3);
				double regerror = sqrt(dx*dx+dy*dy+dz*dz);

				printf("current_frame = %i / %i ",i+1,nr_frames);
				printf("keyframe_ind.back(): %i ",keyframe_ind.back());
				printf("mass registration time: %5.5fs score: %6.6f ",regtime,mfr.score);
				printf("error: %5.5f\n",regerror);

				if(visualize){
					char buf [1024];
					pcl::PointXYZ currp; currp.x = currPose(0,3);currp.y = currPose(1,3); currp.z = currPose(2,3);
					pcl::PointXYZ gtp; gtp.x = gtcurrPose(0,3);gtp.y = gtcurrPose(1,3); gtp.z = gtcurrPose(2,3);
					sprintf(buf,"link_%i",i);viewer->addLine<pcl::PointXYZ> (gtp,currp,255,0,255,buf);
					viewer->spinOnce();
				}

				if(visualize){
					pcl::PointCloud<pcl::PointXYZ>::Ptr    cloudCoord	(new pcl::PointCloud<pcl::PointXYZ>);
					char buf [1024];
					pcl::transformPointCloud (*coordcloud, *cloudCoord, Eigen::Affine3f(poses.back().cast<float>()));
					pcl::PointXYZ p; p.x = poses[i-1](0,3);p.y = poses[i-1](1,3); p.z = poses[i-1](2,3);
					sprintf(buf,"prev_%i",i);viewer->addLine<pcl::PointXYZ> (cloudCoord->points[0],p,0,255,0,buf);
					viewer->spinOnce();
				}

				if( visualize && regerror > 0.1){
					massreg->visualizationLvl = 2;
					massreg->addModel(current);
					//cp.back() = poses.back();
					//cp.back() = gt_poses[i];
					//massreg->getTransforms(cp);

					//cp.back() = gt_poses[i];
					massreg->getTransforms(cp);

					massreg->removeLastNode();
					massreg->visualizationLvl = 0;
				}

				//if(!is_kf.back()){delete prev;}
				prev = frame;
				is_kf.push_back(false);
			}
		}
	}

	//for(unsigned int i = 0; i < is_kf.size();i++){printf("%i ",int(is_kf[i]));}printf("\n");

	if(visualize){viewer->spinOnce();}

	saveModelToFB(poses,timestamps,datapath,name+func->name+"_odometry.txt",benchmarkpath);
	//saveModelToFB(gt_poses,timestamps,datapath,name+"_odometry.txt",benchmarkpath);

	for(unsigned int i = 0; i < keyframes.size(); i++){
		delete keyframes[i];
		delete keyframes_frames[i];
	}
	delete kf;
	delete current;
	delete massreg;

//    saveModelToFB(poses2,timestamps,output_path+"odometry2.txt");
//    saveModelToFB(poses3,timestamps,output_path+"odometry3.txt");

//	if(false && visualize){
//        viewer->removeAllPointClouds();
//        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cld = kf->getPCLcloud(1,3);
//        viewer->addPointCloud<pcl::PointXYZRGB> (cld, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(cld), "cloud");
//        viewer->spin();
//    }

//    saveModelToFB(keyframe_poses,keyframe_time,output_path+"preoptimized_ppr3.txt");

//	reglib::MassRegistrationPPR3 * massreg3 = new reglib::MassRegistrationPPR3();
//	massreg3->viewer = viewer;
//	massreg3->visualizationLvl = 2;
//	massreg3->convergence_mul = 0.5;
//	massreg3->func_setup = 1;

//	reglib::MassRegistrationRecursive * massregRec = new reglib::MassRegistrationRecursive(massreg3,15);
//	for(unsigned int i = 0; i < keyframes.size(); i++){massregRec->addModel(keyframes[i]);}
//	reglib::MassFusionResults mrRecFr = massregRec->getTransforms(keyframe_poses);

//exit(0);

//	for(unsigned int i = 0; i < keyframes.size(); i++){massreg3->addModel(keyframes[i]);}
//	reglib::MassFusionResults mr3fr = massreg3->getTransforms(keyframe_poses);
//	delete massreg3;

////	reglib::MassRegistrationPPR2 * bgmassreg3 = new reglib::MassRegistrationPPR2(0.02);
////	bgmassreg3->timeout = 3600;
////	bgmassreg3->viewer = viewer;http://www.mixedmartialarts.com/
////	bgmassreg3->use_surface = true;
////	bgmassreg3->use_depthedge = false;
////	bgmassreg3->visualizationLvl = 1;
////	bgmassreg3->maskstep = 10;
////	bgmassreg3->nomaskstep = 10;
////	bgmassreg3->nomask = true;
////	bgmassreg3->stopval = 0.0005;

////	for(unsigned int i = 0; i < keyframes.size(); i++){
////		bgmassreg3->addModel(keyframes[i]);
////	}
////	reglib::MassFusionResults bgmfr3 = bgmassreg3->getTransforms(keyframe_poses);
////	delete bgmassreg3;


////	reglib::MassRegistrationPPR2 * bgmassreg4 = new reglib::MassRegistrationPPR2(0.0);
////	bgmassreg4->timeout = 3600;
////	bgmassreg4->viewer = viewer;
////	bgmassreg4->use_surface = true;
////	bgmassreg4->use_depthedge = false;
////	bgmassreg4->visualizationLvl = 1;
////	bgmassreg4->maskstep = 4;
////	bgmassreg4->nomaskstep = 4;
////	bgmassreg4->nomask = true;
////	bgmassreg4->stopval = 0.0005;

////	for(unsigned int i = 0; i < keyframes.size(); i++){
////		bgmassreg4->addModel(keyframes[i]);
////	}
////	reglib::MassFusionResults bgmfr4 = bgmassreg3->getTransforms(bgmfr3.poses);
////	delete bgmassreg4;


//        //    reglib::MassRegistrationPPR2 * bgmassreg3 = new reglib::MassRegistrationPPR2();
//        //    bgmassreg3->visualizationLvl = 1;
//        //    bgmassreg3->viewer = viewer;
//        //    bgmassreg3->convergence_mul = 0.1;
//        //    bgmassreg3->func_setup = 1;
////    for(unsigned int i = 0; i < keyframes.size(); i++){bgmassreg3->addModel(keyframes[i]);}
////    reglib::MassFusionResults bgmfr3 = bgmassreg3->getTransforms(keyframe_poses);
////    delete bgmassreg3;

//	saveModelToFB(mr3fr.poses,keyframe_time,output_path+"optimized_ppr3.txt");

//    reglib::Model * kf2 = new reglib::Model();
//	for(unsigned int k = 0; k < keyframes_frames.size(); k++){
//        int i = keyframe_ind[k];
//		reglib::RGBDFrame * frame = new reglib::RGBDFrame(camera,rgb_images[i],depth_images[i],timestamps[i], Eigen::Matrix4d::Identity(),true,"", true);
//		//reglib::RGBDFrame * frame = keyframes_frames[k];
//        kf2->last_changed = i;

//        std::vector< reglib::KeyPoint > kps = de->extract(frame);
//        for(unsigned int j = 0; j < kps.size(); j++){ kps[j].point.last_update_frame_id = i; }
//        current->frames[0] = frame;
//        current->keypoints = kps;
//        current->recomputeModelPoints();

//		kf2->mergeKeyPoints(current, mr3fr.poses[k]);
//		kf2->addSuperPoints(kf2->points,mr3fr.poses[k],frame,kf->modelmasks.back());
//    }

//	viewer->removeAllPointClouds();
//	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cld = kf2->getPCLcloud(1,3);
//	pcl::PointCloud<pcl::PointXYZRGB>::Ptr edge_cld = kf2->getPCLEdgeCloud(1,0);
//	viewer->addPointCloud<pcl::PointXYZRGB> (cld, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(cld), "cloud");
//	viewer->addPointCloud<pcl::PointXYZRGB> (edge_cld, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(edge_cld), "edge_cloud");
//	viewer->spinOnce();


//    reglib::MassRegistrationPPR3 * massreg2 = new reglib::MassRegistrationPPR3();
//    massreg2->viewer = viewer;
//    massreg2->visualizationLvl = 0;
//    massreg2->convergence_mul = 0.5;
//    massreg2->func_setup = 1;
//    massreg2->addModel(kf2);

//    printf("kf2: %i\n",kf2->points.size());


//    std::vector<Eigen::Matrix4d> opt_poses;
//    int current_kf = 0;
//	for(unsigned int i = 0; i < nr_frames; i++){
//        if(is_kf[i]){
//			opt_poses.push_back(mr3fr.poses[current_kf]);
//            current_kf++;

//			if(visualize){
//				pcl::PointCloud<pcl::PointXYZ>::Ptr    cloudCoord	(new pcl::PointCloud<pcl::PointXYZ>);
//				char buf [1024];
//				if(i > 0){
//					pcl::transformPointCloud (*coordcloud, *cloudCoord, Eigen::Affine3f(poses.back().cast<float>()));
//					pcl::PointXYZ p; p.x = opt_poses[i-1](0,3);p.y = opt_poses[i-1](1,3); p.z = opt_poses[i-1](2,3);
//					sprintf(buf,"prev_opt_%i",i);viewer->addLine<pcl::PointXYZ> (cloudCoord->points[0],p,0,255,0,buf);
//				}
//				viewer->spinOnce();
//			}

//            continue;
//        }
//        reglib::RGBDFrame * frame = new reglib::RGBDFrame(camera,rgb_images[i],depth_images[i],timestamps[i], Eigen::Matrix4d::Identity(),true,"", false);


//        std::vector< reglib::KeyPoint > kps = de->extract(frame);
//        for(unsigned int j = 0; j < kps.size(); j++){ kps[j].point.last_update_frame_id = i; }
//        current->frames[0] = frame;
//        current->keypoints = kps;
//        current->recomputeModelPoints();


//        massreg2->addModel(current);
//		cp.back() = opt_poses.back();

//        double regStart2 = quasimodo_brain::getTime();
//        reglib::MassFusionResults mfr = massreg2->getTransforms(cp);
//        massreg2->removeLastNode();

//		printf("opt %i / %i: mass registration time: %5.5fs score: %f\n",i+1,nr_frames,quasimodo_brain::getTime()-regStart2,mfr.score);

//        opt_poses.push_back(mfr.poses.back());

//		if(i % 10 == 0){
//			kf2->addSuperPoints(kf2->points,opt_poses.back(),frame,kf->modelmasks.back());
//			viewer->removeAllPointClouds();
//			pcl::PointCloud<pcl::PointXYZRGB>::Ptr cld = kf2->getPCLcloud(1,3);
//			pcl::PointCloud<pcl::PointXYZRGB>::Ptr edge_cld = kf2->getPCLEdgeCloud(1,0);
//			viewer->addPointCloud<pcl::PointXYZRGB> (cld, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(cld), "cloud");
//			viewer->addPointCloud<pcl::PointXYZRGB> (edge_cld, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(edge_cld), "edge_cloud");
//			viewer->spinOnce();
//		}

//		if(visualize){
//			pcl::PointCloud<pcl::PointXYZ>::Ptr    cloudCoord	(new pcl::PointCloud<pcl::PointXYZ>);
//			char buf [1024];
//			if(i > 0){
//				pcl::transformPointCloud (*coordcloud, *cloudCoord, Eigen::Affine3f(poses.back().cast<float>()));
//				pcl::PointXYZ p; p.x = opt_poses[i-1](0,3);p.y = opt_poses[i-1](1,3); p.z = opt_poses[i-1](2,3);
//				sprintf(buf,"prev_opt_%i",i);viewer->addLine<pcl::PointXYZ> (cloudCoord->points[0],p,255,0,255,buf);
//			}
//			viewer->spinOnce();
//		}
//    }


//    saveModelToFB(opt_poses,timestamps,output_path+"opt_poses.txt");

//	viewer->spin();

//    //Merge new map

//    //Relocalize frames to map
////exit(0);
////        std::vector<Eigen::Matrix4d> kfposes;
////    for(unsigned int i = 0; i < keyframes.size(); i++){
////        kfposes.push_back(Eigen::Matrix4d::Identity());
////    }
////    reglib::MassRegistrationPPR2 * bgmassreg = new reglib::MassRegistrationPPR2(0.00);
////    bgmassreg->timeout = 3600;
////    bgmassreg->viewer = viewer;
////    bgmassreg->use_surface = true;
////    bgmassreg->use_depthedge = false;
////    bgmassreg->visualizationLvl = 1;
////    bgmassreg->maskstep = 5;
////    bgmassreg->nomaskstep = 5;
////    bgmassreg->nomask = true;
////    bgmassreg->stopval = 0.0005;
////    for(unsigned int i = 0; i < keyframes.size(); i++){bgmassreg->addModel(keyframes[i]);}
////    reglib::MassFusionResults bgmfr = bgmassreg->getTransforms(keyframe_poses);
////    delete bgmassreg;
////    saveModelToFB(bgmfr.poses,keyframe_time,output_path+"optimized.txt");




////    reglib::MassRegistrationPPR2 * bgmassreg2 = new reglib::MassRegistrationPPR2(0.0);
////    bgmassreg2->timeout = 3600;
////    bgmassreg2->viewer = viewer;
////    bgmassreg2->use_surface = true;
////    bgmassreg2->use_depthedge = false;
////    bgmassreg2->visualizationLvl = 1;
////    bgmassreg2->maskstep = 2;
////    bgmassreg2->nomaskstep = 2;
////    bgmassreg2->nomask = true;
////    bgmassreg2->stopval = 0.0005;

////    for(unsigned int i = 0; i < keyframes.size(); i++){
////        bgmassreg2->addModel(keyframes[i]);
////    }
////    reglib::MassFusionResults bgmfr2 = bgmassreg2->getTransforms(bgmfr.poses);
////    delete bgmassreg2;

////    saveModelToFB(bgmfr2.poses,keyframe_time,output_path+"optimized2.txt");




	return poses;
}



std::vector<Eigen::Matrix4d> slam_vo3(reglib::DistanceWeightFunction2 * func, reglib::Registration * reg, reglib::DescriptorExtractor * de, string datapath = "./", string name = "testoutput.txt", string benchmarkpath = "./rgbd_benchmark_tools/scripts/", bool fuse_surface = true, bool fuse_kp = true, bool visualize = true, int norange = 0){
	//printf("slam_vo3\n");
	//saveModelToFB(gt_poses,timestamps,datapath,name+"_gtodometry.txt",benchmarkpath);
	unsigned int nr_frames = rgb_images_path.size();
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
	pcl::PointCloud<pcl::PointXYZ>::Ptr    coordcloud	(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr    gt_coordcloud	(new pcl::PointCloud<pcl::PointXYZ>);

    visualize = false;
	if(visualize){
		viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer ("viewer"));
		viewer->setBackgroundColor(1.0,1.0,1.0);
		viewer->removeAllPointClouds();
		viewer->removeAllShapes();

		coordcloud->points.resize(4);
		coordcloud->points[0].x = 0.00; coordcloud->points[0].y = 0.00; coordcloud->points[0].z = 0.00;
		coordcloud->points[1].x = 0.02; coordcloud->points[1].y = 0.00; coordcloud->points[1].z = 0.00;
		coordcloud->points[2].x = 0.00; coordcloud->points[2].y = 0.02; coordcloud->points[2].z = 0.00;
		coordcloud->points[3].x = 0.00; coordcloud->points[3].y = 0.00; coordcloud->points[3].z = 0.02;

		for(unsigned int i = 0; i < gt_poses.size(); i++){
			std::cout << gt_poses[i] << std::endl << std::endl;
			pcl::PointCloud<pcl::PointXYZ>::Ptr    cloudCoord	(new pcl::PointCloud<pcl::PointXYZ>);
			char buf [1024];
			pcl::transformPointCloud (*coordcloud, *cloudCoord, gt_poses[i]);
			if(i > 0){
				pcl::PointXYZ p; p.x = gt_poses[i-1](0,3);p.y = gt_poses[i-1](1,3); p.z = gt_poses[i-1](2,3);
				gt_coordcloud->points.push_back(p);
				sprintf(buf,"gt_%i",i);viewer->addLine<pcl::PointXYZ> (cloudCoord->points[0],p,0,0,255,buf);
			}
		}
		viewer->spinOnce();
	}

	std::vector<Eigen::Matrix4d> poses;

	cv::Mat fullmask;
	fullmask.create(480,640,CV_8UC1);
	unsigned char * maskdata = (unsigned char *)fullmask.data;
	for(unsigned int j = 0; j < 480*640; j++){maskdata[j] = 255;}

	reglib::Model * kf = new reglib::Model();
	kf->last_changed = 0;
	kf->relativeposes.push_back(Eigen::Matrix4d::Identity());
	kf->modelmasks.push_back(new reglib::ModelMask(fullmask));
	kf->frames.push_back(0);

	reglib::Model * current = new reglib::Model();
	current->relativeposes.push_back(Eigen::Matrix4d::Identity());
	current->modelmasks.push_back(new reglib::ModelMask(fullmask));
	current->frames.push_back(0);


	reglib::MassRegistrationPPR3 * massreg = new reglib::MassRegistrationPPR3(func);
	massreg->viewer = viewer;
    massreg->visualizationLvl = 2*int(visualize);
	massreg->convergence_mul = 1.0;
	massreg->func_setup = 0;
	massreg->tune_regularizer = true;
	//massreg->next_regularizer = 0.1;


	//register to last kf
	std::vector<Eigen::Matrix4d> cp;
	cp.push_back(Eigen::Matrix4d::Identity());
	cp.push_back(Eigen::Matrix4d::Identity());


	int kfstep = 1;
	//Merge into a new kf


	double max_error = 0;
	int max_ind = 0;
	int max_last_kf_nr = 0;
	Eigen::Matrix4d max_pose =  Eigen::Matrix4d::Identity();

	int last_kf_nr = 0;
	bool first = true;
	for(unsigned int i = 0; i < nr_frames; i++){

		//reglib::RGBDFrame * frame = new reglib::RGBDFrame(camera,rgb_images[i],depth_images[i],timestamps[i], Eigen::Matrix4d::Identity(),true,"", false);
		reglib::RGBDFrame * frame = new reglib::RGBDFrame(camera,cv::imread(rgb_images_path[i], CV_LOAD_IMAGE_COLOR),cv::imread(depth_images_path[i], CV_LOAD_IMAGE_UNCHANGED),timestamps[i], Eigen::Matrix4d::Identity(),true,"", false);
		frame->use_range_information = norange == 0;
		//        cv::Mat rgbimage    = cv::imread(rgbpath, CV_LOAD_IMAGE_COLOR);
		//        cv::Mat depthimage  = cv::imread(depthpath, CV_LOAD_IMAGE_UNCHANGED);
		//        rgb_images.push_back(rgbimage);
		//        depth_images.push_back(depthimage);
		current->frames[0] = frame;
		current->recomputeModelPoints();

		if(!first){//First frame is always a kf

			massreg->addModel(current);
			cp.back() = poses.back();

			double regStart2 = quasimodo_brain::getTime();
			reglib::MassFusionResults mfr = massreg->getTransforms(cp);
			massreg->removeLastNode();

			double regtime = quasimodo_brain::getTime()-regStart2;
			poses.push_back(mfr.poses.back());

			printf("frame %i / %i kf: %i registration time: %5.5fs score: %6.6f ",i,nr_frames,last_kf_nr,regtime,mfr.score);
//            if(mfr.score < 0.7){
//                massreg->visualizationLvl = 10;
//                massreg->addModel(current);
//                reglib::MassFusionResults mfr2 = massreg->getTransforms(cp);
//                massreg->removeLastNode();
//                massreg->visualizationLvl = 0;
//            }
			//fflush(stdout);
		}else{
			last_kf_nr = i;
			poses.push_back(Eigen::Matrix4d::Identity());
		}

		Eigen::Matrix4d currPose = poses.back();
		Eigen::Matrix4d gtcurrPose = gt_poses[i];

			//Eigen::Matrix4d regChange = currPose * poses[last_kf_nr].inverse();
		Eigen::Matrix4d regChange = poses.back() * cp.back().inverse();
		Eigen::Matrix4d gtChange = gtcurrPose * gt_poses[last_kf_nr].inverse();

		Eigen::Matrix4d errmat = regChange * gtChange.inverse();

		double dx = regChange(0,3)-gtChange(0,3);
		double dy = regChange(1,3)-gtChange(1,3);
		double dz = regChange(2,3)-gtChange(2,3);
		double regerror = sqrt(dx*dx+dy*dy+dz*dz);

		if(regerror > max_error){
			max_error = regerror;
			max_ind = i;
			max_last_kf_nr = last_kf_nr;
			max_pose = regChange;
		}
		printf("error: %5.5f ",regerror);
		printf("max_error = %5.5f ",max_error);
		printf("max_ind = %5.5i",max_ind);
		printf("\n");

		if( visualize && regerror < 0.02){

			viewer->removeAllShapes();
			massreg->visualizationLvl = 10;
			std::cout << "regChange\n" << regChange << std::endl << std::endl;
			std::cout << "gtChange\n" << gtChange << std::endl << std::endl;
			std::cout << "errmat\n" << errmat << std::endl << std::endl;


			massreg->addModel(current);
			reglib::MassFusionResults mfr = massreg->getTransforms(cp);
			massreg->removeLastNode();

			massreg->visualizationLvl = 0;
		}

		if(visualize){
			char buf [1024];
			pcl::PointXYZ currp; currp.x = currPose(0,3);currp.y = currPose(1,3); currp.z = currPose(2,3);
			pcl::PointXYZ gtp; gtp.x = gtcurrPose(0,3);gtp.y = gtcurrPose(1,3); gtp.z = gtcurrPose(2,3);
			sprintf(buf,"link_%i",i);viewer->addLine<pcl::PointXYZ> (gtp,currp,255,0,255,buf);
			viewer->spinOnce();
		}

		if(!first && visualize){
			pcl::PointCloud<pcl::PointXYZ>::Ptr    cloudCoord	(new pcl::PointCloud<pcl::PointXYZ>);
			char buf [1024];
			pcl::transformPointCloud (*coordcloud, *cloudCoord, Eigen::Affine3f(poses.back().cast<float>()));
			pcl::PointXYZ p; p.x = poses[i-1](0,3);p.y = poses[i-1](1,3); p.z = poses[i-1](2,3);
            sprintf(buf,"prev_%i",i);viewer->addLine<pcl::PointXYZ> (cloudCoord->points[0],p,0,255,0,buf);
			viewer->spinOnce();
		}

		if(i % kfstep == 0){
			last_kf_nr = i;
			kf->points.clear();
			kf->addSuperPoints( kf->points,poses.back(),frame,kf->modelmasks.back());

//            int nr_kfp = kf->points.size();
//            for(int k = 0; k < nr_kfp; k++){
//                if( kf->last_changed - kf->points[k].last_update_frame_id > 20){
//                    kf->points[k] = kf->points.back();
//                    kf->points.pop_back();
//                    k--;
//                    nr_kfp--;
//                }
//            }

			if(!first){massreg->removeLastNode();}
            massreg->addModel(kf);
		}

		delete frame;
		if(first){first = false;}
	}

	if(visualize){viewer->spinOnce();}

	printf("\r                                                                                                                                                       \r");
	fflush(stdout);
	saveModelToFB(poses,timestamps,datapath,func->name+"_odometry.txt",benchmarkpath);

	printf("max_error = %f\n",max_error);
	printf("max_ind = %i\n",max_ind);
	std::cout << max_pose << std::endl << std::endl;

	delete kf;
	delete current;
	delete massreg;


	return poses;
}

int main(int argc, char **argv){

	char buf [1024];
	std::string benchmarkpath = "./rgbd_benchmark_tools/scripts/";

	vector< reglib::DistanceWeightFunction2 * > funcs;

//	{
//		reglib::GeneralizedGaussianDistribution * gd = new reglib::GeneralizedGaussianDistribution(true,false,false);
//		gd->nr_refineiters = 6;
//		gd->costpen = 10;
//		gd->ratio_costpen = 0;
//		gd->debugg_print = false;
//		reglib::DistanceWeightFunction2PPR3 * sfunc = new reglib::DistanceWeightFunction2PPR3(gd);
//		sfunc->noise_min                            = 0.0005;
//		sfunc->startreg                             = 0.01;
//		sfunc->blur                                 = 0.02;//0.02;//0.03;
//		sfunc->data_per_bin                         = 40;
//		sfunc->debugg_print                         = false;
//		sfunc->threshold                            = false;//(setup & 0b1) > 0;
//		sfunc->useIRLSreweight                      = false;
//		sfunc->reg_shrinkage                        = 0.8;
//		sfunc->max_under_mean                       = false;
//		if(sfunc->useIRLSreweight){
//			sprintf(buf,"ppr_%i_%i_%i_%i_ggd",int(100.0*sfunc->startreg),int(100.0*gd->costpen),int(sfunc->data_per_bin),int(1000.0*sfunc->blur));
//		}else{
//			sprintf(buf,"ppr_%i_%i_%i_%i_gausian",int(100.0*sfunc->startreg),int(100.0*gd->costpen),int(sfunc->data_per_bin),int(1000.0*sfunc->blur));
//		}
//		sfunc->name = string(buf);
//		funcs.push_back(sfunc);
//	}
    {
        reglib::GeneralizedGaussianDistribution * gd = new reglib::GeneralizedGaussianDistribution(true,false,false);
        gd->nr_refineiters = 1;
        gd->costpen = 10;
        gd->ratio_costpen = 0;
        gd->debugg_print = false;
        reglib::DistanceWeightFunction2PPR3 * sfunc = new reglib::DistanceWeightFunction2PPR3(gd);
        sfunc->startreg                             = 1;
        sfunc->noise_min                            = 0.0005;
        sfunc->blur                                 = 0.02;//0.03;
        sfunc->data_per_bin                         = 40;
        sfunc->debugg_print                         = false;
        sfunc->max_under_mean                       = false;
        sfunc->reg_shrinkage                        = 0.5;
        sfunc->useIRLSreweight                      = false;
        sfunc->name = "ppr";
        funcs.push_back(sfunc);
    }


	{
		reglib::GeneralizedGaussianDistribution * gd = new reglib::GeneralizedGaussianDistribution(true,true,false);
		gd->nr_refineiters = 6;
		gd->costpen = 10;
		gd->ratio_costpen = 0;
		gd->debugg_print = false;
		reglib::DistanceWeightFunction2PPR3 * sfunc = new reglib::DistanceWeightFunction2PPR3(gd);
		sfunc->noise_min                            = 0.0005;
		sfunc->startreg                             = 0.01;
		sfunc->blur                                 = 0.02;//0.02;//0.03;
		sfunc->data_per_bin                         = 40;
		sfunc->debugg_print                         = false;
		sfunc->threshold                            = false;//(setup & 0b1) > 0;
		sfunc->useIRLSreweight                      = true;
		sfunc->reg_shrinkage                        = 0.8;
		sfunc->max_under_mean                       = false;
		if(sfunc->useIRLSreweight){
			sprintf(buf,"ppr_%i_%i_%i_%i_ggd",int(100.0*sfunc->startreg),int(100.0*gd->costpen),int(sfunc->data_per_bin),int(1000.0*sfunc->blur));
		}else{
			sprintf(buf,"ppr_%i_%i_%i_%i_gausian",int(100.0*sfunc->startreg),int(100.0*gd->costpen),int(sfunc->data_per_bin),int(1000.0*sfunc->blur));
		}
		sfunc->name = string(buf);
		funcs.push_back(sfunc);
	}




	double noise = 0.002;

	for(double mul = 3.5; mul <= 3.5; mul += 0.5){
		reglib::DistanceWeightFunction2 * tfunc = new reglib::DistanceWeightFunction2();
		tfunc->f                     = reglib::THRESHOLD;
		tfunc->p                     = mul*noise;
		sprintf(buf,"norange_threshold_%i",int(100.0*mul));
		tfunc->name = string(buf);
		funcs.push_back(tfunc);
	}

	{
		reglib::DistanceWeightFunction2 * tfunc = new reglib::DistanceWeightFunction2();
		tfunc->f                        = reglib::PNORM;
		tfunc->p                        = 0.1;
		tfunc->name                     = "pnorm01";
		funcs.push_back(tfunc);
	}

	{
		reglib::DistanceWeightFunction2 * tfunc = new reglib::DistanceWeightFunction2();
		tfunc->f                     = reglib::PNORM;
		tfunc->p                     = 1.0;
		//tfunc->debugg_print                         = true;false
		tfunc->name                     = "norange_pnorm10";
		funcs.push_back(tfunc);
	}

	{
		reglib::DistanceWeightFunction2Tdist * tfunc = new reglib::DistanceWeightFunction2Tdist();
		tfunc->name                     = "norange_Tdist";
		funcs.push_back(tfunc);
	}

	for(unsigned i = 0 ; i < funcs.size(); i++){
		funcs[i]->name += "_framestep_"+string(argv[2]);
	}

	camera				= new reglib::Camera();

//    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer ("viewer"));
//    viewer->setBackgroundColor(1.0,0.0,1.0);
//    viewer->addCoordinateSystem(0.1);

	std::vector<reglib::Registration *> regs;


	reglib::DistanceWeightFunction2PPR3 * func  = new reglib::DistanceWeightFunction2PPR3(new reglib::GaussianDistribution());
	func->debugg_print                          = false;
	func->noise_min                             = 0.0001;
	func->startreg                              = 0.01;
	func->reg_shrinkage                         = 0.5;

//    reglib::DistanceWeightFunction2 * func  = new reglib::DistanceWeightFunction2();
//    func->p = 0.1;


//    reglib::DistanceWeightFunction2 * func  = new reglib::DistanceWeightFunction2();
//    func->f = reglib::Function::THRESHOLD;
//    func->p = 0.025;


	reglib::RegistrationRefinement3 * rrf = new reglib::RegistrationRefinement3(func);
	rrf->target_points = 1000;
	rrf->dst_points = 20000000;
	//rrf->viewer = viewer;
	rrf->visualizationLvl = 0;
	rrf->regularization = 0.001;
	rrf->convergence = 0.05;
	rrf->normalize_matchweights	= true;
	rrf->useKeyPoints = false;
	rrf->useSurfacePoints = true;
	regs.push_back(rrf);

//    reglib::RegistrationRefinement * rrf = new reglib::RegistrationRefinement();
//    rrf->target_points = 5000;
//    rrf->dst_points = 20000000;
//    rrf->viewer = viewer;
//    rrf->visualizationLvl = 0;
//    rrf->regularization = 0.01;
////    rrf->convergence = 0.2;
//    rrf->normalize_matchweights	= true;
////    rrf->useKeyPoints = true;
////    rrf->useSurfacePoints = true;
//    regs.push_back(rrf);

	std::vector<reglib::DescriptorExtractor *> des;
	des.push_back(new reglib::DescriptorExtractorSURF());
	des.back()->setDebugg(0);

	int funcnr = atoi(argv[1]);

	int framestep = atoi(argv[2]);

	int startind = atoi(argv[3]);
	int stopind = atoi(argv[4]);
	int norange = atoi(argv[5]);

	if(norange == 1){
		funcs[funcnr]->name = "norange_"+funcs[funcnr]->name;
	}

	printf("%i %i %i %i %i\n",funcnr,framestep,startind,stopind,norange);

	scriptPath = funcs[funcnr]->name+"_scroring.sh";

	printf("scriptPath: %s\n",scriptPath.c_str());

	std::ofstream ofs;
	ofs.open(scriptPath, std::ofstream::out | std::ofstream::trunc);
	ofs.close();

	for(unsigned int arg = 6; arg < argc; arg++){
		std::string datapath = std::string(argv[arg]);
		addFBdata2(argv[arg], startind, stopind, framestep);

		for(unsigned int i = 0; i < regs.size(); i++){
			for(unsigned int j = 0; j < des.size(); j++){
				//std::vector<Eigen::Matrix4d> vo = slam_vo2(regs[i],des[j],std::string(argv[arg])+"/"+regs[i]->getString()+"_",true,true);//rrf->useSurfacePoints,rrf->useKeyPoints)
                std::vector<Eigen::Matrix4d> vo = slam_vo3(funcs[funcnr],regs[i],des[j],datapath,regs[i]->getString(),benchmarkpath,true,true,true,norange);//rrf->useSurfacePoints,rrf->useKeyPoints)
			}
		}
		rgb_images.clear();
		depth_images.clear();
		timestamps.clear();
	}

	return 0;
}
