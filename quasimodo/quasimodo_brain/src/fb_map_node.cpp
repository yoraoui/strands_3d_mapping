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

#include "ModelDatabase/ModelDatabase.h"
#include "ModelStorage/ModelStorage.h"
#include "Util/Util.h"
#include "CameraOptimizer/CameraOptimizer.h"

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
            if(diff_current <= gtdiff_best){
                gtdiff_best = diff_current;
                gt_counter = current_counter;
                gt_ts = dts;
            }else{break;}

        }

        rgb_indexes.push_back(rgb_counter);
        depth_indexes.push_back(depth_counter);
        gt_indexes.push_back(gt_counter);
    }

    for(unsigned int i = start; i < depth_indexes.size() && i < stop; i+= skip){
        string rgbpath = input+"/"+rgb_lines.at(rgb_indexes.at(i)).second;
        string depthpath = input+"/"+depth_lines.at(depth_indexes.at(i)).second;

        rgb_images_path.push_back(rgbpath);
        depth_images_path.push_back(depthpath);

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
}

void saveModelToFB(std::vector<Eigen::Matrix4d> poses, std::vector<double> timestamps, string datapath = "./", string name = "testoutput.txt", string benchmarkpath = "./rgbd_benchmark_tools/scripts/"){
    string pathname = datapath+"/"+name;
    string figurename = name;
    figurename.pop_back();
    figurename.pop_back();
    figurename.pop_back();
    figurename.pop_back();
    //printf("Saving map in: %s\n",pathname.c_str());

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
    outfile << std::string(buf);
    printf("%s\n",buf);

    sprintf(buf,"%sevaluate_ate.py %s/groundtruth.txt %s --plot %s/%s_figure_ate.png --verbose | grep absolute_translational_error.rmse\n",benchmarkpath.c_str(),datapath.c_str(),pathname.c_str(),datapath.c_str(),figurename.c_str());
    outfile << std::string(buf);
    printf("%s\n",buf);
}

std::vector<Eigen::Matrix4d> slam_vo3(reglib::DistanceWeightFunction2 * func, reglib::DescriptorExtractor * de, string datapath = "./", string benchmarkpath = "./rgbd_benchmark_tools/scripts/", bool visualize = true){
    CameraOptimizer * co = CameraOptimizer::load("current_CameraOptimizerGridXYZ.bin");
    //co->show(true);

    unsigned int nr_frames = rgb_images_path.size();
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
    pcl::PointCloud<pcl::PointXYZ>::Ptr    coordcloud	(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr    gt_coordcloud	(new pcl::PointCloud<pcl::PointXYZ>);

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
            //std::cout << gt_poses[i] << std::endl << std::endl;
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


    reglib::Model * total_model = new reglib::Model();

    reglib::MassRegistrationPPR3 * massreg = new reglib::MassRegistrationPPR3(func);
    massreg->viewer                 = viewer;
    massreg->visualizationLvl       = 0;//10*int(visualize);
    massreg->refine_translation     = true;
    massreg->refine_rotation        = true;
    massreg->convergence_mul        = 1.0;
    massreg->func_setup             = 0;
    massreg->tune_regularizer       = true;
    //massreg->next_regularizer = 0.1;

    //register to last kf
    std::vector<Eigen::Matrix4d> cp;
    cp.push_back(Eigen::Matrix4d::Identity());
    cp.push_back(Eigen::Matrix4d::Identity());

    int kfstep = 10;

    double max_error = 0;
    int max_ind = 0;
    int max_last_kf_nr = 0;
    Eigen::Matrix4d max_pose =  Eigen::Matrix4d::Identity();

    int last_kf_nr = 0;
    bool first = true;
    for(unsigned int i = 0; i < nr_frames; i++){
        reglib::RGBDFrame * frame = new reglib::RGBDFrame(camera,cv::imread(rgb_images_path[i], CV_LOAD_IMAGE_COLOR), co->improveDepth(cv::imread(depth_images_path[i], CV_LOAD_IMAGE_UNCHANGED),camera->idepth_scale),timestamps[i], Eigen::Matrix4d::Identity(),true,"", false);
        current->frames[0] = frame;
        current->recomputeModelPoints();

        bool is_kf = i % kfstep == 0;

        if(!first){//First frame is always a kf
            massreg->addModel(current,1000);
            cp.back() = poses.back();
            //cp.back() = gt_poses[i];
//gt_poses[i];//poses.back();

            double regStart2 = quasimodo_brain::getTime();
            reglib::MassFusionResults mfr = massreg->getTransforms(cp);
            massreg->removeLastNode();

            double regtime = quasimodo_brain::getTime()-regStart2;
            poses.push_back(mfr.poses.back());

            printf("frame %i / %i kf: %i registration time: %5.5fs score: %6.6f ",i,nr_frames,last_kf_nr,regtime,mfr.score);
            if(mfr.score < 0.75){is_kf = true;}
        }else{
            last_kf_nr = i;
            poses.push_back(Eigen::Matrix4d::Identity());
        }

        Eigen::Matrix4d currPose = poses.back();
        Eigen::Matrix4d gtcurrPose = gt_poses[i];

        //Eigen::Matrix4d regChange = currPose * poses[last_kf_nr].inverse();
        Eigen::Matrix4d regChange = poses.back() * cp.back().inverse();
        Eigen::Matrix4d gtChange = gtcurrPose * gt_poses[last_kf_nr].inverse();

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
        printf("error: %5.5f max_error = %5.5f max_ind = %5.5i\n",regerror,max_error,max_ind);


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

		if(is_kf){
			//reglib::RGBDFrame * last_kf_frame = new reglib::RGBDFrame(camera,cv::imread(rgb_images_path[last_kf_nr], CV_LOAD_IMAGE_COLOR),cv::imread(depth_images_path[last_kf_nr], CV_LOAD_IMAGE_UNCHANGED),timestamps[last_kf_nr], Eigen::Matrix4d::Identity(),true,"", false);
            //reglib::RGBDFrame * kf_frame = new reglib::RGBDFrame(camera,cv::imread(rgb_images_path[i], CV_LOAD_IMAGE_COLOR),co->improveDepth( cv::imread(depth_images_path[i], CV_LOAD_IMAGE_UNCHANGED),camera->idepth_scale),timestamps[i], Eigen::Matrix4d::Identity(),true,"", true);
            reglib::RGBDFrame * kf_frame = new reglib::RGBDFrame(camera,cv::imread(rgb_images_path[i], CV_LOAD_IMAGE_COLOR),cv::imread(depth_images_path[i], CV_LOAD_IMAGE_UNCHANGED),timestamps[i], Eigen::Matrix4d::Identity(),true,"", true);

            last_kf_nr = i;
            kf->points.clear();
            kf->addSuperPoints( kf->points,poses.back(),frame,kf->modelmasks.back());

            if(!first){massreg->removeLastNode();}
            massreg->addModel(kf,1000);
			kf_frame->keyval = "xyz_"+std::to_string(i);
            total_model->relativeposes.push_back(poses.back());
            total_model->modelmasks.push_back(new reglib::ModelMask(fullmask));
            total_model->frames.push_back(kf_frame);

//			if(false && visualize){
//				viewer->removeAllPointClouds();
//				for(unsigned int k = 0; k < src_cld->points.size(); k++){
//					src_cld->points[k].r = 255;
//					src_cld->points[k].g = 0;
//					src_cld->points[k].b = 0;
//				}
//				for(unsigned int k = 0; k < dst_cld->points.size(); k++){
//					dst_cld->points[k].r = 0;
//					dst_cld->points[k].g = 255;
//					dst_cld->points[k].b = 0;
//				}
//				Eigen::Matrix4d m3 = total_model->relativeposes[fnrj].inverse() * total_model->relativeposes[fnr];
//				pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZRGB> ());
//				pcl::transformPointCloud (*src_cld, *transformed_cloud, m3);
//				viewer->addPointCloud<pcl::PointXYZRGB> (transformed_cloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(transformed_cloud), "src_cloud");
//				viewer->addPointCloud<pcl::PointXYZRGB> (dst_cld, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(dst_cld), "dst_cloud");                    viewer->spin();
//				viewer->spin();
//				viewer->removeAllPointClouds();
//			}

            if(visualize){
                viewer->removeAllPointClouds();
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr cld = kf->getPCLcloud(1,3);
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr edge_cld = kf->getPCLEdgeCloud(1,0);
                viewer->addPointCloud<pcl::PointXYZRGB> (cld, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(cld), "cloud");
                viewer->addPointCloud<pcl::PointXYZRGB> (edge_cld, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(edge_cld), "edge_cloud");
                viewer->addCoordinateSystem(0.1,Eigen::Affine3f(poses.back().cast<float>()));
                viewer->spinOnce();
            }
        }

        delete frame;
        if(first){first = false;}
    }

    if(visualize){viewer->spinOnce();}

    fflush(stdout);
    saveModelToFB(poses,timestamps,datapath,func->name+"_odometry.txt",benchmarkpath);

//    printf("max_error = %f\n",max_error);
//    printf("max_ind = %i\n",max_ind);

    ModelStorageFile * storage = new ModelStorageFile("fb_model/");
    storage->add(total_model, "xyz");

    delete kf;
    delete current;
    delete massreg;
    return poses;
}

int main(int argc, char **argv){
    char buf [1024];
    std::string benchmarkpath = "./rgbd_benchmark_tools/scripts/";

    vector< reglib::DistanceWeightFunction2 * > funcs;

    {
        reglib::GeneralizedGaussianDistribution * gd = new reglib::GeneralizedGaussianDistribution(true,false,false);
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
        sfunc->useIRLSreweight                      = false;
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

    for(unsigned i = 0 ; i < funcs.size(); i++){
        funcs[i]->name += "_framestep_"+string(argv[2]);
    }

    camera				= new reglib::Camera();

    std::vector<reglib::Registration *> regs;

    std::vector<reglib::DescriptorExtractor *> des;
    des.push_back(new reglib::DescriptorExtractorSURF());
    des.back()->setDebugg(0);

    int framestep = atoi(argv[1]);

    int startind = atoi(argv[2]);
    int stopind = atoi(argv[3]);

    int funcnr = 0;
    scriptPath = funcs[funcnr]->name+"_scroring.sh";

    std::ofstream ofs;
    ofs.open(scriptPath, std::ofstream::out | std::ofstream::trunc);
    ofs.close();

    for(unsigned int arg = 4; arg < argc; arg++){
        std::string datapath = std::string(argv[arg]);
        addFBdata2(argv[arg], startind, stopind, framestep);


        for(unsigned int j = 0; j < des.size(); j++){
//            camera->fx = 525;//541;
//            camera->cx = 319.5;//291;
//            camera->fy = 525;//513;
//            camera->cy = 239.5;//255;
            //double fx_start = camera->fx-5;
            //double fx_stop = camera->fx+5;
            //std::string name = funcs[funcnr]->name;
            //for(camera->cy = cy_start; camera->cy <= cy_stop; camera->cy += 1.0){
            //    sprintf(buf,"%s_camera_%i_%i_%i_%i",name.c_str(),int(10.0*camera->fx),int(10.0*camera->fy),int(10.0*camera->cx),int(10.0*camera->cy));
            //    funcs[funcnr]->name = std::string(buf);
                std::vector<Eigen::Matrix4d> vo = slam_vo3(funcs[funcnr],des[j],datapath,benchmarkpath,true);
            //}
        }

        rgb_images.clear();
        depth_images.clear();
        timestamps.clear();
    }

    return 0;
}
