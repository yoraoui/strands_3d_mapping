#include "core/RGBDFrame.h"

#include <pcl/console/parse.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/segmentation/supervoxel_clustering.h>

#include <vtkPolyLine.h>

#include <iostream>
#include <vector>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/region_growing.h>

#include <vector>
#include <string>

#include <cv.h>
#include <highgui.h>
#include <fstream>

#include <ctime>

#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/console/time.h>
#include "organized_edge_detection.hpp"
#include <pcl/features/integral_image_normal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/PCLPointCloud2.h>


namespace reglib
{
unsigned long RGBDFrame_id_counter;
RGBDFrame::RGBDFrame(){
	id = RGBDFrame_id_counter++;
	capturetime = 0;
	pose = Eigen::Matrix4d::Identity();
	keyval = "";
}

bool updated = true;
void on_trackbar( int, void* ){updated = true;}

RGBDFrame::RGBDFrame(Camera * camera_, cv::Mat rgb_, cv::Mat depth_, double capturetime_, Eigen::Matrix4d pose_, bool compute_normals){
	keyval = "";

    sweepid = -1;
	id = RGBDFrame_id_counter++;
	camera = camera_;
	rgb = rgb_;
	depth = depth_;
	capturetime = capturetime_;
	pose = pose_;

	IplImage iplimg = rgb_;
	IplImage* img = &iplimg;

	//printf("%s LINE:%i\n",__FILE__,__LINE__);

    int width = img->width;
    int height = img->height;
    int sz = height*width;
	const double idepth			= camera->idepth_scale;
	const double cx				= camera->cx;
	const double cy				= camera->cy;
	const double ifx			= 1.0/camera->fx;
	const double ify			= 1.0/camera->fy;

	//
//	printf("camera: cx %5.5f cy %5.5f",cx,cy);
//	printf(": fx %5.5f fy %5.5f",camera->fx,camera->fy);
//	printf(": ifx %5.5f ify %5.5f\n",ifx,ify);

	connections.resize(1);
	connections[0].resize(1);
	connections[0][0] = 0;

	intersections.resize(1);
	intersections[0].resize(1);
	intersections[0][0] = 0;

	nr_labels = 1;
	labels = new int[width*height];
    for(int i = 0; i < width*height; i++){labels[i] = 0;}


	//printf("%s LINE:%i\n",__FILE__,__LINE__);
	unsigned short * depthdata = (unsigned short *)depth.data;
	unsigned char * rgbdata = (unsigned char *)rgb.data;

	depthedges.create(height,width,CV_8UC1);
	unsigned char * depthedgesdata = (unsigned char *)depthedges.data;
	for(int i = 0; i < width*height; i++){
		depthedgesdata[i] = 0;
	}
/*
	double t = 0.01;
    for(int w = 0; w < width; w++){
        for(int h = 0; h < height;h++){
			int ind = h*width+w;
			depthedgesdata[ind] = 0;
			double z = idepth*double(depthdata[ind]);
			if(w > 0){
				double z2 = idepth*double(depthdata[ind-1]);
				double info = 1.0/(z*z+z2*z2);
				double diff = fabs(z2-z)*info;
				if(diff > t){depthedgesdata[ind] = 255;}
			}

			if(w < width-1){
				double z2 = idepth*double(depthdata[ind+1]);
				double info = 1.0/(z*z+z2*z2);
				double diff = fabs(z2-z)*info;
				if(diff > t){depthedgesdata[ind] = 255;}
			}

			if(h > 0){
				double z2 = idepth*double(depthdata[ind-width]);
				double info = 1.0/(z*z+z2*z2);
				double diff = fabs(z2-z)*info;
				if(diff > t){depthedgesdata[ind] = 255;}
			}

			if(h < height-1){
				double z2 = idepth*double(depthdata[ind+width]);
				double info = 1.0/(z*z+z2*z2);
				double diff = fabs(z2-z)*info;
				if(diff > t){depthedgesdata[ind] = 255;}
			}

			if(h > 0 && w > 0){
				double z2 = idepth*double(depthdata[ind-width-1]);
				double info = 1.0/(z*z+z2*z2);
				double diff = fabs(z2-z)*info;
				if(diff > t){depthedgesdata[ind] = 255;}
			}

			if(w > 0 && h < height-1){
				double z2 = idepth*double(depthdata[ind+width-1]);
				double info = 1.0/(z*z+z2*z2);
				double diff = fabs(z2-z)*info;
				if(diff > t){depthedgesdata[ind] = 255;}
			}

			if(h > 0 && w < width-1){
				double z2 = idepth*double(depthdata[ind-width+1]);
				double info = 1.0/(z*z+z2*z2);
				double diff = fabs(z2-z)*info;
				if(diff > t){depthedgesdata[ind] = 255;}
			}

			if(h < height-1 && w < width-1){
				double z2 = idepth*double(depthdata[ind+width+1]);
				double info = 1.0/(z*z+z2*z2);
				double diff = fabs(z2-z)*info;
				if(diff > t){depthedgesdata[ind] = 255;}
			}
		}
	}
*/


	//printf("%s LINE:%i\n",__FILE__,__LINE__);
	if(compute_normals){
		normals.create(height,width,CV_32FC3);
		float * normalsdata = (float *)normals.data;

	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr	cloud	(new pcl::PointCloud<pcl::PointXYZRGBA>);
		pcl::PointCloud<pcl::Normal>::Ptr	normals_cloud (new pcl::PointCloud<pcl::Normal>);
		cloud->width	= width;
		cloud->height	= height;
		cloud->points.resize(width*height);
		//cloud->dense = false;

        for(int w = 0; w < width; w++){
            for(int h = 0; h < height;h++){
				int ind = h*width+w;
				pcl::PointXYZRGBA & p = cloud->points[ind];
				p.b = rgbdata[3*ind+0];
				p.g = rgbdata[3*ind+1];
				p.r = rgbdata[3*ind+2];
				double z = idepth*double(depthdata[ind]);
				if(z > 0){
					p.x = (double(w) - cx) * z * ifx;
					p.y = (double(h) - cy) * z * ify;
					p.z = z;
				}else{
					p.x = NAN;
					p.y = NAN;
					p.z = NAN;
				}
			}
		}

		//printf("%s LINE:%i\n",__FILE__,__LINE__);
		pcl::IntegralImageNormalEstimation<pcl::PointXYZRGBA, pcl::Normal> ne;
		ne.setInputCloud(cloud);

		bool tune = false;
		unsigned char * combidata;
		cv::Mat combined;

		int NormalEstimationMethod = 0;
		int MaxDepthChangeFactor = 20;
		int NormalSmoothingSize = 7;
		int depth_dependent_smoothing = 1;

		ne.setMaxDepthChangeFactor(0.001*double(MaxDepthChangeFactor));
		ne.setNormalSmoothingSize(NormalSmoothingSize);
		ne.setDepthDependentSmoothing (depth_dependent_smoothing);
		ne.compute(*normals_cloud);

        for(int w = 0; w < width; w++){
            for(int h = 0; h < height;h++){
				int ind = h*width+w;
				pcl::PointXYZRGBA p = cloud->points[ind];
				pcl::Normal		p2		= normals_cloud->points[ind];


				//if(w % 20 == 0 && h % 20 == 0){printf("%i %i -> point(%f %f %f) normal(%f %f %f)\n",w,h,p.x,p.y,p.z,p2.normal_x,p2.normal_y,p2.normal_z);}

				if(!isnan(p2.normal_x)){
					normalsdata[3*ind+0]	= p2.normal_x;
					normalsdata[3*ind+1]	= p2.normal_y;
					normalsdata[3*ind+2]	= p2.normal_z;
				}else{
					normalsdata[3*ind+0]	= 2;
					normalsdata[3*ind+1]	= 2;
					normalsdata[3*ind+2]	= 2;
				}
			}
		}

//		cv::Mat curvature;
//		curvature.create(height,width,CV_8UC3);
//		unsigned char * curvaturedata = (unsigned char *)curvature.data;
//		for(int w = 0; w < width; w++){
//			for(int h = 0; h < height;h++){
//				int ind = h*width+w;
//				pcl::Normal		p2		= normals_cloud->points[ind];
//				//if(w % 5 == 0 && h % 5 == 0){printf("%i %i -> curvature %f\n",w,h,p2.curvature);}
//				curvaturedata[3*ind+0]	= 255.0*p2.curvature;
//				curvaturedata[3*ind+1]	= 255.0*p2.curvature;
//				curvaturedata[3*ind+2]	= 255.0*p2.curvature;
//			}
//		}
//		cv::namedWindow( "curvature", cv::WINDOW_AUTOSIZE );
//		cv::imshow( "curvature", curvature );
//		cv::waitKey(0);



		//printf("%s LINE:%i\n",__FILE__,__LINE__);
		if(tune){
			combined.create(height,2*width,CV_8UC3);
			combidata = (unsigned char *)combined.data;

			cv::namedWindow( "normals", cv::WINDOW_AUTOSIZE );
			cv::namedWindow( "combined", cv::WINDOW_AUTOSIZE );

			//cv::createTrackbar( "NormalEstimationMethod", "combined", &NormalEstimationMethod, 3, on_trackbar );
			//cv::createTrackbar( "MaxDepthChangeFactor", "combined", &MaxDepthChangeFactor, 1000, on_trackbar );
			//cv::createTrackbar( "NormalSmoothingSize", "combined", &NormalSmoothingSize, 100, on_trackbar );
			//cv::createTrackbar( "depth_dependent_smoothing", "combined", &depth_dependent_smoothing, 1, on_trackbar );

			//while(true){

				if(NormalEstimationMethod == 0){ne.setNormalEstimationMethod (ne.COVARIANCE_MATRIX);}
				if(NormalEstimationMethod == 1){ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT);}
				if(NormalEstimationMethod == 2){ne.setNormalEstimationMethod (ne.AVERAGE_DEPTH_CHANGE);}
				if(NormalEstimationMethod == 3){ne.setNormalEstimationMethod (ne.SIMPLE_3D_GRADIENT);}

				ne.setMaxDepthChangeFactor(0.001*double(MaxDepthChangeFactor));
				ne.setNormalSmoothingSize(NormalSmoothingSize);
				ne.setDepthDependentSmoothing (depth_dependent_smoothing);
				ne.compute(*normals_cloud);
                for(int w = 0; w < width; w++){
                    for(int h = 0; h < height;h++){
						int ind = h*width+w;
						pcl::Normal		p2		= normals_cloud->points[ind];
						if(!isnan(p2.normal_x)){
							normalsdata[3*ind+0]	= p2.normal_x;
							normalsdata[3*ind+1]	= p2.normal_y;
							normalsdata[3*ind+2]	= p2.normal_z;
						}else{
							normalsdata[3*ind+0]	= 2;
							normalsdata[3*ind+1]	= 2;
							normalsdata[3*ind+2]	= 2;
						}
					}
				}


                for(int w = 0; w < width; w++){
                    for(int h = 0; h < height;h++){
						int ind = h*width+w;
						int indn = h*2*width+(w+width);
						int indc = h*2*width+(w);
						combidata[3*indc+0]	= rgbdata[3*ind+0];
						combidata[3*indc+1]	= rgbdata[3*ind+1];
						combidata[3*indc+2]	= rgbdata[3*ind+2];
						pcl::Normal		p2		= normals_cloud->points[ind];
						if(!isnan(p2.normal_x)){
							combidata[3*indn+0]	= 255.0*fabs(p2.normal_x);
							combidata[3*indn+1]	= 255.0*fabs(p2.normal_y);
							combidata[3*indn+2]	= 255.0*fabs(p2.normal_z);
						}else{
							combidata[3*indn+0]	= 255;
							combidata[3*indn+1]	= 255;
							combidata[3*indn+2]	= 255;
						}
					}
				}
				char buf [1024];
                sprintf(buf,"combined%i.png",int(id));
				cv::imwrite( buf, combined );
				printf("saving: %s\n",buf);
				cv::imshow( "combined", combined );
				cv::waitKey(0);
				//while(!updated){cv::waitKey(50);}
				updated = false;
			//}
		}

		//pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2 = getPCLcloud();
		pcl::OrganizedEdgeFromRGBNormals<pcl::PointXYZRGBA, pcl::Normal, pcl::Label> oed;
		oed.setInputNormals (normals_cloud);
		oed.setInputCloud (cloud);
		oed.setDepthDisconThreshold (0.02); // 2cm
//		oed.setHCCannyLowThreshold (0.4);
//		oed.setHCCannyHighThreshold (1.1);
		oed.setMaxSearchNeighbors (50);
		pcl::PointCloud<pcl::Label> labels;
		std::vector<pcl::PointIndices> label_indices;
		oed.compute (labels, label_indices);

//		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr occluding_edges (new pcl::PointCloud<pcl::PointXYZRGBNormal>),
//				occluded_edges (new pcl::PointCloud<pcl::PointXYZRGBNormal>),
//				boundary_edges (new pcl::PointCloud<pcl::PointXYZRGBNormal>),
//				high_curvature_edges (new pcl::PointCloud<pcl::PointXYZRGBNormal>),
//				rgb_edges (new pcl::PointCloud<pcl::PointXYZRGBNormal>);

//		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud3 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
//		pcl::copyPointCloud (*cloud2, label_indices[0].indices, *boundary_edges);
//		pcl::copyPointCloud (*cloud2, label_indices[1].indices, *occluding_edges);
//		pcl::copyPointCloud (*cloud2, label_indices[2].indices, *occluded_edges);
//		pcl::copyPointCloud (*cloud2, label_indices[3].indices, *high_curvature_edges);
//		pcl::copyPointCloud (*cloud2, label_indices[4].indices, *rgb_edges);

//		cv::Mat out = rgb.clone();

		for(unsigned int i = 0; i < label_indices[1].indices.size(); i++){
			int ind = label_indices[1].indices[i];
			//depthedgesdata[ind] = 1;
//			out.data[3*ind+0] =255;
//			out.data[3*ind+1] =0;
//			out.data[3*ind+2] =255;
		}

//		for(unsigned int i = 0; i < label_indices[3].indices.size(); i++){
//			int ind = label_indices[3].indices[i];
//			depthedgesdata[ind] = 255;
//			out.data[3*ind+0] =0;
//			out.data[3*ind+1] =255;
//			out.data[3*ind+2] =255;
//		}


		for(unsigned int i = 0; i < label_indices[4].indices.size(); i++){
			int ind = label_indices[4].indices[i];
            depthedgesdata[ind] = 255;
//			out.data[3*ind+0] =0;
//			out.data[3*ind+1] =255;
//			out.data[3*ind+2] =0;
		}

//		cv::namedWindow( "edges", cv::WINDOW_AUTOSIZE );
//		cv::imshow( "edges", out );
//		cv::waitKey(0);

		//show(true);
	}

    if(true){

        for(int it = 0; it < 30; it++){
            for(int w = 0; w < width; w++){
                for(int h = 0; h < height;h++){
                    int ind = h*width+w;
                }
            }
        }
    }

if(false){
	show(false);

	int * last = new int[width*height];
	for(int i = 0; i < width*height; i++){last[i] = 0;}
	int current = 1;

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer ("3D Viewer"));
	viewer->setBackgroundColor (0, 0, 0);
	viewer->addCoordinateSystem (1.0);
	viewer->initCameraParameters ();

	for(int w = 0; w < width; w+=5){
		for(int h = 0; h < height;h+=5){
			int ind = h*width+w;

            int edgetype = depthedgesdata[ind];
            if(edgetype != 0 && depthdata[ind] != 0){
				double xx = 0.001;
				double xy = 0;
				double xz = 0;
				double yy = xx;
				double yz = 0;
				double zz = xx;
				double sum = 0.001;

				double ww = 4.0;
                double wh = 0;
				double hh = 4.0;
				double sum2 = 4.0;


				double z = idepth*double(depthdata[ind]);
				double x = (double(w) - cx) * z * ifx;
				double y = (double(h) - cy) * z * ify;

//				for(int ww = 0; ww < width; ww++){
//					for(int hh = 0; hh < height;hh++){
//						int ind = hh*width+ww;
//						pcl::PointXYZRGBA & p = cloud->points[ind];
//						p.b = 0;
//						p.g = 255;
//						p.r = 0;
//					}
//				}




				std::vector<int> todolistw;
				todolistw.push_back(w);
				std::vector<int> todolisth;
				todolisth.push_back(h);
				current++;
				last[ind] = current;

				cv::Mat rgbclone = rgb.clone();
                unsigned char * rgbclonedata = (unsigned char *)rgbclone.data;


				cv::circle(rgbclone, cv::Point(w,h), 2, cv::Scalar(0,255,0));
				cv::imshow( "rgbclone", rgbclone );


				printf("ratio = [");
				double best_ratio = 0;
				int tmp;
				for(int go = 0; go < 100000 && go < todolistw.size(); go++){
					int cw = todolistw[go];
					int ch = todolisth[go];
					int ind = ch*width+cw;

                    int dw = cw-w;
                    int dh = ch-h;
                    ww+=dw*dw;
                    wh+=dw*dh;
                    hh+=dh*dh;
                    sum2++;

                    Eigen::Matrix2d mimg;
                    mimg(0,0) = ww/sum2;
                    mimg(0,1) = wh/sum2;
                    mimg(1,0) = wh/sum2;
                    mimg(1,1) = hh/sum2;

					Eigen::EigenSolver<Eigen::Matrix2d> es(mimg);
					double e1 = (es.eigenvalues())(0).real();
					double e2 = (es.eigenvalues())(1).real();
					double ratio = 1;
					if(e1 > e2){ratio = e1/e2;}
					if(e2 > e1){ratio = e2/e1;}
					printf("%5.5f ",ratio);

					if(ratio < best_ratio){
						ww-=dw*dw;
						wh-=dw*dh;
						hh-=dh*dh;
						sum2--;
						//cv::circle(rgbclone, cv::Point(cw,ch), 2, cv::Scalar(255,0,255));
						continue;
					}

					best_ratio = ratio;
//                    cout << "The mimg are:" << endl << mimg << endl;
//					cout << "The eigenvalues of mimg are:" << endl << es.eigenvalues() << endl;
//					printf("%f %f\n",e1,e2);
//					cout << "The matrix of eigenvectors, V, is:" << endl << es.eigenvectors() << endl << endl;

//					B2=(-cov[0][0]-cov[1][1])*(-cov[0][0]-cov[1][1]);
//				    c= (4*cov[0][0]*cov[1][1])-(4*cov[0][1]*cov[1][0]);


//				    sq= sqrt(B2-c);


//				    B=(cov[0][0])+(cov[1][1]);

//				    r1=(B+sq)(.5);
//				    r2=(B-sq)(.5);


//				    printf("\nThe eigenvalues are &#37;f and %f", r1, r2);


					double z2 = idepth*double(depthdata[ind]);
					if(z2 > 0){
						double x2 = (double(cw) - cx) * z2 * ifx;
						double y2 = (double(ch) - cy) * z2 * ify;

						double dx = x-x2;
						double dy = y-y2;
						double dz = z-z2;

						xx += dx*dx;
						xy += dx*dy;
						xz += dx*dz;

						yy += dy*dy;
						yz += dy*dz;

						zz += dz*dz;
						sum++;
					}

					//double z = idepth*double(depthdata[ind]);
					rgbclonedata[3*ind+0] = 0;
					rgbclonedata[3*ind+1] = 0;
					rgbclonedata[3*ind+2] = 255;

//					cloud->points[ind].r = 255;
//					cloud->points[ind].g = 0;
//					cloud->points[ind].b = 0;

					int startw	= std::max(int(0),cw-1);
					int stopw	= std::min(int(width-1),cw+1);

					int starth	= std::max(int(0),ch-1);
					int stoph	= std::min(int(height-1),ch+1);

					for(int tw = startw; tw <= stopw; tw++){
						for(int th = starth; th <= stoph; th++){
							int ind2 = th*width+tw;
                            if(depthedgesdata[ind2] == edgetype && last[ind2] != current){
								todolistw.push_back(tw);
								todolisth.push_back(th);
								last[ind2] = current;
							}
						}
					}

				}


				printf("];\n");
                ww /= sum2;
                wh /= sum2;
                hh /= sum2;
                Eigen::Matrix2d mimg;
                mimg(0,0) = ww;
                mimg(0,1) = wh;
                mimg(1,0) = wh;
                mimg(1,1) = hh;

                Eigen::EigenSolver<Eigen::Matrix2d> es(mimg);
                cout << "The mimg are:" << endl << mimg << endl;
                cout << "The eigenvalues of mimg are:" << endl << es.eigenvalues() << endl;
                cout << "The matrix of eigenvectors, V, is:" << endl << es.eigenvectors() << endl << endl;

				//cv::circle(rgbclone, cv::Point(w,h), 3, cv::Scalar(0,255,0));


				cv::imshow( "rgbclone", rgbclone );
				cv::waitKey(0);
/*

				pcl::PointCloud<pcl::PointXYZRGBA>::Ptr	cloud2	(new pcl::PointCloud<pcl::PointXYZRGBA>);
				cloud2->width	= width;
				cloud2->height	= height;
				cloud2->points.resize(width*height);
					//cloud->dense = false;

				unsigned short *	depthdata	= (unsigned short *)depth.data;
				unsigned char *		rgbdata		= (unsigned char *)rgb.data;
				for(int w3 = 0; w3 < width; w3++){
					for(int h3 = 0; h3 < height;h3++){
						int ind3 = h3*width+w3;
						pcl::PointXYZRGBA & p = cloud2->points[ind3];
						p.b = rgbdata[3*ind3+0];//255;
						p.g = rgbdata[3*ind3+1];//255;
						p.r = rgbdata[3*ind3+2];//255;
					}
				}

				for(int go = 0; go < todolistw.size(); go++){
					int cw = todolistw[go];
					int ch = todolisth[go];
					int ind3 = ch*width+cw;

					pcl::PointXYZRGBA & p = cloud2->points[ind3];
					p.b = 0;
					p.g = 0;
					p.r = 255;
				}

                for(int w3 = 0; w3 < width; w3++){
                    for(int h3 = 0; h3 < height;h3++){
                        int ind3 = h3*width+w3;
                        pcl::PointXYZRGBA & p = cloud2->points[ind3];
                        double z3 = idepth*double(depthdata[ind3]);
                        if(z3 > 0){
                            p.x = (double(w3) - cx) * z3 * ifx;
                            p.y = (double(h3) - cy) * z3 * ify;
                            p.z = z3;
                        }else{
                            p.x = 0;
                            p.y = 0;
                            p.z = 0;
                        }
                    }
                }

				viewer->removeAllPointClouds();
				viewer->addPointCloud<pcl::PointXYZRGBA> (cloud2, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA>(cloud2), "cloud");
				viewer->spin();
				*/
			}
		}
	}
}

}

RGBDFrame::~RGBDFrame(){
	rgb.release();
	normals.release();
	depth.release();
	depthedges.release();
	if(labels != 0){delete labels; labels = 0;}
}

void RGBDFrame::show(bool stop){
	cv::namedWindow( "rgb", cv::WINDOW_AUTOSIZE );			cv::imshow( "rgb", rgb );
	cv::namedWindow( "normals", cv::WINDOW_AUTOSIZE );		cv::imshow( "normals", normals );
	cv::namedWindow( "depth", cv::WINDOW_AUTOSIZE );		cv::imshow( "depth", depth );
	cv::namedWindow( "depthedges", cv::WINDOW_AUTOSIZE );	cv::imshow( "depthedges", depthedges );
	if(stop){	cv::waitKey(0);}else{					cv::waitKey(30);}
}

pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr RGBDFrame::getPCLcloud(){
	unsigned char * rgbdata = (unsigned char *)rgb.data;
	unsigned short * depthdata = (unsigned short *)depth.data;

	const unsigned int width	= camera->width; const unsigned int height	= camera->height;
	const double idepth			= camera->idepth_scale;
	const double cx				= camera->cx;		const double cy				= camera->cy;
	const double ifx			= 1.0/camera->fx;	const double ify			= 1.0/camera->fy;

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr	cloud	(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::Normal>::Ptr		normals (new pcl::PointCloud<pcl::Normal>);
	cloud->width	= width;
	cloud->height	= height;
	cloud->points.resize(width*height);

	for(unsigned int w = 0; w < width; w++){
		for(unsigned int h = 0; h < height;h++){
			int ind = h*width+w;
			double z = idepth*double(depthdata[ind]);

			pcl::PointXYZRGB p;
			p.b = rgbdata[3*ind+0];
			p.g = rgbdata[3*ind+1];
			p.r = rgbdata[3*ind+2];
			if(z > 0){
				p.x = (double(w) - cx) * z * ifx;
				p.y = (double(h) - cy) * z * ify;
				p.z = z;
			}else{
				p.x = NAN;
				p.y = NAN;
				p.z = NAN;
			}
			cloud->points[ind] = p;
		}
	}

	pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
	ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT);
	ne.setMaxDepthChangeFactor(0.02f);
	ne.setNormalSmoothingSize(10.0f);
	ne.setInputCloud(cloud);
	ne.compute(*normals);

	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	cloud_ptr->width	= width;
	cloud_ptr->height	= height;
	cloud_ptr->points.resize(width*height);

	for(unsigned int w = 0; w < width; w++){
		for(unsigned int h = 0; h < height;h++){
			int ind = h*width+w;
			pcl::PointXYZRGBNormal & p0	= cloud_ptr->points[ind];
			pcl::PointXYZRGB p1			= cloud->points[ind];
			pcl::Normal p2				= normals->points[ind];
			p0.x		= p1.x;
			p0.y		= p1.y;
			p0.z		= p1.z;
			p0.r		= p1.r;
			p0.g		= p1.g;
			p0.b		= p1.b;
			p0.normal_x	= p2.normal_x;
			p0.normal_y	= p2.normal_y;
			p0.normal_z	= p2.normal_z;
		}
	}
	return cloud_ptr;
}

void RGBDFrame::savePCD(std::string path, Eigen::Matrix4d pose){
    printf("saving pcd: %s\n",path.c_str());
    unsigned char * rgbdata = (unsigned char *)rgb.data;
    unsigned short * depthdata = (unsigned short *)depth.data;

    const unsigned int width	= camera->width; const unsigned int height	= camera->height;
    const double idepth			= camera->idepth_scale;
    const double cx				= camera->cx;		const double cy				= camera->cy;
    const double ifx			= 1.0/camera->fx;	const double ify			= 1.0/camera->fy;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr	cloud	(new pcl::PointCloud<pcl::PointXYZRGB>);
    cloud->width	= width;
    cloud->height	= height;
    cloud->points.resize(width*height);

    for(unsigned int w = 0; w < width; w++){
        for(unsigned int h = 0; h < height;h++){
            int ind = h*width+w;
            double z = idepth*double(depthdata[ind]);
            if(z > 0){
                pcl::PointXYZRGB p;
                p.x = (double(w) - cx) * z * ifx;
                p.y = (double(h) - cy) * z * ify;
                p.z = z;
                p.b = rgbdata[3*ind+0];
                p.g = rgbdata[3*ind+1];
                p.r = rgbdata[3*ind+2];
                cloud->points[ind] = p;
            }
        }
    }

    //Mat4f2RotTrans(const Eigen::Matrix4f &tf, Eigen::Quaternionf &q, Eigen::Vector4f &trans)
    Mat4f2RotTrans(pose.cast<float>(),cloud->sensor_orientation_,cloud->sensor_origin_);
    int success = pcl::io::savePCDFileBinaryCompressed(path,*cloud);
}

void RGBDFrame::save(std::string path){
	//printf("saving frame %i to %s\n",id,path.c_str());

	cv::imwrite( path+"_rgb.png", rgb );
	cv::imwrite( path+"_depth.png", depth );

	unsigned long buffersize = 19*sizeof(double);
	char* buffer = new char[buffersize];
	double * buffer_double = (double *)buffer;
	unsigned long * buffer_long = (unsigned long *)buffer;

	int counter = 0;
	buffer_double[counter++] = capturetime;
	for(int i = 0; i < 4; i++){
		for(int j = 0; j < 4; j++){
			buffer_double[counter++] = pose(i,j);
		}
	}
	buffer_long[counter++] = sweepid;
	buffer_long[counter++] = camera->id;
	std::ofstream outfile (path+"_data.txt",std::ofstream::binary);
	outfile.write (buffer,buffersize);
	outfile.close();
	delete[] buffer;
}

RGBDFrame * RGBDFrame::load(Camera * cam, std::string path){
	printf("RGBDFrame * RGBDFrame::load(Camera * cam, std::string path)\n");

	std::streampos size;
	char * buffer;
	char buf [1024];
	std::string datapath = path+"_data.txt";
	std::ifstream file (datapath, std::ios::in | std::ios::binary | std::ios::ate);
	if (file.is_open()){
		size = file.tellg();
		buffer = new char [size];
		file.seekg (0, std::ios::beg);
		file.read (buffer, size);
		file.close();

		double * buffer_double = (double *)buffer;
		unsigned long * buffer_long = (unsigned long *)buffer;

		int counter = 0;
		double capturetime = buffer_double[counter++];
		Eigen::Matrix4d pose;
		for(int i = 0; i < 4; i++){
			for(int j = 0; j < 4; j++){
				pose(i,j) = buffer_double[counter++];
			}
		}
		int sweepid = buffer_long[counter++];
		int camera_id = buffer_long[counter++];

		cv::Mat rgb = cv::imread(path+"_rgb.png", -1);   // Read the file
		cv::Mat depth = cv::imread(path+"_depth.png", -1);   // Read the file

		RGBDFrame * frame = new RGBDFrame(cam,rgb,depth,capturetime,pose);
		frame->sweepid = sweepid;
		//printf("sweepid: %i",sweepid);

		return frame;
	}else{printf("cant open %s\n",(path+"/data.txt").c_str());}
}

}
