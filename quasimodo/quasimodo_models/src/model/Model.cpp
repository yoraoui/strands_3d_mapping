#include "model/Model.h"
#include <map>
#include <sys/stat.h>

namespace reglib
{
using namespace Eigen;

unsigned int model_id_counter = 0;

Model::Model(){
    updated = false;
	total_scores = 0;
	score = 0;
	id = model_id_counter++;
	last_changed = -1;
	savePath = "";
	soma_id = "";
	pointspath = "";
	retrieval_object_id = "";
	retrieval_vocabulary_id = "";
	parrent = 0;
}

Model::Model(RGBDFrame * frame, cv::Mat mask, Eigen::Matrix4d pose){
    updated = false;
	total_scores = 0;
	scores.resize(1);
	scores.back().resize(1);
	scores[0][0] = 0;

	score = 0;
	id = model_id_counter++;

	last_changed = -1;

	relativeposes.push_back(pose);
	frames.push_back(frame);
	modelmasks.push_back(new ModelMask(mask));

	savePath = "";
	soma_id = "";
	pointspath = "";
	retrieval_object_id = "";
	retrieval_vocabulary_id = "";
    parrent = 0;
}

void Model::getData(std::vector<Eigen::Matrix4d> & po, std::vector<RGBDFrame*> & fr, std::vector<ModelMask*> & mm, Eigen::Matrix4d p){
	for(unsigned int i = 0; i < frames.size(); i++){
		fr.push_back(frames[i]);
		mm.push_back(modelmasks[i]);
		po.push_back(p*relativeposes[i]);
	}

	for(unsigned int i = 0; i < submodels.size(); i++){
		submodels[i]->getData(po,fr,mm, p*submodels_relativeposes[i]);
	}
}

void Model::mergeKeyPoints(Model * model, Eigen::Matrix4d p){
    const double & m00 = p(0,0); const double & m01 = p(0,1); const double & m02 = p(0,2); const double & m03 = p(0,3);
    const double & m10 = p(1,0); const double & m11 = p(1,1); const double & m12 = p(1,2); const double & m13 = p(1,3);
    const double & m20 = p(2,0); const double & m21 = p(2,1); const double & m22 = p(2,2); const double & m23 = p(2,3);

    std::vector<KeyPoint> & kps = model->keypoints;

    for(unsigned int i = 0; i < kps.size(); i++){
        superpoint & sp = kps[i].point;
        const double x = sp.x;
        const double y = sp.y;
        const double z = sp.z;
        const double nx = sp.nx;
        const double ny = sp.ny;
        const double nz = sp.nz;

        sp.x = m00*x + m01*y + m02*z + m03;
        sp.y = m10*x + m11*y + m12*z + m13;
        sp.z = m20*x + m21*y + m22*z + m23;
        sp.nx = m00*nx + m01*ny + m02*nz;
        sp.ny = m10*nx + m11*ny + m12*nz;
        sp.nz = m20*nx + m21*ny + m22*nz;
    }

    unsigned int d_nrp = keypoints.size();
    if(d_nrp == 0){keypoints = kps;}
    else{
        double * dp = new double[3*d_nrp];
        //printf("d_nrp: %i\n",d_nrp);


        for(unsigned int i = 0; i < d_nrp; i++){
            superpoint & p = keypoints[i].point;
            dp[3*i+0]   = p.x;
            dp[3*i+1]   = p.y;
            dp[3*i+2]   = p.z;
        }

        ArrayData3D<double> * a3d = new ArrayData3D<double>;
        a3d->data	= dp;
        a3d->rows	= d_nrp;
        Tree3d * trees3d	= new Tree3d(3, *a3d, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        trees3d->buildIndex();
        //keypoints = kps;

        int capacity = 1;
        size_t * ret_indexes = new size_t[capacity];
        double * out_dists_sqr = new double[capacity];
        nanoflann::KNNResultSet<double> resultSet(capacity);
        std::vector<int> matches;
        matches.resize(kps.size());

        //printf("kps.size()): %i\n",kps.size());

        std::vector<double> residuals;
        double stdval = 0;

        double tsp [3];
        for(unsigned int i = 0; i < kps.size(); i++){
            superpoint & sp = kps[i].point;


            tsp[0] = sp.x;
            tsp[1] = sp.y;
            tsp[2] = sp.z;

            resultSet.init(ret_indexes, out_dists_sqr);
            trees3d->findNeighbors(resultSet,tsp, nanoflann::SearchParams(10));
            unsigned int id = ret_indexes[0];

            matches[i] = id;

            superpoint & skp = keypoints[id].point;

            double rangew = sqrt( 1.0/(1.0/skp.point_information +1.0/sp.point_information) );

            double dx = rangew * (tsp[0]-dp[3*id+0]);
            double dy = rangew * (tsp[1]-dp[3*id+1]);
            double dz = rangew * (tsp[2]-dp[3*id+2]);

            residuals.push_back(dx);
            residuals.push_back(dy);
            residuals.push_back(dz);
            stdval += dx*dx+dy*dy+dz*dz;
        }


        stdval = sqrt(stdval/double(kps.size()*3 -1 ));

        DistanceWeightFunction2PPR2 * func = new DistanceWeightFunction2PPR2();
        func->zeromean				= true;
        func->maxp					= 0.99;
        func->startreg				= 0.0005;
        func->debugg_print			= false;
        func->maxd					= 0.1;
        func->startmaxd				= func->maxd;
        func->histogram_size		= 100;
        func->starthistogram_size	= func->histogram_size;
        func->stdval2				= stdval;
        func->maxnoise				= stdval;
        func->reset();
        ((DistanceWeightFunction2*)func)->computeModel(residuals);

        printf("===================================\n");
        printf("before keypoints.size(): %i kps.size(): %i\n",keypoints.size(),kps.size());
        for(unsigned int i = 0; i < kps.size(); i++){
            KeyPoint & sp1 = kps[i];
            KeyPoint & sp2 = keypoints[matches[i]];

            double probX = func->getProb(residuals[3*i+0]);
            double probY = func->getProb(residuals[3*i+1]);
            double probZ = func->getProb(residuals[3*i+2]);
            double prob = probX*probY*probZ/(probX*probY*probZ + (1-probX)*(1-probY)*(1-probZ));

            if(prob > 0.5){
                sp2.merge(sp1);
            }else{
                keypoints.push_back(sp1);
            }

        }
        printf("after keypoints.size(): %i\n",keypoints.size());

        delete func;
        delete[] out_dists_sqr;
        delete[] ret_indexes;
        delete trees3d;
        delete a3d;
        delete[] dp;
    }
}

void Model::addSuperPoints(vector<superpoint> & spvec, Matrix4d p, RGBDFrame* frame, ModelMask* modelmask, boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer){
	Camera * camera				= frame->camera;
	const unsigned int width	= camera->width;
	const unsigned int height	= camera->height;
	unsigned long nr_pixels		= width*height;

	std::vector<float> sumw;
	sumw.resize(nr_pixels);
	for(unsigned long ind = 0; ind < nr_pixels;ind++){sumw[ind] = 0;}

	std::vector<bool> isfused;
	isfused.resize(nr_pixels);
	for(unsigned int i = 0; i < width*height; i++){isfused[i] = false;}

	bool * maskvec							= modelmask->maskvec;
	std::vector<ReprojectionResult> rr_vec	= frame->getReprojections(spvec,p.inverse(),modelmask->maskvec,false);
	std::vector<superpoint> framesp			= frame->getSuperPoints(p);


    for(unsigned long ind = 0; ind < nr_pixels;ind++){framesp[ind].last_update_frame_id = last_changed;}

	unsigned long nr_rr = rr_vec.size();
	if(nr_rr > 10){
		std::vector<double> residualsZ;
		double stdval = 0;
		for(unsigned long ind = 0; ind < nr_rr;ind++){
			ReprojectionResult & rr = rr_vec[ind];
			superpoint & src_p =   spvec[rr.src_ind];
			superpoint & dst_p = framesp[rr.dst_ind];
			double src_variance = 1.0/src_p.point_information;
			double dst_variance = 1.0/dst_p.point_information;
			double total_variance = src_variance+dst_variance;
			double total_stdiv = sqrt(total_variance);
			double d = rr.residualZ;
			residualsZ.push_back(d/total_stdiv);
			stdval += residualsZ.back()*residualsZ.back();
		}
		stdval = sqrt(stdval/double(nr_rr));

		DistanceWeightFunction2PPR2 * func = new DistanceWeightFunction2PPR2();
		func->zeromean				= true;
		func->maxp					= 0.99;
		func->startreg				= 0.001;
		func->debugg_print			= false;
		func->maxd					= 0.1;
		func->startmaxd				= func->maxd;
		func->histogram_size		= 100;
		func->starthistogram_size	= func->histogram_size;
		func->stdval2				= stdval;
		func->maxnoise				= stdval;
		func->reset();
		((DistanceWeightFunction2*)func)->computeModel(residualsZ);

		for(unsigned long ind = 0; ind < nr_rr;ind++){
			double p = func->getProb(residualsZ[ind]);
			if(p > 0.5){sumw[rr_vec[ind].dst_ind] += p;}
		}

		for(unsigned long ind = 0; ind < nr_rr;ind++){
			double rz = residualsZ[ind];
			double p = func->getProb(rz);

			if(p > 0.5){
				ReprojectionResult & rr = rr_vec[ind];
				superpoint & src_p =   spvec[rr.src_ind];
				superpoint & dst_p = framesp[rr.dst_ind];
				float weight = p/sumw[rr.dst_ind];
				src_p.merge(dst_p,weight);
			}else if(false && p < 0.001 && rz > 0){//If occlusion: either the new or the old point is unreliable, reduce confidence in both
				ReprojectionResult & rr = rr_vec[ind];
				superpoint & src_p =   spvec[rr.src_ind];
				superpoint & dst_p = framesp[rr.dst_ind];

				double dst_pi = dst_p.point_information;
				src_p.point_information -= 0.5*dst_pi;
				dst_p.point_information -= 0.5*dst_pi;
			}
		}
		delete func;
	}

	for(unsigned long ind = 0; ind < nr_pixels;ind++){
		if(maskvec[ind] != 0 && sumw[ind] < 0.5){
			superpoint & sp = framesp[ind];
			if(sp.point_information > 0){
				spvec.push_back(sp);
			}
		}
	}


	//Clear out points with bad/no information
	long nr_spvec = spvec.size();
	for(long ind = 0; ind < nr_spvec;ind++){
		if(spvec[ind].point_information <= 0){
			spvec[ind] = spvec.back();
			spvec.pop_back();
			ind--;
			nr_spvec--;
		}
	}

	if(viewer != 0){
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud = getPointCloudFromVector(spvec,1);
		viewer->addPointCloud<pcl::PointXYZRGBNormal> (cloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(cloud), "model");
		viewer->spin();
		viewer->removeAllPointClouds();
	}
}

void Model::addAllSuperPoints(std::vector<superpoint> & spvec, Eigen::Matrix4d pose, boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer){
    updated = true;
	for(unsigned int i = 0; i < frames.size(); i++){
		addSuperPoints(spvec, pose*relativeposes[i], frames[i], modelmasks[i], viewer);
	}

	for(unsigned int i = 0; i < submodels.size(); i++){
		submodels[i]->addAllSuperPoints(spvec, pose*submodels_relativeposes[i], viewer);
	}
}

void Model::recomputeModelPoints(Eigen::Matrix4d pose, boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer){
	double startTime = getTime();
	points.clear();
	addAllSuperPoints(points,pose,viewer);
	//printf("recomputeModelPoints time: %5.5fs total points: %i \n",getTime()-startTime,int(points.size()));
}

void Model::showHistory(boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer){
	for(unsigned int i = 0; i < submodels.size(); i++){
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr room_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

		for(unsigned int j = 0; j < submodels[i]->frames.size(); j++){
			bool * maskvec					= submodels[i]->modelmasks[j]->maskvec;
			unsigned char  * rgbdata		= (unsigned char	*)(submodels[i]->frames[j]->rgb.data);
			unsigned short * depthdata		= (unsigned short	*)(submodels[i]->frames[j]->depth.data);
			float		   * normalsdata	= (float			*)(submodels[i]->frames[j]->normals.data);

			Camera * camera				= submodels[i]->frames[j]->camera;
			const unsigned int width	= camera->width;
			const unsigned int height	= camera->height;
			const float idepth			= camera->idepth_scale;
			const float cx				= camera->cx;
			const float cy				= camera->cy;
			const float ifx				= 1.0/camera->fx;
			const float ify				= 1.0/camera->fy;

			Eigen::Matrix4d p = submodels[i]->relativeposes[j];

			float m00 = p(0,0); float m01 = p(0,1); float m02 = p(0,2); float m03 = p(0,3);
			float m10 = p(1,0); float m11 = p(1,1); float m12 = p(1,2); float m13 = p(1,3);
			float m20 = p(2,0); float m21 = p(2,1); float m22 = p(2,2); float m23 = p(2,3);

			for(unsigned int w = 0; w < width; w++){
				for(unsigned int h = 0; h < height;h++){
					int ind = h*width+w;
					if(!maskvec[ind]){
						float z = idepth*float(depthdata[ind]);
						float nx = normalsdata[3*ind+0];

						if(z > 0 && nx != 2){
							float ny = normalsdata[3*ind+1];
							float nz = normalsdata[3*ind+2];

							float x = (w - cx) * z * ifx;
							float y = (h - cy) * z * ify;

							pcl::PointXYZRGB po;
							po.x	= m00*x + m01*y + m02*z + m03;
							po.y	= m10*x + m11*y + m12*z + m13;
							po.z	= m20*x + m21*y + m22*z + m23;
							//							float pnx	= m00*nx + m01*ny + m02*nz;
							//							float pny	= m10*nx + m11*ny + m12*nz;
							//							float pnz	= m20*nx + m21*ny + m22*nz;

							po.b = rgbdata[3*ind+0];
							po.g = rgbdata[3*ind+1];
							po.r = rgbdata[3*ind+2];


							room_cloud->points.push_back(po);
						}
					}
				}
			}
		}

		viewer->removeAllPointClouds();
		viewer->addPointCloud<pcl::PointXYZRGB> (room_cloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(room_cloud), "room_cloud");

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = submodels[i]->getPCLcloud(1, true);
		for(unsigned int j = 0; j < cloud->points.size(); j++){
			cloud->points[j].r = 255;
			cloud->points[j].g = 0;
			cloud->points[j].b = 0;
		}
		viewer->addPointCloud<pcl::PointXYZRGB> (cloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(cloud), "submodel");

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud2 = getPCLcloud(1, true);

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr tcloud (new pcl::PointCloud<pcl::PointXYZRGB> ());
		pcl::transformPointCloud (*cloud2, *tcloud, Eigen::Affine3d(submodels_relativeposes[i]).inverse());
		viewer->addPointCloud<pcl::PointXYZRGB> (tcloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(tcloud), "model");
		viewer->spin();
	}
}

std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> Model::getHistory(){
	std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> ret;
	for(unsigned int i = 0; i < submodels.size(); i++){
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr room_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
		for(unsigned int j = 0; j < submodels[i]->frames.size(); j++){
			bool * maskvec					= submodels[i]->modelmasks[j]->maskvec;
			unsigned char  * rgbdata		= (unsigned char	*)(submodels[i]->frames[j]->rgb.data);
			unsigned short * depthdata		= (unsigned short	*)(submodels[i]->frames[j]->depth.data);
			float		   * normalsdata	= (float			*)(submodels[i]->frames[j]->normals.data);

			Camera * camera				= submodels[i]->frames[j]->camera;
			const unsigned int width	= camera->width;
			const unsigned int height	= camera->height;
			const float idepth			= camera->idepth_scale;
			const float cx				= camera->cx;
			const float cy				= camera->cy;
			const float ifx				= 1.0/camera->fx;
			const float ify				= 1.0/camera->fy;

			Eigen::Matrix4d p = submodels[i]->relativeposes[j];

			float m00 = p(0,0); float m01 = p(0,1); float m02 = p(0,2); float m03 = p(0,3);
			float m10 = p(1,0); float m11 = p(1,1); float m12 = p(1,2); float m13 = p(1,3);
			float m20 = p(2,0); float m21 = p(2,1); float m22 = p(2,2); float m23 = p(2,3);

			for(unsigned int w = 0; w < width; w++){
				for(unsigned int h = 0; h < height;h++){
					int ind = h*width+w;
					if(!maskvec[ind]){
						float z = idepth*float(depthdata[ind]);
						float nx = normalsdata[3*ind+0];

						if(z > 0 && nx != 2){
							float ny = normalsdata[3*ind+1];
							float nz = normalsdata[3*ind+2];

							float x = (w - cx) * z * ifx;
							float y = (h - cy) * z * ify;

							pcl::PointXYZRGB po;
							po.x	= m00*x + m01*y + m02*z + m03;
							po.y	= m10*x + m11*y + m12*z + m13;
							po.z	= m20*x + m21*y + m22*z + m23;
							//							float pnx	= m00*nx + m01*ny + m02*nz;
							//							float pny	= m10*nx + m11*ny + m12*nz;
							//							float pnz	= m20*nx + m21*ny + m22*nz;

							po.b = rgbdata[3*ind+0];
							po.g = rgbdata[3*ind+1];
							po.r = rgbdata[3*ind+2];


							room_cloud->points.push_back(po);
						}
					}
				}
			}
		}


		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = submodels[i]->getPCLcloud(1, true);
		for(unsigned int j = 0; j < cloud->points.size(); j++){
			cloud->points[j].r = 255;
			cloud->points[j].g = 0;
			cloud->points[j].b = 0;
		}

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud2 = getPCLcloud(1, true);

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr tcloud (new pcl::PointCloud<pcl::PointXYZRGB> ());
		pcl::transformPointCloud (*cloud2, *tcloud, Eigen::Affine3d(submodels_relativeposes[i]).inverse());


		*room_cloud += *cloud;
		*room_cloud += *tcloud;
		ret.push_back(room_cloud);
	}
	return ret;
}

void Model::addPointsToModel(RGBDFrame * frame, ModelMask * modelmask, Eigen::Matrix4d p){
	bool * maskvec = modelmask->maskvec;
	unsigned char  * rgbdata		= (unsigned char	*)(frame->rgb.data);
	unsigned short * depthdata		= (unsigned short	*)(frame->depth.data);
	float		   * normalsdata	= (float			*)(frame->normals.data);

	unsigned int frameid = frame->id;

	float m00 = p(0,0); float m01 = p(0,1); float m02 = p(0,2); float m03 = p(0,3);
	float m10 = p(1,0); float m11 = p(1,1); float m12 = p(1,2); float m13 = p(1,3);
	float m20 = p(2,0); float m21 = p(2,1); float m22 = p(2,2); float m23 = p(2,3);

	Camera * camera				= frame->camera;
	const unsigned int width	= camera->width;
	const unsigned int height	= camera->height;
	const float idepth			= camera->idepth_scale;
	const float cx				= camera->cx;
	const float cy				= camera->cy;
	const float ifx				= 1.0/camera->fx;
	const float ify				= 1.0/camera->fy;

	for(unsigned int w = 0; w < width; w++){
		for(unsigned int h = 0; h < height;h++){
			int ind = h*width+w;
			if(maskvec[ind]){
				float z = idepth*float(depthdata[ind]);
				float nx = normalsdata[3*ind+0];

				if(z > 0 && nx != 2){
					float ny = normalsdata[3*ind+1];
					float nz = normalsdata[3*ind+2];

					float x = (w - cx) * z * ifx;
					float y = (h - cy) * z * ify;

					float px	= m00*x + m01*y + m02*z + m03;
					float py	= m10*x + m11*y + m12*z + m13;
					float pz	= m20*x + m21*y + m22*z + m23;
					float pnx	= m00*nx + m01*ny + m02*nz;
					float pny	= m10*nx + m11*ny + m12*nz;
					float pnz	= m20*nx + m21*ny + m22*nz;

					float pb = rgbdata[3*ind+0];
					float pg = rgbdata[3*ind+1];
					float pr = rgbdata[3*ind+2];

					Vector3f	pxyz	(px	,py	,pz );
					Vector3f	pnxyz	(pnx,pny,pnz);
					Vector3f	prgb	(pr	,pg	,pb );
					float		weight	= 1.0/(z*z);
					points.push_back(superpoint(pxyz,pnxyz,prgb, weight, weight, frameid));
				}
			}
		}
	}
}

bool Model::testFrame(int ind){
	printf("testing frame %i\n",ind);
	Eigen::Matrix3f covMat = Eigen::Matrix3f::Identity();

	ModelMask * modelmask = modelmasks[ind];
	RGBDFrame * frame = frames[ind];

	bool * maskvec = modelmask->maskvec;
	unsigned char  * rgbdata		= (unsigned char	*)(frame->rgb.data);
	unsigned short * depthdata		= (unsigned short	*)(frame->depth.data);
	float		   * normalsdata	= (float			*)(frame->normals.data);

	Camera * camera				= frame->camera;
	const unsigned int width	= camera->width;
	const unsigned int height	= camera->height;

	double tot_w = 0;
	for(unsigned int w = 0; w < width; w++){
		for(unsigned int h = 0; h < height;h++){
			int ind = h*width+w;
			if(maskvec[ind]){
				float nx = normalsdata[3*ind+0];
				if(nx != 2){
					float ny = normalsdata[3*ind+1];
					float nz = normalsdata[3*ind+2];
					covMat(0,0) += nx*nx;
					covMat(0,1) += nx*ny;
					covMat(0,2) += nx*nz;

					covMat(1,0) += ny*nx;
					covMat(1,1) += ny*ny;
					covMat(1,2) += ny*nz;


					covMat(2,0) += nz*nx;
					covMat(2,1) += nz*ny;
					covMat(2,2) += nz*nz;

					tot_w++;
				}
			}
		}
	}




	printf("totw: %f\n",tot_w);
	double threshold = 500;
	if(tot_w < threshold){return false;}

	//	for(int i = 0; i < 3; i++){
	//		for(int j = 0; j < 3; j++){
	//			covMat(i,j) /= tot_w;
	//		}
	//	}

	Eigen::EigenSolver<Eigen::Matrix3f> es(covMat, false);
	auto e = es.eigenvalues();

	double e1 = e(0).real();
	double e2 = e(1).real();
	double e3 = e(2).real();

	printf("%f %f %f\n",e1,e2,e3);

	if(e1 > threshold || e2 > threshold || e3 > threshold){return false;}
	return true;
}

void Model::print(){
	printf("id: %i ",int(id));
	printf("last_changed: %i ",int(last_changed));
	printf("score: %f ",score);
	printf("total_scores: %f ",total_scores);
	printf("frames: %i ",int(frames.size()));
	printf("modelmasks: %i ",int(modelmasks.size()));
	printf("relativeposes: %i\n",int(relativeposes.size()));
}

void Model::addFrameToModel(RGBDFrame * frame,  ModelMask * modelmask, Eigen::Matrix4d p){
	addPointsToModel(frame, modelmask, p);

	relativeposes.push_back(p);
	frames.push_back(frame);
	modelmasks.push_back(modelmask);
}

void Model::merge(Model * model, Eigen::Matrix4d p){
    printf("merge: %i\n",__LINE__);
	for(unsigned int i = 0; i < model->frames.size(); i++){
		relativeposes.push_back(p * model->relativeposes[i]);
		frames.push_back(model->frames[i]);
		modelmasks.push_back(model->modelmasks[i]);
	}

	for(unsigned int i = 0; i < model->submodels.size(); i++){
		submodels_relativeposes.push_back(p * model->submodels_relativeposes[i]);
		submodels.push_back(model->submodels[i]);
		model->submodels[i]->parrent = this;
	}

	recomputeModelPoints();
}

//CloudData * Model::getCD(unsigned int target_points){
//	std::vector<unsigned int> ro;
//	unsigned int nc = points.size();
//	ro.resize(nc);
//	for(unsigned int i = 0; i < nc; i++){ro[i] = i;}
//	for(unsigned int i = 0; i < nc; i++){
//		unsigned int randval = rand();
//		unsigned int rind = randval%nc;
//		int tmp = ro[i];
//		ro[i] = ro[rind];
//		ro[rind] = tmp;
//	}
//	//Build registration input
//	unsigned int nr_points = std::min(unsigned(points.size()),target_points);
//	MatrixXd data			(6,nr_points);
//	MatrixXd data_normals	(3,nr_points);
//	MatrixXd information	(6,nr_points);

//	for(unsigned int k = 0; k < nr_points; k++){
//		superpoint & p		= points[ro[k]];
//		data(0,k)			= p.x;
//		data(1,k)			= p.y;
//		data(2,k)			= p.z;
//		data(3,k)			= p.r;
//		data(4,k)			= p.g;
//		data(5,k)			= p.b;
//		data_normals(0,k)	= p.nx;
//		data_normals(1,k)	= p.ny;
//		data_normals(2,k)	= p.nz;
//		information(0,k)	= p.point_information;
//		information(1,k)	= p.point_information;
//		information(2,k)	= p.point_information;
//		information(3,k)	= p.colour_information;
//		information(4,k)	= p.colour_information;
//		information(5,k)	= p.colour_information;
//	}

//	CloudData * cd			= new CloudData();
//	cd->data				= data;
//	cd->information			= information;
//	cd->normals				= data_normals;
//	return cd;
//}

Model::~Model(){
    //printf("delete(%i) -> %s\n",long(this),keyval.c_str());
}

void Model::fullDelete(){
    //printf("fullDelete(%i) -> %s\n",long(this),keyval.c_str());
	points.clear();
	all_keypoints.clear();
	all_descriptors.clear();
	relativeposes.clear();

    for(size_t i = 0; i < frames.size(); i++){
        delete frames[i]->camera;
        delete frames[i];
	}
	frames.clear();

	for(size_t i = 0; i < modelmasks.size(); i++){delete modelmasks[i];}
	modelmasks.clear();

	rep_relativeposes.clear();
	rep_frames.clear();
	rep_modelmasks.clear();
	total_scores = 0;
	scores.clear();

	for(size_t i = 0; i < submodels.size(); i++){
		submodels[i]->fullDelete();
		delete submodels[i];
	}
	submodels.clear();
	submodels_relativeposes.clear();
	submodels_scores.clear();
}

pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr Model::getPCLnormalcloud(int step, bool color){
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	for(unsigned int i = 0; i < points.size(); i+=step){
		superpoint & sp = points[i];
		pcl::PointXYZRGBNormal p;
		p.x = sp.x;
		p.y = sp.y;
		p.z = sp.z;

		p.normal_x = sp.nx;
		p.normal_y = sp.ny;
		p.normal_z = sp.nz;
		if(color){
			p.b =   0;
			p.g = 255;
			p.r =   0;
		}else{
			p.b = sp.r;
			p.g = sp.g;
			p.r = sp.b;
		}
		cloud_ptr->points.push_back(p);
	}
	return cloud_ptr;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr Model::getPCLcloud(int step, bool color){
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
    for(unsigned int i = 0; i < points.size(); i+=step){
        superpoint & sp = points[i];
        pcl::PointXYZRGB p;
        p.x = sp.x;
        p.y = sp.y;
        p.z = sp.z;

        if(color){
            p.b =   0;
            p.g = 255;
            p.r =   0;
        }else{
            p.b = sp.r;
            p.g = sp.g;
            p.r = sp.b;
        }
        cloud_ptr->points.push_back(p);
    }
    return cloud_ptr;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr Model::getPCLEdgeCloud(int step, int type){

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);

    int rr = 55 + rand()%200;
    int rg = 55 + rand()%200;
    int rb = 55 + rand()%200;
    for(unsigned int i = 0; i < color_edgepoints.size(); i+=step){
        superpoint & sp = color_edgepoints[i];
        pcl::PointXYZRGB p;
        p.x = sp.x;
        p.y = sp.y;
        p.z = sp.z;

        if(type == 0){
            p.b =   0;
            p.g =   0;
            p.r =   255;
        }

        if(type == 1){
            p.b = sp.r;
            p.g = sp.g;
            p.r = sp.b;
        }

        if(type == 2){
            p.b = rb;
            p.g = rg;
            p.r = rr;
        }

        if(type == 3){
            if(sp.is_boundry){
                p.b = 0;
                p.g = 0;
                p.r = 0;
            }else{
                p.b = 255*fabs(sp.nz);
                p.g = 255*fabs(sp.ny);
                p.r = 255*fabs(sp.nx);
            }
        }

        cloud_ptr->points.push_back(p);
    }
    return cloud_ptr;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr Model::getPCLcloud(int step, int type){
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);

    int rr = 55 + rand()%200;
    int rg = 55 + rand()%200;
    int rb = 55 + rand()%200;
    for(unsigned int i = 0; i < points.size(); i+=step){
        superpoint & sp = points[i];
        pcl::PointXYZRGB p;
        p.x = sp.x;
        p.y = sp.y;
        p.z = sp.z;

        if(type == 0){
            p.b =   0;
            p.g = 255;
            p.r =   0;
        }

        if(type == 1){
            p.b = sp.r;
            p.g = sp.g;
            p.r = sp.b;
        }

        if(type == 2){
            p.b = rb;
            p.g = rg;
            p.r = rr;
        }

        if(type == 3){
            if(sp.is_boundry){
                p.b = 0;
                p.g = 0;
                p.r = 0;
            }else{
                p.b = 255*fabs(sp.nz);
                p.g = 255*fabs(sp.ny);
                p.r = 255*fabs(sp.nx);
            }
        }

        cloud_ptr->points.push_back(p);
    }
    return cloud_ptr;
}

void Model::mergePoints( std::vector<superpoint> & spoints, std::vector<superpoint> & dpoints, Eigen::Matrix4d p){
    const double & m00 = p(0,0); const double & m01 = p(0,1); const double & m02 = p(0,2); const double & m03 = p(0,3);
    const double & m10 = p(1,0); const double & m11 = p(1,1); const double & m12 = p(1,2); const double & m13 = p(1,3);
    const double & m20 = p(2,0); const double & m21 = p(2,1); const double & m22 = p(2,2); const double & m23 = p(2,3);


    unsigned int s_nrp = spoints.size();
    unsigned int d_nrp = dpoints.size();
    for(unsigned int i = 0; i <d_nrp; i++){
        superpoint & sp = dpoints[i];
        const double x = sp.x;
        const double y = sp.y;
        const double z = sp.z;
        const double nx = sp.nx;
        const double ny = sp.ny;
        const double nz = sp.nz;

        sp.x = m00*x + m01*y + m02*z + m03;
        sp.y = m10*x + m11*y + m12*z + m13;
        sp.z = m20*x + m21*y + m22*z + m23;
        sp.nx = m00*nx + m01*ny + m02*nz;
        sp.ny = m10*nx + m11*ny + m12*nz;
        sp.nz = m20*nx + m21*ny + m22*nz;
    }


    if(s_nrp == 0){spoints = dpoints;}
    else{
        double * sp = new double[3*s_nrp];
        for(unsigned int i = 0; i < s_nrp; i++){
            superpoint & p = spoints[i];
            sp[3*i+0]   = p.x;
            sp[3*i+1]   = p.y;
            sp[3*i+2]   = p.z;
        }

        ArrayData3D<double> * a3d = new ArrayData3D<double>;
        a3d->data	= sp;
        a3d->rows	= s_nrp;
        Tree3d * trees3d	= new Tree3d(3, *a3d, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        trees3d->buildIndex();
//        //keypoints = kps;

        int capacity = 1;
        size_t * ret_indexes = new size_t[capacity];
        double * out_dists_sqr = new double[capacity];
        nanoflann::KNNResultSet<double> resultSet(capacity);
        std::vector<int> matches;
        matches.resize(d_nrp);

        std::vector<double> residuals;
        double stdval = 0;
        double tdp [3];
        for(unsigned int i = 0; i < d_nrp; i++){
            superpoint & dp = dpoints[i];
            tdp[0] = dp.x;
            tdp[1] = dp.y;
            tdp[2] = dp.z;

            resultSet.init(ret_indexes, out_dists_sqr);
            trees3d->findNeighbors(resultSet,tdp, nanoflann::SearchParams(10));
            unsigned int id = ret_indexes[0];

            matches[i] = id;

            superpoint & skp = spoints[id];

            double rangew = sqrt( 1.0/(1.0/skp.point_information +1.0/dp.point_information) );

            double dx = rangew * (tdp[0]-sp[3*id+0]);
            double dy = rangew * (tdp[1]-sp[3*id+1]);
            double dz = rangew * (tdp[2]-sp[3*id+2]);

            residuals.push_back(dx);
            residuals.push_back(dy);
            residuals.push_back(dz);
            stdval += dx*dx+dy*dy+dz*dz;
        }


        stdval = sqrt(stdval/double(d_nrp*3 -1 ));

        DistanceWeightFunction2PPR2 * func = new DistanceWeightFunction2PPR2();
        func->zeromean				= true;
        func->maxp					= 0.99;
        func->startreg				= 0.0005;
        func->debugg_print			= false;
        func->maxd					= 0.1;
        func->startmaxd				= func->maxd;
        func->histogram_size		= 100;
        func->starthistogram_size	= func->histogram_size;
        func->stdval2				= stdval;
        func->maxnoise				= stdval;
        func->reset();
        ((DistanceWeightFunction2*)func)->computeModel(residuals);

        printf("===================================\n");
        printf("before s_nrp: %i d_nrp.size(): %i\n",s_nrp,d_nrp);
        for(unsigned int i = 0; i < d_nrp; i++){
            superpoint & sp1 = dpoints[i];
            superpoint & sp2 = spoints[matches[i]];

            double probX = func->getProb(residuals[3*i+0]);
            double probY = func->getProb(residuals[3*i+1]);
            double probZ = func->getProb(residuals[3*i+2]);
            double prob = probX*probY*probZ/(probX*probY*probZ + (1-probX)*(1-probY)*(1-probZ));

            if(prob > 0.5){
                sp2.merge(sp1);
            }else{
                spoints.push_back(sp1);
            }

        }
        printf("after s_nrp: %i\n",spoints.size());

        delete   func;
        delete[] out_dists_sqr;
        delete[] ret_indexes;
        delete   trees3d;
        delete   a3d;
        delete[] sp;
    }
}

void Model::save(std::string path){
	char buf [1024];
	struct stat sb;

	if (stat(path.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)){
		sprintf(buf,"rm -r %s",path.c_str());
		system(buf);
	}

	sprintf(buf,"mkdir %s",path.c_str());
	system(buf);

	ofstream posesfile;
	posesfile.open (path+"/relativeposes.txt");
	for(unsigned int f = 0; f < frames.size(); f++){
		posesfile << relativeposes[f] << endl <<endl;

		sprintf(buf,"%s/frame_%08i",path.c_str(),f);
		frames[f]->save(std::string(buf));

		sprintf(buf,"%s/frame_%08i.pcd",path.c_str(),f);
		frames[f]->savePCD(std::string(buf),relativeposes[f]);
		sprintf(buf,"%s/modelmask_%08i.png",path.c_str(),f);
		cv::imwrite( buf, modelmasks[f]->getMask() );
	}
    posesfile.close();

	ofstream submodels_posesfile;
	submodels_posesfile.open (path+"/submodels_relativeposes.txt");
	for(unsigned int i = 0; i < submodels.size(); i++){
		submodels_posesfile << submodels_relativeposes[i] << endl <<endl;
		sprintf(buf,"%s/submodel%08i",path.c_str(),i);
		submodels[i]->save(std::string(buf));
	}
	submodels_posesfile.close();
}

void Model::saveFast(std::string path){
    //printf("Model::saveFast(%s)\n",path.c_str());

	double startTime = getTime();
	pointspath = path+"points.bin";

	long sizeofSuperPoint = 3*(3+1);
	unsigned int nr_points = points.size();
	float * data = new float[nr_points * sizeofSuperPoint];
	long count = 0;

	for(unsigned long i = 0; i < nr_points; i++){
		reglib::superpoint & p = points[i];
		data[count++] = p.x;
		data[count++] = p.y;
		data[count++] = p.z;
		data[count++] = p.nx;
		data[count++] = p.ny;
		data[count++] = p.nz;
		data[count++] = p.r;
		data[count++] = p.g;
		data[count++] = p.b;
		data[count++] = p.point_information;
		data[count++] = p.normal_information;
		data[count++] = p.colour_information;
	}

	std::ofstream pointfile;
	pointfile.open(pointspath, ios::out | ios::binary);
	if(nr_points > 0){
        //printf("saving %i points\n",nr_points);
		pointfile.write( (char*)data, nr_points*sizeofSuperPoint*sizeof(float));
	}
	pointfile.close();

	delete[] data;
//printf("saveFast(%s): %5.5fs\n",path.c_str(),getTime()-startTime);

	for(unsigned int i = 0; i < modelmasks.size();i++){
		char buf [1024];
		sprintf(buf,"%smodelmask_%10.10i_",path.c_str(),i);
		modelmasks[i]->saveFast(std::string(buf));
	}

	std::vector<std::string> spms;
	for(unsigned int k = 0; k < frames.size();k++){
		std::string spm = modelmasks[k]->savepath;
		spm = spm.substr(path.length(),spm.length());
		spms.push_back(spm);
	}

	std::vector<std::string> rep_spms;
	for(unsigned int k = 0; k < rep_frames.size();k++){
		std::string spm = rep_modelmasks[k]->savepath;
		spm = spm.substr(path.length(),spm.length());
		rep_spms.push_back(spm);
	}

	unsigned long buffersize = 2*sizeof(double)+10*sizeof(unsigned long)+keyval.length()+soma_id.length()+retrieval_object_id.length()+retrieval_vocabulary_id.length();

	buffersize += sizeof(unsigned long);
	buffersize += scores.size()*sizeof(unsigned long);
	for(unsigned int k = 0; k < scores.size();k++){
		buffersize += scores[k].size()*sizeof(double);
	}

	buffersize += sizeof(unsigned long);
	buffersize += submodels_scores.size()*sizeof(unsigned long);
	for(unsigned int k = 0; k < submodels_scores.size();k++){
		buffersize += submodels_scores[k].size()*sizeof(double);
	}

	if(parrent != 0){buffersize += parrent->keyval.length();}

	for(unsigned int k = 0; k < frames.size();k++){
		buffersize += 2*sizeof(unsigned long)+frames[k]->keyval.length()+spms[k].length();
		buffersize += 16*sizeof(double);
	}

	for(unsigned int k = 0; k < rep_frames.size();k++){
		buffersize += 2*sizeof(unsigned long)+rep_frames[k]->keyval.length()+rep_spms[k].length();
		buffersize += 16*sizeof(double);
	}

	buffersize += sizeof(unsigned long);
	for(unsigned int k = 0; k < submodels.size();k++){
		buffersize += sizeof(unsigned long)+submodels[k]->keyval.length();
		buffersize += 16*sizeof(double);
	}

	char* buffer = new char[buffersize];
	double * buffer_double = (double *)buffer;
	unsigned long * buffer_long = (unsigned long *)buffer;

	int counter = 0;
    if(parrent != 0){	buffer_long[counter++] = parrent->keyval.length();}
    else{				buffer_long[counter++] = 0;}

	buffer_double[counter++] = score;
	buffer_double[counter++] = total_scores;
	buffer_long[counter++] = id;
	buffer_long[counter++] = last_changed;
	buffer_long[counter++] = keyval.length();
	buffer_long[counter++] = soma_id.length();
	buffer_long[counter++] = pointspath.length();
	buffer_long[counter++] = retrieval_object_id.length();
	buffer_long[counter++] = retrieval_vocabulary_id.length();
	buffer_long[counter++] = frames.size();

    //printf("frames: %i\n",frames.size());
    buffer_long[counter++] = rep_frames.size();
    //printf("rep_frames: %i\n",rep_frames.size());

	buffer_long[counter++] = scores.size();
	for(unsigned int i = 0; i < scores.size();i++){
		buffer_long[counter++] = scores[i].size();
		for(unsigned int j = 0; j < scores[i].size();j++){
			buffer_double[counter++] = scores[i][j];
		}
	}

	buffer_long[counter++] = submodels_scores.size();
	for(unsigned int i = 0; i < submodels_scores.size();i++){
		buffer_long[counter++] = submodels_scores[i].size();
		for(unsigned int j = 0; j < submodels_scores[i].size();j++){
			buffer_double[counter++] = submodels_scores[i][j];
		}
	}

	buffer_long[counter++] = submodels.size();
	for(unsigned int k = 0; k < submodels.size();k++){
		Eigen::Matrix4d pose = submodels_relativeposes[k];
		buffer_long[counter++] = submodels[k]->keyval.length();
		for(int i = 0; i < 4; i++){
			for(int j = 0; j < 4; j++){
				buffer_double[counter++] = pose(i,j);
			}
		}
	}

	for(unsigned int k = 0; k < frames.size();k++){
		Eigen::Matrix4d pose = relativeposes[k];
		buffer_long[counter++] = frames[k]->keyval.length();
		buffer_long[counter++] = spms[k].length();
		for(int i = 0; i < 4; i++){
			for(int j = 0; j < 4; j++){
				buffer_double[counter++] = pose(i,j);
			}
		}
	}

	for(unsigned int k = 0; k < rep_frames.size();k++){
		Eigen::Matrix4d pose = rep_relativeposes[k];
		buffer_long[counter++] = rep_frames[k]->keyval.length();
		buffer_long[counter++] = rep_spms[k].length();
		for(int i = 0; i < 4; i++){
			for(int j = 0; j < 4; j++){
				buffer_double[counter++] = pose(i,j);
			}
		}
	}

	unsigned int count4 = sizeof(double)*counter;

	for(unsigned int i = 0; i < keyval.length();i++){
		buffer[count4++] = keyval[i];
	}

	for(unsigned int i = 0; i < soma_id.length();i++){
		buffer[count4++] = soma_id[i];
	}


	for(unsigned int i = 0; i < retrieval_object_id.length();i++){
		buffer[count4++] = retrieval_object_id[i];
	}


	for(unsigned int i = 0; i < retrieval_vocabulary_id.length();i++){
		buffer[count4++] = retrieval_vocabulary_id[i];
	}

	if(parrent != 0){
		for(unsigned int i = 0; i < parrent->keyval.length();i++){
			buffer[count4++] = parrent->keyval[i];
		}
	}

	for(unsigned int k = 0; k < submodels.size();k++){
		for(unsigned int i = 0; i < submodels[k]->keyval.length(); i++){
			buffer[count4++] = submodels[k]->keyval[i];
		}
	}

	for(unsigned int k = 0; k < frames.size();k++){
		for(unsigned int i = 0; i < frames[k]->keyval.length();i++){
			buffer[count4++] = frames[k]->keyval[i];
		}
		for(unsigned int i = 0; i < spms[k].length();i++){
			buffer[count4++] = spms[k][i];
		}
	}

	for(unsigned int k = 0; k < rep_frames.size();k++){
		for(unsigned int i = 0; i < rep_frames[k]->keyval.length();i++){
			buffer[count4++] = rep_frames[k]->keyval[i];
		}
		for(unsigned int i = 0; i < rep_spms[k].length();i++){
			buffer[count4++] = rep_spms[k][i];
		}
	}

	std::ofstream outfile (path+"data.bin",std::ofstream::binary);
	outfile.write (buffer,buffersize);
	outfile.close();
	delete[] buffer;

	//printf("saveFast(%s): %5.5fs\n",path.c_str(),getTime()-startTime);
}

Model * Model::loadFast(std::string path){
    //printf("++++++++++++++++++++++++++++++++++++++++++++++++\n");
	//printf("Model::loadFast(%s)\n",path.c_str());
	double startTime = getTime();
	Model * mod = new Model();
	mod->pointspath = path+"points.bin";
	//printf("Model::loadFast pointspath = %s\n",mod->pointspath.c_str());

	std::ifstream pointfile (mod->pointspath, std::ios::in | std::ios::binary | std::ios::ate);
	if (pointfile.is_open()){
		unsigned long size = pointfile.tellg();
		char * buffer = new char [size];
		pointfile.seekg (0, std::ios::beg);
		pointfile.read (buffer, size);
		pointfile.close();
		float *		data	= (float *)buffer;
		long sizeofSuperPoint = 3*(3+1);

		unsigned int nr_points = size/(sizeofSuperPoint*sizeof(float));
		std::vector<superpoint> points;
		points.resize(nr_points);

		long count = 0;

		for(unsigned long i = 0; i < nr_points; i++){
            reglib::superpoint & p = points[i];
			p.x = data[count++];
			p.y = data[count++];
			p.z = data[count++];
			p.nx = data[count++];
			p.ny = data[count++];
			p.nz = data[count++];
			p.r = data[count++];
			p.g = data[count++];
			p.b = data[count++];
			p.point_information = data[count++];
			p.normal_information = data[count++];
			p.colour_information = data[count++];
		}

		delete[] buffer;
		mod->points = points;
	}
    //printf("time to load points: %5.5fs\n",getTime()-startTime);

	//printf("Model::loadFast points.size  = %i\n",mod->points.size());

	std::ifstream file (path+"data.bin", std::ios::in | std::ios::binary | std::ios::ate);
	if (file.is_open()){
		unsigned long size = file.tellg();
		char * buffer = new char [size];
		file.seekg (0, std::ios::beg);
		file.read (buffer, size);
		file.close();
		double * buffer_double = (double *)buffer;
		unsigned long * buffer_long = (unsigned long *)buffer;
		//printf("Model::loadFast size  = %i\n",size);


		int counter = 0;
        unsigned long parrent_keyvallength = buffer_long[counter++];
		mod->score = buffer_double[counter++];
		mod->total_scores = buffer_double[counter++];
		mod->id = buffer_long[counter++];
		mod->last_changed = buffer_long[counter++];
		unsigned long keyvallength = buffer_long[counter++];
		unsigned long soma_idlength = buffer_long[counter++];
		unsigned long pointspathlength = buffer_long[counter++];
		unsigned long retrieval_object_idlength = buffer_long[counter++];
		unsigned long retrieval_vocabulary_idlength = buffer_long[counter++];

		unsigned long framessize = buffer_long[counter++];
		unsigned long rep_framessize = buffer_long[counter++];
		//printf("nr frames: %i\n",framessize);
        //printf("nr rep_frames: %i\n",framessize);
		mod->scores.resize(buffer_long[counter++]);
		for(unsigned int i = 0; i < mod->scores.size();i++){
			mod->scores[i].resize(buffer_long[counter++]);
			for(unsigned int j = 0; j < mod->scores[i].size();j++){
				mod->scores[i][j] = buffer_double[counter++];
			}
		}

		mod->submodels_scores.resize(buffer_long[counter++]);
		for(unsigned int i = 0; i < mod->submodels_scores.size();i++){
			mod->submodels_scores[i].resize(buffer_long[counter++]);
			for(unsigned int j = 0; j < mod->submodels_scores[i].size();j++){
				mod->submodels_scores[i][j] = buffer_double[counter++];
			}
		}

		unsigned int submodelssize = buffer_long[counter++];
		mod->submodels_relativeposes.resize(submodelssize);
		std::vector<unsigned int> submodels_keyvallength;
		for(unsigned int k = 0; k < submodelssize;k++){
			Eigen::Matrix4d & pose = mod->submodels_relativeposes[k];
			submodels_keyvallength.push_back(buffer_long[counter++]);
			for(int i = 0; i < 4; i++){
				for(int j = 0; j < 4; j++){
					pose(i,j) = buffer_double[counter++];
				}
			}
        }

		std::vector<unsigned int> frames_keyvallength;
		std::vector<unsigned int> spms_length;
		mod->relativeposes.resize(framessize);
		for(unsigned int k = 0; k < framessize;k++){
			Eigen::Matrix4d & pose = mod->relativeposes[k];
			frames_keyvallength.push_back(buffer_long[counter++]);
			spms_length.push_back(buffer_long[counter++]);
			for(int i = 0; i < 4; i++){
				for(int j = 0; j < 4; j++){
					pose(i,j) = buffer_double[counter++];
				}
			}
		}

		std::vector<unsigned int> rep_frames_keyvallength;
		std::vector<unsigned int> rep_spms_length;
		mod->rep_relativeposes.resize(rep_framessize);
		for(unsigned int k = 0; k < rep_framessize;k++){
			Eigen::Matrix4d & pose = mod->rep_relativeposes[k];
			rep_frames_keyvallength.push_back(buffer_long[counter++]);
			rep_spms_length.push_back(buffer_long[counter++]);
			for(int i = 0; i < 4; i++){
				for(int j = 0; j < 4; j++){
					pose(i,j) = buffer_double[counter++];
				}
			}
		}

		unsigned int count4 = sizeof(double)*counter;
		mod->keyval.resize(keyvallength);
		for(unsigned int i = 0; i < keyvallength;i++){
			mod->keyval[i] = buffer[count4++];
		}

		mod->soma_id.resize(soma_idlength);
		for(unsigned int i = 0; i < soma_idlength;i++){
			mod->soma_id[i] = buffer[count4++];
		}

		mod->retrieval_object_id.resize(retrieval_object_idlength);
		for(unsigned int i = 0; i < retrieval_object_idlength;i++){
			mod->retrieval_object_id[i] = buffer[count4++];
		}

		mod->retrieval_vocabulary_id.resize(retrieval_vocabulary_idlength);
		for(unsigned int i = 0; i < retrieval_vocabulary_idlength;i++){
			mod->retrieval_vocabulary_id[i] = buffer[count4++];
		}

		std::string parrent_keyval = "";
		parrent_keyval.resize(parrent_keyvallength);
		for(unsigned int i = 0; i < parrent_keyvallength;i++){
			parrent_keyval[i] = buffer[count4++];
		}
		//printf("parrent_keyval: %s\n",parrent_keyval.c_str());

		mod->submodels.resize(submodelssize);
		for(unsigned int k = 0; k < submodelssize;k++){
			std::string submodels_keyval = "";
			submodels_keyval.resize(submodels_keyvallength[k]);
			for(unsigned int i = 0; i < submodels_keyvallength[k]; i++){
				submodels_keyval[i] = buffer[count4++];
			}
			//printf("submodels_keyval[%i] = %s\n",k,submodels_keyval.c_str());
			mod->submodels[k] = Model::loadFast(path+"/../"+submodels_keyval+"/");
            mod->submodels[k]->parrent = mod;
		}

		mod->frames.resize(framessize);
		mod->modelmasks.resize(framessize);
        //printf("framessize: %i\n",framessize);
		for(unsigned int k = 0; k < framessize;k++){
            ////printf("---------------------------\n");
			std::string frames_keyval;
			frames_keyval.resize(frames_keyvallength[k]);
			for(unsigned int i = 0; i < frames_keyvallength[k];i++){
				frames_keyval[i] = buffer[count4++];
			}
			mod->frames[k] = RGBDFrame::loadFast(path+"/../frames/"+frames_keyval);
			std::string spms;
			spms.resize(spms_length[k]);
			for(unsigned int i = 0; i < spms_length[k];i++){
				spms[i] = buffer[count4++];
			}
			mod->modelmasks[k] = ModelMask::loadFast(path+spms);

            //mod->frames[k]->show(true);
		}
		//printf("---------------------------\n");

		//Set up to use already loaded frames
        //printf("rep_framessize: %i\n",rep_framessize);
		for(unsigned int k = 0; k < rep_framessize;k++){
			std::string rep_frames_keyval;
			rep_frames_keyval.resize(rep_frames_keyvallength[k]);
			for(unsigned int i = 0; i < rep_frames_keyvallength[k];i++){
				rep_frames_keyval[i] = buffer[count4++];
			}
			std::string rep_spm;
			rep_spm.resize(rep_spms_length[k]);
			for(unsigned int i = 0; i < rep_spms_length[k];i++){
				rep_spm[i] = buffer[count4++];
			}
            RGBDFrame * frame = 0;
            ModelMask * modelmask = 0;
            mod->getRepFrame(frame,modelmask,rep_frames_keyval);
            mod->rep_frames.push_back(frame);
            mod->rep_modelmasks.push_back(modelmask);
            //frame->show(true);
		}
	}
	//if(mod->parrent == 0){printf("Model::loadFast(%s): %7.7fs\n",path.c_str(),getTime()-startTime);}
	return mod;
}

Model * Model::load(Camera * cam, std::string path){
	printf("Model * Model::load(Camera * cam, std::string path)\n");
	std::streampos size;
	char * buffer;
	char buf [1024];
	std::string datapath = path+"/data.txt";
	std::ifstream file (datapath, std::ios::in | std::ios::binary | std::ios::ate);
	if (file.is_open()){
		size = file.tellg();
		buffer = new char [size];
		file.seekg (0, std::ios::beg);
		file.read (buffer, size);
		file.close();

		Model * mod = new Model();
		double *		buffer_double	= (double *)buffer;
		unsigned long * buffer_long		= (unsigned long *)buffer;
		//		std::vector<ModelMask*> modelmasks;

		int counter = 0;
		unsigned int nr_frames = buffer_long[counter++];
		mod->score = buffer_double[counter++];
		for(unsigned int f = 0; f < nr_frames; f++){
			Eigen::Matrix4d pose;
			for(int i = 0; i < 4; i++){
				for(int j = 0; j < 4; j++){
					pose(i,j) = buffer_double[counter++];
				}
			}

			sprintf(buf,"%s/frame_%i",path.c_str(),int(f));
			RGBDFrame * frame = RGBDFrame::load(cam, std::string(buf));

			sprintf(buf,"%s/modelmask_%i.png",path.c_str(),int(f));
			cv::Mat mask = cv::imread(buf, -1);   // Read the file

			mod->relativeposes.push_back(pose);
			mod->frames.push_back(frame);
			mod->modelmasks.push_back(new ModelMask(mask));
		}

		mod->total_scores = buffer_double[counter++];
		mod->scores.resize(nr_frames);
		for(unsigned int f1 = 0; f1 < nr_frames; f1++){
			mod->scores[f1].resize(nr_frames);
			for(unsigned int f2 = 0; f2 < nr_frames; f2++){
				mod->scores[f1][f2] = buffer_double[counter++];
			}
		}

		for(unsigned int f = 0; f < nr_frames; f++){
			mod->modelmasks[f]->sweepid = buffer_long[counter++];
			printf("modelmask sweepid: %i\n",int(mod->modelmasks[f]->sweepid));
		}

		mod->recomputeModelPoints();
		delete[] buffer;
		return mod;
	}else{std::cout << "Unable to open model file " << datapath << std::endl; exit(0);}
	return 0;
}

void Model::getRepFrame(RGBDFrame * & frame, ModelMask * & modelmask, std::string keyval){
    if(keyval.length() == 0){return;}
    for(unsigned int i = 0; i < frames.size(); i++){
        if(frames[i]->keyval.compare(keyval) == 0){
            frame = frames[i];
            modelmask = modelmasks[i];
            return;
        }
    }

    for(unsigned int i = 0; i < submodels.size(); i++){
        RGBDFrame * fr = 0;
        ModelMask * mm = 0;
        submodels[i]->getRepFrame(fr,mm,keyval);
        if(fr != 0){
            frame = fr;
            modelmask = mm;
            return;
        }
    }
    return;
}


}

