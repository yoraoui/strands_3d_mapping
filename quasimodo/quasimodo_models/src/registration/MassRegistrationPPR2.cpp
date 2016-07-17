#include "registration/MassRegistrationPPR2.h"

#include <iostream>
#include <fstream>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace reglib
{

MassRegistrationPPR2::MassRegistrationPPR2(double startreg, bool visualize){
	type					= PointToPlane;
	//type					= PointToPoint;
	use_PPR_weight			= true;
	use_features			= true;
	normalize_matchweights	= true;


	use_surface = true;
	use_depthedge = true;

	DistanceWeightFunction2PPR2 * dwf = new DistanceWeightFunction2PPR2();
	dwf->update_size		= true;
	dwf->startreg			= startreg;
	dwf->debugg_print		= false;
	func					= dwf;


	DistanceWeightFunction2PPR2 * kpdwf = new DistanceWeightFunction2PPR2();
	kpdwf->update_size		= true;
	kpdwf->startreg			= startreg;
	kpdwf->debugg_print		= false;
	kpfunc					= kpdwf;

	DistanceWeightFunction2PPR2 * depthedge_dwf = new DistanceWeightFunction2PPR2();
	depthedge_dwf->update_size		= true;
	depthedge_dwf->startreg			= startreg;
	depthedge_dwf->debugg_print		= false;
	depthdege_func					= depthedge_dwf;

	fast_opt				= true;

	nomask = true;
	maskstep = 1;
	nomaskstep = 100000;

	stopval = 0.001;
	steps = 4;

	timeout = 6000;

	if(visualize){
		visualizationLvl = 1;
		viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer ("3D Viewer"));
		viewer->setBackgroundColor (0, 0, 0);
		viewer->addCoordinateSystem (1.0);
		viewer->initCameraParameters ();
	}else{
		visualizationLvl = 0;
	}


	Qp_arr = new double[3*maxcount+0];
	Qn_arr = new double[3*maxcount+0];
	Xp_arr = new double[3*maxcount+0];
	Xn_arr = new double[3*maxcount+0];
	rangeW_arr = new double[maxcount+0];


	kp_Qp_arr = new double[3*maxcount+0];
	kp_Qn_arr = new double[3*maxcount+0];
	kp_Xp_arr = new double[3*maxcount+0];
	kp_Xn_arr = new double[3*maxcount+0];
	kp_rangeW_arr = new double[maxcount+0];

	depthedge_Qp_arr = new double[3*maxcount+0];
	depthedge_Xp_arr = new double[3*maxcount+0];
	depthedge_rangeW_arr = new double[maxcount+0];
}
MassRegistrationPPR2::~MassRegistrationPPR2(){

	for(unsigned int i = 0; i < arraypoints.size(); i++){delete[] arraypoints[i];}
	for(unsigned int i = 0; i < arraynormals.size(); i++){delete[] arraynormals[i];}
	for(unsigned int i = 0; i < arraycolors.size(); i++){delete[] arraycolors[i];}
	for(unsigned int i = 0; i < arrayinformations.size(); i++){delete[] arrayinformations[i];}

	for(unsigned int i = 0; i < trees3d.size(); i++){delete trees3d[i];}
	for(unsigned int i = 0; i < a3dv.size(); i++){delete a3dv[i];}

	delete func;
	delete kpfunc;

	delete[] Qp_arr;
	delete[] Qn_arr;
	delete[] Xp_arr;
	delete[] Xn_arr;
	delete[] rangeW_arr;

	delete[] kp_Qp_arr;
	delete[] kp_Qn_arr;
	delete[] kp_Xp_arr;
	delete[] kp_Xn_arr;
	delete[] kp_rangeW_arr;
}

void MassRegistrationPPR2::addModel(Model * model){

    double total_load_time_start = getTime();


    nr_matches.push_back(	0);
    matchids.push_back(		std::vector< std::vector<int> >() );
    matchdists.push_back(	std::vector< std::vector<double> >() );
    nr_datas.push_back(		0);

    points.push_back(				Eigen::Matrix<double, 3, Eigen::Dynamic>());
    colors.push_back(				Eigen::Matrix<double, 3, Eigen::Dynamic>());
    normals.push_back(				Eigen::Matrix<double, 3, Eigen::Dynamic>());
    transformed_points.push_back(	Eigen::Matrix<double, 3, Eigen::Dynamic>());
    transformed_normals.push_back(	Eigen::Matrix<double, 3, Eigen::Dynamic>());

    informations.push_back(			Eigen::VectorXd());

    nr_arraypoints.push_back(0);
    arraypoints.push_back(0);
    arraynormals.push_back(0);
    arraycolors.push_back(0);
    arrayinformations.push_back(0);
    trees3d.push_back(0);
    a3dv.push_back(0);

    depthedge_nr_arraypoints.push_back(0);
    depthedge_arraypoints.push_back(0);
    depthedge_arrayinformations.push_back(0);
    depthedge_trees3d.push_back(0);
    depthedge_a3dv.push_back(0);

	int step = maskstep*maskstep;

    is_ok.push_back(false);


	int count = model->points.size()/step;
	int i = points.size()-1;
    if(count < 10){
        is_ok[i] = false;
        return;
    }else{
        is_ok[i] = true;
    }

//    printf("%i is ok %i\n",i,is_ok[i]);

    nr_datas[i] = count;
    points[i].resize(Eigen::NoChange,count);
    colors[i].resize(Eigen::NoChange,count);
    normals[i].resize(Eigen::NoChange,count);
    transformed_points[i].resize(Eigen::NoChange,count);
    transformed_normals[i].resize(Eigen::NoChange,count);

    double * ap = new double[3*count];
    double * an = new double[3*count];
    double * ac = new double[3*count];
    double * ai = new double[3*count];
    nr_arraypoints[i] = count;
    arraypoints[i] = ap;
    arraynormals[i] = an;
    arraycolors[i] = ac;
    arrayinformations[i] = ai;

    Eigen::Matrix<double, 3, Eigen::Dynamic> & X	= points[i];
    Eigen::Matrix<double, 3, Eigen::Dynamic> & C	= colors[i];
    Eigen::Matrix<double, 3, Eigen::Dynamic> & Xn	= normals[i];
    Eigen::Matrix<double, 3, Eigen::Dynamic> & tX	= transformed_points[i];
    Eigen::Matrix<double, 3, Eigen::Dynamic> & tXn	= transformed_normals[i];
    Eigen::VectorXd information (count);

	for(unsigned int c = 0; c < count; c++){

		ap[3*c+0] = model->points[c*step].point(0);
		ap[3*c+1] = model->points[c*step].point(1);
		ap[3*c+2] = model->points[c*step].point(2);

		an[3*c+0] = model->points[c*step].normal(0);
		an[3*c+1] = model->points[c*step].normal(1);
		an[3*c+2] = model->points[c*step].normal(2);

		ac[3*c+0] =model->points[c*step].feature(0);
		ac[3*c+1] =model->points[c*step].feature(1);
		ac[3*c+2] =model->points[c*step].feature(2);

		ai[c] = 1.0/model->points[c*step].point_information;//1.0/(z*z);

		X(0,c)	= ap[3*c+0];
		X(1,c)	= ap[3*c+1];
		X(2,c)	= ap[3*c+2];
		Xn(0,c)	= an[3*c+0];
		Xn(1,c)	= an[3*c+1];
		Xn(2,c)	= an[3*c+2];

		information(c) = ai[c];//1.0/(z*z);
		C(0,c) = ac[3*c+0];
		C(1,c) = ac[3*c+1];
		C(2,c) = ac[3*c+2];
	}

    informations[i] = information;

    ArrayData3D<double> * a3d = new ArrayData3D<double>;
    a3d->data	= ap;
    a3d->rows	= count;
    a3dv[i]		= a3d;
    trees3d[i]	= new Tree3d(3, *a3d, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    trees3d[i]->buildIndex();

/*
    int depthedge_count = 0;
    for(unsigned int w = 0; w < width; w++){
        for(unsigned int h = 0; h < height; h++){
            int ind = h*width+w;
            if(maskvec[ind] && edgedata[ind] == 255){
                float z = idepth*float(depthdata[ind]);
                if(z > 0.2){depthedge_count++;}
            }
        }
    }

    if(depthedge_count < 10){
        double * depthedge_ap = new double[3*depthedge_count];
        double * depthedge_ai = new double[3*depthedge_count];
        depthedge_nr_arraypoints[i] = depthedge_count;
        depthedge_arraypoints[i] = depthedge_ap;
        depthedge_arrayinformations[i] = depthedge_ai;

        c = 0;
        for(unsigned int w = 0; w < width; w++){
            for(unsigned int h = 0; h < height; h++){
                if(c == depthedge_count){continue;}
                int ind = h*width+w;
                if(maskvec[ind] && edgedata[ind] == 255){
                    float z = idepth*float(depthdata[ind]);
                    if(z > 0.2){
                        depthedge_ap[3*c+0] = (w - cx) * z * ifx;
                        depthedge_ap[3*c+1] = (h - cy) * z * ify;;
                        depthedge_ap[3*c+2] = z;
                        depthedge_ai[c] = pow(fabs(z),-2);//1.0/(z*z);
                        c++;
                    }
                }
            }
        }

        ArrayData3D<double> * depthedge_a3d = new ArrayData3D<double>;
        depthedge_a3d->data					= depthedge_ap;
        depthedge_a3d->rows					= depthedge_count;
        depthedge_a3dv[i]					= depthedge_a3d;
        depthedge_trees3d[i]				= new Tree3d(3, *depthedge_a3d, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        depthedge_trees3d[i]->buildIndex();
    }
*/
    printf("total load time:          %5.5f\n",getTime()-total_load_time_start);
}

void MassRegistrationPPR2::addModelData(Model * model_, bool submodels){
	model = model_;
	printf("addModelData\n");

	if(submodels){
		for(unsigned int i = 0; i < model->submodels.size(); i++){
			addData(model->submodels[i]->getPCLnormalcloud(1,false));
		}
	}else{
        //setData(model->frames,model->modelmasks);

        for(unsigned int i = 0; i < model->submodels.size(); i++){
            addData(model->submodels[i]->getPCLnormalcloud(1,false));
//            kp_nr_arraypoints.push_back(0);
//            kp_arraypoints.push_back(0);
//            kp_arraynormals.push_back(0);
//            kp_arrayinformations.push_back(0);
//            kp_arraydescriptors.push_back(0);
//            frameid.push_back(-1);
        }

		unsigned int nr_frames = model->frames.size();

		for(unsigned int i = 0; i < nr_frames; i++){
//            frameid.push_back(i);
            addData(model->frames[i], model->modelmasks[i]);

//			RGBDFrame* frame = model->frames[i];
//			std::vector<cv::KeyPoint> & keypoints = model->all_keypoints[i];
//			cv::Mat & descriptors = model->all_descriptors[i];
//			//uint64_t * descriptorsdata		= (uint64_t			*)(descriptors.data);


//			float		   * normalsdata	= (float			*)(frame->normals.data);
//			unsigned short * depthdata		= (unsigned short	*)(frame->depth.data);

//			Camera * camera				= frame->camera;
//			const unsigned int width	= camera->width;
//			const unsigned int height	= camera->height;
//			const float idepth			= camera->idepth_scale;
//			const float cx				= camera->cx;
//			const float cy				= camera->cy;
//			const float ifx				= 1.0/camera->fx;
//			const float ify				= 1.0/camera->fy;

//			unsigned int nr_kp = keypoints.size();
//			if(nr_kp){
//				double * ap = new double[3*nr_kp];
//				double * an = new double[3*nr_kp];
//				double * ai = new double[nr_kp];
//				for(unsigned int k = 0; k < nr_kp; k++){
//					cv::KeyPoint & kp = keypoints[k];
//					double w = kp.pt.x;
//					double h = kp.pt.y;
//					int ind = int(h+0.5)*width+int(w+0.5);

//					float z = idepth*float(depthdata[ind]);
//					float x = (w - cx) * z * ifx;
//					float y = (h - cy) * z * ify;

//					ap[3*k+0] =x;
//					ap[3*k+1] =y;
//					ap[3*k+2] =z;

//					an[3*k+0] = normalsdata[3*ind+0];
//					an[3*k+1] = normalsdata[3*ind+1];
//					an[3*k+2] = normalsdata[3*ind+2];

//					ai[k] = pow(fabs(z),-2);
//				}

//				kp_nr_arraypoints.push_back(nr_kp);
//				kp_arraypoints.push_back(ap);
//				kp_arraynormals.push_back(an);
//				kp_arrayinformations.push_back(ai);
//				kp_arraydescriptors.push_back((uint64_t			*)(descriptors.data));
//			}else{
//				kp_nr_arraypoints.push_back(0);
//				kp_arraypoints.push_back(0);
//				kp_arraynormals.push_back(0);
//				kp_arrayinformations.push_back(0);
//				kp_arraydescriptors.push_back(0);
//			}
		}

//        for(int i = 0; i < is_ok.size(); i++){
//            if(is_ok[i]){printf("%i is ok\n",i);
//            }else{printf("%i is not ok\n",i);}
//        }


	}
}


void MassRegistrationPPR2::setData(std::vector< pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr > all_clouds){
	double total_load_time_start = getTime();
	unsigned int nr_frames = all_clouds.size();

	if(arraypoints.size() > 0){
		for(unsigned int i = 0; i < arraypoints.size(); i++){delete[] arraypoints[i];}
		arraypoints.resize(0);
	}

	if(a3dv.size() > 0){
		for(unsigned int i = 0; i < a3dv.size(); i++){delete a3dv[i];}
		a3dv.resize(0);
	}

	if(trees3d.size() > 0){
		for(unsigned int i = 0; i < trees3d.size(); i++){delete trees3d[i];}
		trees3d.resize(0);
	}

    nr_matches.resize(nr_frames);
    matchids.resize(nr_frames);
    matchdists.resize(nr_frames);
	nr_datas.resize(nr_frames);
	points.resize(nr_frames);
	colors.resize(nr_frames);
	normals.resize(nr_frames);
	transformed_points.resize(nr_frames);
	transformed_normals.resize(nr_frames);
	informations.resize(nr_frames);

	nr_arraypoints.resize(nr_frames);

	arraypoints.resize(nr_frames);
	arraynormals.resize(nr_frames);
	arraycolors.resize(nr_frames);
	arrayinformations.resize(nr_frames);

	trees3d.resize(nr_frames);
	a3dv.resize(nr_frames);
	is_ok.resize(nr_frames);

	for(unsigned int i = 0; i < nr_frames; i++){
		//printf("loading data for %i\n",i);
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud = all_clouds[i];
		int count = 0;
		for(unsigned int i = 0; i < cloud->points.size(); i++){
			if (isValidPoint(cloud->points[i])){count++;}
		}

		if(count < 10){
			is_ok[i] = false;
			continue;
		}else{
			is_ok[i] = true;
		}

		double * ap = new double[3*count];
		double * an = new double[3*count];
		double * ac = new double[3*count];
		double * ai = new double[3*count];

        nr_datas[i] = count;
        matchids[i].resize(nr_frames);
        matchdists[i].resize(nr_frames);
		points[i].resize(Eigen::NoChange,count);
		colors[i].resize(Eigen::NoChange,count);
		normals[i].resize(Eigen::NoChange,count);
		transformed_points[i].resize(Eigen::NoChange,count);
		transformed_normals[i].resize(Eigen::NoChange,count);

		nr_arraypoints[i] = count;

		arraypoints[i] = ap;
		arraynormals[i] = an;
		arraycolors[i] = ac;
		arrayinformations[i] = ai;

		Eigen::Matrix<double, 3, Eigen::Dynamic> & X	= points[i];
		Eigen::Matrix<double, 3, Eigen::Dynamic> & C	= colors[i];
		Eigen::Matrix<double, 3, Eigen::Dynamic> & Xn	= normals[i];

		Eigen::VectorXd information (count);

		int c = 0;
		for(unsigned int i = 0; i < cloud->points.size(); i++){
			pcl::PointXYZRGBNormal p = cloud->points[i];
			if (isValidPoint(p)){

				float xn = p.normal_x;
				float yn = p.normal_y;
				float zn = p.normal_z;

				float x = p.x;
				float y = p.y;
				float z = p.z;

				ap[3*c+0] =x;
				ap[3*c+1] =y;
				ap[3*c+2] =z;

				an[3*c+0] =xn;
				an[3*c+1] =yn;
				an[3*c+2] =zn;

				ac[3*c+0] =p.r;
				ac[3*c+1] =p.g;
				ac[3*c+2] =p.b;

				ai[c] = 1.0/(x*x+y*y+z*z);

				X(0,c)	= x;
				X(1,c)	= y;
				X(2,c)	= z;
				Xn(0,c)	= xn;
				Xn(1,c)	= yn;
				Xn(2,c)	= zn;

				information(c) = ai[c];//1.0/(z*z);
				C(0,c) = p.r;
				C(1,c) = p.g;
				C(2,c) = p.b;
				c++;

			}
		}

		informations[i] = information;

		ArrayData3D<double> * a3d = new ArrayData3D<double>;
		a3d->data	= ap;
		a3d->rows	= count;
		a3dv[i]		= a3d;
		trees3d[i]	= new Tree3d(3, *a3d, nanoflann::KDTreeSingleIndexAdaptorParams(10));
		trees3d[i]->buildIndex();
	}

	//printf("total load time:          %5.5f\n",getTime()-total_load_time_start);
}

void MassRegistrationPPR2::setData(std::vector<RGBDFrame*> frames_,std::vector<ModelMask *> mmasks_){
	double total_load_time_start = getTime();

	unsigned int nr_frames = frames_.size();
	if(arraypoints.size() > 0){
		for(unsigned int i = 0; i < arraypoints.size(); i++){delete[] arraypoints[i];}
		arraypoints.resize(0);
	}

	if(a3dv.size() > 0){
		for(unsigned int i = 0; i < a3dv.size(); i++){delete a3dv[i];}
		a3dv.resize(0);
	}

	if(trees3d.size() > 0){
		for(unsigned int i = 0; i < trees3d.size(); i++){delete trees3d[i];}
		trees3d.resize(0);
	}

	for(unsigned int i = 0; i < nr_frames; i++){addData(frames_[i], mmasks_[i]);}
	//printf("total load time:          %5.5f\n",getTime()-total_load_time_start);
}

void matchframes(Eigen::Affine3d rp, int nr_ap, double * ap, std::vector<int> & matchid, std::vector<double> & matchdist, Tree3d * t3d, double & new_good_rematches, double & new_total_rematches){
	const double & m00 = rp(0,0); const double & m01 = rp(0,1); const double & m02 = rp(0,2); const double & m03 = rp(0,3);
	const double & m10 = rp(1,0); const double & m11 = rp(1,1); const double & m12 = rp(1,2); const double & m13 = rp(1,3);
	const double & m20 = rp(2,0); const double & m21 = rp(2,1); const double & m22 = rp(2,2); const double & m23 = rp(2,3);

	matchid.resize(nr_ap);
	matchdist.resize(nr_ap);

#pragma omp parallel for num_threads(8)
	for(unsigned int k = 0; k < nr_ap; ++k) {
		int prev = matchid[k];
		double qp [3];
		const double & src_x = ap[3*k+0];
		const double & src_y = ap[3*k+1];
		const double & src_z = ap[3*k+2];

		qp[0] = m00*src_x + m01*src_y + m02*src_z + m03;
		qp[1] = m10*src_x + m11*src_y + m12*src_z + m13;
		qp[2] = m20*src_x + m21*src_y + m22*src_z + m23;

		size_t ret_index; double out_dist_sqr;
		nanoflann::KNNResultSet<double> resultSet(1);
		resultSet.init(&ret_index, &out_dist_sqr );
		t3d->findNeighbors(resultSet, qp, nanoflann::SearchParams(10));

		int current = ret_index;
		new_good_rematches += prev != current;
		new_total_rematches++;
		matchid[k] = current;
		matchdist[k] = out_dist_sqr;
	}
}

void MassRegistrationPPR2::rematch(std::vector<Eigen::Matrix4d> poses, std::vector<Eigen::Matrix4d> prev_poses, bool first){
	//printf("rematch\n");
	double new_good_rematches = 0;
	double new_total_rematches = 0;
	unsigned int nr_frames = poses.size();

	int rmt = 2;

	if(rmt==2){
		int ignores = 0;
		for(unsigned int i = 0; i < nr_frames; i++){
			if(!is_ok[i]){continue;}
			nr_matches[i] = 0;

			if(use_depthedge){

			}
			if(use_surface){
				for(unsigned int j = 0; j < nr_frames; j++){
					if(!is_ok[j]){continue;}
					if(i == j){continue;}
					Eigen::Affine3d rp = Eigen::Affine3d(poses[j].inverse()*poses[i]);
					if(!first){
						Eigen::Affine3d prev_rp = Eigen::Affine3d(prev_poses[j].inverse()*prev_poses[i]);

						Eigen::Affine3d diff = prev_rp.inverse()*rp;

						double change_trans = 0;
						double change_rot = 0;
						double dt = 0;
						for(unsigned int k = 0; k < 3; k++){
							dt += diff(k,3)*diff(k,3);
							for(unsigned int l = 0; l < 3; l++){
								if(k == l){ change_rot += fabs(1-diff(k,l));}
								else{		change_rot += fabs(diff(k,l));}
							}
						}
						change_trans += sqrt(dt);

						//if(change_trans < 1*stopval && change_rot < 1*stopval){ignores++;continue;}
					}

					//prev_poses[j] = poses[j];
					matchframes(rp, nr_arraypoints[i], arraypoints[i], matchids[i][j], matchdists[i][j],trees3d[j],	new_good_rematches,new_total_rematches);
/*
					const double & m00 = rp(0,0); const double & m01 = rp(0,1); const double & m02 = rp(0,2); const double & m03 = rp(0,3);
					const double & m10 = rp(1,0); const double & m11 = rp(1,1); const double & m12 = rp(1,2); const double & m13 = rp(1,3);
					const double & m20 = rp(2,0); const double & m21 = rp(2,1); const double & m22 = rp(2,2); const double & m23 = rp(2,3);

					const unsigned int nr_ap = nr_arraypoints[i];
					double * ap = arraypoints[i];

					std::vector<int> & matchid = matchids[i][j];
					std::vector<double> & matchdist = matchdists[i][j];
					matchid.resize(nr_ap);
					matchdist.resize(nr_ap);
					Tree3d * t3d = trees3d[j];

#pragma omp parallel for num_threads(8)
					for(unsigned int k = 0; k < nr_ap; ++k) {
						int prev = matchid[k];
						double qp [3];
						const double & src_x = ap[3*k+0];
						const double & src_y = ap[3*k+1];
						const double & src_z = ap[3*k+2];

						qp[0] = m00*src_x + m01*src_y + m02*src_z + m03;
						qp[1] = m10*src_x + m11*src_y + m12*src_z + m13;
						qp[2] = m20*src_x + m21*src_y + m22*src_z + m23;

						size_t ret_index; double out_dist_sqr;
						nanoflann::KNNResultSet<double> resultSet(1);
						resultSet.init(&ret_index, &out_dist_sqr );
						t3d->findNeighbors(resultSet, qp, nanoflann::SearchParams(10));

						int current = ret_index;
						new_good_rematches += prev != current;
						new_total_rematches++;
						matchid[k] = current;
						matchdist[k] = out_dist_sqr;
					}
					nr_matches[i] += matchid.size();
					*/
				}
			}
		}
		//printf("ignores: %i rematches: %i ratio of good rematches: %4.4f\n",ignores,nr_frames*(nr_frames-1)-ignores,new_good_rematches/new_total_rematches);
	}

	if(rmt==1){
		for(unsigned int i = 0; i < nr_frames; i++){
			if(!is_ok[i]){continue;}
			nr_matches[i] = 0;

			double * ap = arraypoints[i];
			const unsigned int nr_ap = nr_arraypoints[i];

			for(unsigned int j = 0; j < nr_frames; j++){
				if(!is_ok[j]){continue;}
				if(i == j){continue;}
				Eigen::Affine3d rp = Eigen::Affine3d(poses[j].inverse()*poses[i]);
				const double & m00 = rp(0,0); const double & m01 = rp(0,1); const double & m02 = rp(0,2); const double & m03 = rp(0,3);
				const double & m10 = rp(1,0); const double & m11 = rp(1,1); const double & m12 = rp(1,2); const double & m13 = rp(1,3);
				const double & m20 = rp(2,0); const double & m21 = rp(2,1); const double & m22 = rp(2,2); const double & m23 = rp(2,3);

				std::vector<int> & matchid = matchids[i][j];
				matchid.resize(nr_ap);
				Tree3d * t3d = trees3d[j];

				for(unsigned int k = 0; k < nr_ap; ++k) {
					int prev = matchid[k];
					double qp [3];
					const double & src_x = ap[3*k+0];
					const double & src_y = ap[3*k+1];
					const double & src_z = ap[3*k+2];

					qp[0] = m00*src_x + m01*src_y + m02*src_z + m03;
					qp[1] = m10*src_x + m11*src_y + m12*src_z + m13;
					qp[2] = m20*src_x + m21*src_y + m22*src_z + m23;

					size_t ret_index; double out_dist_sqr;
					nanoflann::KNNResultSet<double> resultSet(1);
					resultSet.init(&ret_index, &out_dist_sqr );
					t3d->findNeighbors(resultSet, qp, nanoflann::SearchParams(10));

					int current = ret_index;
					new_good_rematches += prev != current;
					new_total_rematches++;
					matchid[k] = current;
				}
				nr_matches[i] += matchid.size();
			}
		}
	}
	if(rmt==0){
		for(unsigned int i = 0; i < nr_frames; i++){
			if(!is_ok[i]){continue;}
			nr_matches[i] = 0;

			double * ap = arraypoints[i];
			int nr_ap = nr_arraypoints[i];

			for(unsigned int j = 0; j < nr_frames; j++){
				if(!is_ok[j]){continue;}
				if(i == j){continue;}
				Eigen::Affine3d rp = Eigen::Affine3d(poses[j].inverse()*poses[i]);
				const double & m00 = rp(0,0); const double & m01 = rp(0,1); const double & m02 = rp(0,2); const double & m03 = rp(0,3);
				const double & m10 = rp(1,0); const double & m11 = rp(1,1); const double & m12 = rp(1,2); const double & m13 = rp(1,3);
				const double & m20 = rp(2,0); const double & m21 = rp(2,1); const double & m22 = rp(2,2); const double & m23 = rp(2,3);

				Eigen::Matrix<double, 3, Eigen::Dynamic> tX	= rp*points[i];

				unsigned int nr_data = nr_datas[i];
				std::vector<int> & matchid = matchids[i][j];
				matchid.resize(nr_data);
				Tree3d * t3d = trees3d[j];

				for(unsigned int k = 0; k < nr_data; ++k) {
					int prev = matchid[k];
					double * qp = tX.col(k).data();

					size_t ret_index; double out_dist_sqr;
					nanoflann::KNNResultSet<double> resultSet(1);
					resultSet.init(&ret_index, &out_dist_sqr );
					t3d->findNeighbors(resultSet, qp, nanoflann::SearchParams(10));

					int current = ret_index;
					new_good_rematches += prev != current;
					new_total_rematches++;
					matchid[k] = current;
				}
				nr_matches[i] += matchid.size();
			}
		}
	}
	//	good_rematches += new_good_rematches;
	//	total_rematches += new_total_rematches
}


Eigen::MatrixXd MassRegistrationPPR2::getAllResiduals(std::vector<Eigen::Matrix4d> poses){
	unsigned int nr_frames = poses.size();
	Eigen::MatrixXd all_residuals;

	int total_matches = 0;
	for(unsigned int i = 0; i < nr_frames; i++){
		if(!is_ok[i]){continue;}
		for(unsigned int j = 0; j < nr_frames; j++){
			if(!is_ok[j]){continue;}
			total_matches += matchids[i][j].size();
		}
	}

	int all_residuals_type = 1;

	if(all_residuals_type == 1){
		switch(type) {
		case PointToPoint:	{all_residuals = Eigen::Matrix3Xd::Zero(3,total_matches);}break;
		case PointToPlane:	{all_residuals = Eigen::MatrixXd::Zero(1,total_matches);}break;
		default:			{printf("type not set\n");}					break;
		}

		int count = 0;
		for(unsigned int i = 0; i < nr_frames; i++){
			if(!is_ok[i]){continue;}

			double * api = arraypoints[i];
			double * ani = arraynormals[i];
			double * aci = arraycolors[i];
			double * aii = arrayinformations[i];
			const unsigned int nr_api = nr_arraypoints[i];
			for(unsigned int j = 0; j < nr_frames; j++){
				if(!is_ok[j]){continue;}
				if(i == j){continue;}

				double * apj = arraypoints[j];
				double * anj = arraynormals[j];
				double * acj = arraycolors[j];
				double * aij = arrayinformations[j];
				const unsigned int nr_apj = nr_arraypoints[j];

				std::vector<int> & matchidi = matchids[i][j];
				unsigned int matchesi = matchidi.size();

				Eigen::Affine3d rp = Eigen::Affine3d(poses[j].inverse()*poses[i]);
				const double & m00 = rp(0,0); const double & m01 = rp(0,1); const double & m02 = rp(0,2); const double & m03 = rp(0,3);
				const double & m10 = rp(1,0); const double & m11 = rp(1,1); const double & m12 = rp(1,2); const double & m13 = rp(1,3);
				const double & m20 = rp(2,0); const double & m21 = rp(2,1); const double & m22 = rp(2,2); const double & m23 = rp(2,3);

				if(type == PointToPlane){
					for(unsigned int ki = 0; ki < nr_api; ++ki) {
						int kj = matchidi[ki];
						if( kj < 0 || kj >= nr_apj){continue;}
						const double & src_x = api[3*ki+0];
						const double & src_y = api[3*ki+1];
						const double & src_z = api[3*ki+2];

						const double & src_nx = ani[3*ki+0];
						const double & src_ny = ani[3*ki+1];
						const double & src_nz = ani[3*ki+2];

						const double & info_i = aii[ki];

						const double & dst_x = apj[3*kj+0];
						const double & dst_y = apj[3*kj+1];
						const double & dst_z = apj[3*kj+2];


						const double & info_j = aij[kj];

						float tx = m00*src_x + m01*src_y + m02*src_z + m03;
						float ty = m10*src_x + m11*src_y + m12*src_z + m13;
						float tz = m20*src_x + m21*src_y + m22*src_z + m23;

						float txn = m00*src_nx + m01*src_ny + m02*src_nz;
						float tyn = m10*src_nx + m11*src_ny + m12*src_nz;
						float tzn = m20*src_nx + m21*src_ny + m22*src_nz;

						const double rw = 1.0/(1.0/info_i+1.0/info_j);
						const double di = (txn*(tx-dst_x) + tyn*(ty-dst_y) + tzn*(tz-dst_z))*rw;
						all_residuals(0,count) = di;
						count++;
					}
				}

				if(type == PointToPoint){
					for(unsigned int ki = 0; ki < nr_api; ++ki) {
						int kj = matchidi[ki];
						if( kj < 0 ){continue;}
						const double & src_x = api[3*ki+0];
						const double & src_y = api[3*ki+1];
						const double & src_z = api[3*ki+2];

						const double & info_i = aii[ki];

						const double & dst_x = apj[3*kj+0];
						const double & dst_y = apj[3*kj+1];
						const double & dst_z = apj[3*kj+2];

						const double & info_j = aij[kj];

                        float tx = m00*src_x + m01*src_y + m02*src_z + m03;//        const unsigned int src_nr_ap = kp_nr_arraypoints[i];
                        //        if(src_nr_ap == 0){continue;}
						float ty = m10*src_x + m11*src_y + m12*src_z + m13;
						float tz = m20*src_x + m21*src_y + m22*src_z + m23;

						const double rw = 1.0/(1.0/info_i+1.0/info_j);
						all_residuals(0,count) = (tx-dst_x)*rw;
						all_residuals(1,count) = (ty-dst_y)*rw;
						all_residuals(2,count) = (tz-dst_z)*rw;
						count++;
					}
				}
			}
		}
	}

	if(all_residuals_type == 0){
		switch(type) {
		case PointToPoint:	{all_residuals = Eigen::Matrix3Xd::Zero(3,total_matches);}break;
		case PointToPlane:	{all_residuals = Eigen::MatrixXd::Zero(1,total_matches);}break;
		default:			{printf("type not set\n");}					break;
		}

		int count = 0;
		for(unsigned int i = 0; i < nr_frames; i++){
			if(!is_ok[i]){continue;}
			Eigen::Matrix<double, 3, Eigen::Dynamic> & tXi	= transformed_points[i];
			Eigen::Matrix<double, 3, Eigen::Dynamic> & tXni	= transformed_normals[i];
			Eigen::VectorXd & informationi					= informations[i];
			for(unsigned int j = 0; j < nr_frames; j++){
				if(!is_ok[j]){continue;}
				if(i == j){continue;}
				std::vector<int> & matchidi = matchids[i][j];
				unsigned int matchesi = matchidi.size();
				Eigen::Matrix<double, 3, Eigen::Dynamic> & tXj	= transformed_points[j];
				Eigen::Matrix<double, 3, Eigen::Dynamic> & tXnj	= transformed_normals[j];
				Eigen::VectorXd & informationj					= informations[j];
				Eigen::Matrix3Xd Xp		= Eigen::Matrix3Xd::Zero(3,	matchesi);
				Eigen::Matrix3Xd Xn		= Eigen::Matrix3Xd::Zero(3,	matchesi);
				Eigen::Matrix3Xd Qp		= Eigen::Matrix3Xd::Zero(3,	matchesi);
				Eigen::Matrix3Xd Qn		= Eigen::Matrix3Xd::Zero(3,	matchesi);
				Eigen::VectorXd  rangeW	= Eigen::VectorXd::Zero(	matchesi);

				for(unsigned int ki = 0; ki < matchesi; ki++){
					int kj = matchidi[ki];
					if( ki >= Qp.cols() || kj < 0 || kj >= tXj.cols() ){continue;}
					Qp.col(ki) = tXj.col(kj);
					Qn.col(ki) = tXnj.col(kj);
					Xp.col(ki) = tXi.col(ki);
					Xn.col(ki) = tXni.col(ki);
					rangeW(ki) = 1.0/(1.0/informationi(ki)+1.0/informationj(kj));
				}
				Eigen::MatrixXd residuals;
				switch(type) {
				case PointToPoint:	{residuals = Xp-Qp;} 						break;
				case PointToPlane:	{
					residuals		= Eigen::MatrixXd::Zero(1,	Xp.cols());
					for(int i=0; i<Xp.cols(); ++i) {
						float dx = Xp(0,i)-Qp(0,i);
						float dy = Xp(1,i)-Qp(1,i);
						float dz = Xp(2,i)-Qp(2,i);
						float qx = Qn(0,i);
						float qy = Qn(1,i);
						float qz = Qn(2,i);
						float di = qx*dx + qy*dy + qz*dz;
						residuals(0,i) = di;
					}
				}break;
				default:			{printf("type not set\n");}					break;
				}
				for(unsigned int k=0; k < matchesi; ++k) {residuals.col(k) *= rangeW(k);}
				all_residuals.block(0,count,residuals.rows(),residuals.cols()) = residuals;
				count += residuals.cols();
			}
		}
	}
	return all_residuals;
}

Eigen::MatrixXd MassRegistrationPPR2::getAllKpResiduals(std::vector<Eigen::Matrix4d> poses){
	unsigned int nr_frames = poses.size();


//    printf("is_ok size: %i\n",is_ok.size());
	int total_matches = 0;
	for(unsigned int i = 0; i < nr_frames; i++){
        const unsigned int src_nr_ap = kp_nr_arraypoints[i];
        if(src_nr_ap == 0){continue;}
        for(unsigned int j = 0; j < nr_frames; j++){
            const unsigned int dst_nr_ap = kp_nr_arraypoints[j];
            if(dst_nr_ap == 0){continue;}
			total_matches += kp_matches[i][j].size();
		}
	}



//    printf("total matches: %i\n",total_matches);


//    total_matches = 0;
//    for(unsigned int i = 0; i < nr_frames; i++){
//        for(unsigned int j = 0; j < nr_frames; j++){
//            total_matches += kp_matches[i][j].size();
//        }
//    }


//    printf("total matches: %i\n",total_matches);

//    exit(0);

	int count = 0;
	Eigen::MatrixXd all_residuals = Eigen::Matrix3Xd::Zero(3,total_matches);
//if(total_matches == 0){return all_residuals;}

	int ignores = 0;
	for(unsigned int i = 0; i < nr_frames; i++){
		const unsigned int src_nr_ap = kp_nr_arraypoints[i];
		if(src_nr_ap == 0){continue;}
		double * src_ap = kp_arraypoints[i];
		double * src_ai = kp_arrayinformations[i];

		for(unsigned int j = 0; j < nr_frames; j++){
			if(i == j){continue;}

			const unsigned int dst_nr_ap = kp_nr_arraypoints[j];
			if(dst_nr_ap == 0){continue;}

//            printf("nr kp(%i %i): %i %i\n",i,j,src_nr_ap,dst_nr_ap);
			double * dst_ap = kp_arraypoints[j];
			double * dst_ai = kp_arrayinformations[j];

			Eigen::Affine3d rp = Eigen::Affine3d(poses[j].inverse()*poses[i]);

			const double & m00 = rp(0,0); const double & m01 = rp(0,1); const double & m02 = rp(0,2); const double & m03 = rp(0,3);
			const double & m10 = rp(1,0); const double & m11 = rp(1,1); const double & m12 = rp(1,2); const double & m13 = rp(1,3);
			const double & m20 = rp(2,0); const double & m21 = rp(2,1); const double & m22 = rp(2,2); const double & m23 = rp(2,3);

			std::vector< TestMatch > & matches = kp_matches[i][j];
			unsigned int nr_matches = matches.size();

			for(unsigned int k = 0; k < nr_matches; ++k) {
				TestMatch & tm = matches[k];
				unsigned int src_k = tm.src;
				unsigned int dst_k = tm.dst;

				const double & src_x = src_ap[3*src_k+0];
				const double & src_y = src_ap[3*src_k+1];
				const double & src_z = src_ap[3*src_k+2];
				const double & src_info = src_ai[src_k];

				double sx = m00*src_x + m01*src_y + m02*src_z + m03;
				double sy = m10*src_x + m11*src_y + m12*src_z + m13;
				double sz = m20*src_x + m21*src_y + m22*src_z + m23;

				double dx = sx-dst_ap[3*dst_k+0];
				double dy = sy-dst_ap[3*dst_k+1];
				double dz = sz-dst_ap[3*dst_k+2];

				const double & dst_info = dst_ai[dst_k];
				const double rw = 1.0/(1.0/src_info+1.0/dst_info);
//                if(total_matches <= count){printf("strange\n");}
				all_residuals(0,count) = dx*rw;
				all_residuals(1,count) = dy*rw;
				all_residuals(2,count) = dz*rw;
				count++;
			}
		}
	}
	return all_residuals;
}

void MassRegistrationPPR2::clearData(){}

void MassRegistrationPPR2::addData(RGBDFrame* frame, ModelMask * mmask){

	double total_load_time_start = getTime();
	frames.push_back(frame);
	mmasks.push_back(mmask);

    nr_matches.push_back(	0);
    matchids.push_back(		std::vector< std::vector<int> >() );
    matchdists.push_back(	std::vector< std::vector<double> >() );
	nr_datas.push_back(		0);

	points.push_back(				Eigen::Matrix<double, 3, Eigen::Dynamic>());
	colors.push_back(				Eigen::Matrix<double, 3, Eigen::Dynamic>());
	normals.push_back(				Eigen::Matrix<double, 3, Eigen::Dynamic>());
	transformed_points.push_back(	Eigen::Matrix<double, 3, Eigen::Dynamic>());
	transformed_normals.push_back(	Eigen::Matrix<double, 3, Eigen::Dynamic>());

	informations.push_back(			Eigen::VectorXd());

	nr_arraypoints.push_back(0);
	arraypoints.push_back(0);
	arraynormals.push_back(0);
	arraycolors.push_back(0);
	arrayinformations.push_back(0);
	trees3d.push_back(0);
	a3dv.push_back(0);

	depthedge_nr_arraypoints.push_back(0);
	depthedge_arraypoints.push_back(0);
	depthedge_arrayinformations.push_back(0);
	depthedge_trees3d.push_back(0);
	depthedge_a3dv.push_back(0);


	is_ok.push_back(false);

    unsigned int i = points.size()-1;

	bool * maskvec					= mmask->maskvec;
	unsigned char *  edgedata		= (unsigned char *)(frame->depthedges.data);
    unsigned char  * rgbdata		= (unsigned char	*)(frame->rgb.data);
    unsigned short * depthdata		= (unsigned short	*)(frame->depth.data);
    float		   * normalsdata	= (float			*)(frame->normals.data);

    Camera * camera				= frame->camera;
	const unsigned int width	= camera->width;
	const unsigned int height	= camera->height;
	const float idepth			= camera->idepth_scale;
	const float cx				= camera->cx;
	const float cy				= camera->cy;
	const float ifx				= 1.0/camera->fx;
	const float ify				= 1.0/camera->fy;

	int count = 0;
	for(unsigned int w = 0; w < width; w+=maskstep){
		for(unsigned int h = 0; h < height; h+=maskstep){
			int ind = h*width+w;
            if(((w % maskstep == 0) && (h % maskstep == 0) && maskvec[ind]) || ( (w % nomaskstep == 0) && (h % nomaskstep == 0) && nomask)){
				float z = idepth*float(depthdata[ind]);
				float xn = normalsdata[3*ind+0];
				if(z > 0.2 && xn != 2){count++;}
			}
		}
	}


	if(count < 10){
		is_ok[i] = false;
		return;
	}else{
		is_ok[i] = true;
	}

//    printf("%i is ok %i\n",i,is_ok[i]);

	nr_datas[i] = count;
	points[i].resize(Eigen::NoChange,count);
	colors[i].resize(Eigen::NoChange,count);
	normals[i].resize(Eigen::NoChange,count);
	transformed_points[i].resize(Eigen::NoChange,count);
	transformed_normals[i].resize(Eigen::NoChange,count);

	double * ap = new double[3*count];
	double * an = new double[3*count];
	double * ac = new double[3*count];
	double * ai = new double[3*count];
	nr_arraypoints[i] = count;
	arraypoints[i] = ap;
	arraynormals[i] = an;
	arraycolors[i] = ac;
	arrayinformations[i] = ai;

	Eigen::Matrix<double, 3, Eigen::Dynamic> & X	= points[i];
	Eigen::Matrix<double, 3, Eigen::Dynamic> & C	= colors[i];
	Eigen::Matrix<double, 3, Eigen::Dynamic> & Xn	= normals[i];
	Eigen::Matrix<double, 3, Eigen::Dynamic> & tX	= transformed_points[i];
	Eigen::Matrix<double, 3, Eigen::Dynamic> & tXn	= transformed_normals[i];
	Eigen::VectorXd information (count);

	int c = 0;
	for(unsigned int w = 0; w < width; w+=maskstep){
		for(unsigned int h = 0; h < height;h+=maskstep){
			if(c == count){continue;}
			int ind = h*width+w;
            //if(maskvec[ind] || nomask){
            if(((w % maskstep == 0) && (h % maskstep == 0) && maskvec[ind]) || ( (w % nomaskstep == 0) && (h % nomaskstep == 0) && nomask)){
				float z = idepth*float(depthdata[ind]);
				float xn = normalsdata[3*ind+0];

				if(z > 0.2 && xn != 2){
					float yn = normalsdata[3*ind+1];
					float zn = normalsdata[3*ind+2];

					float x = (w - cx) * z * ifx;
					float y = (h - cy) * z * ify;

					ap[3*c+0] =x;
					ap[3*c+1] =y;
					ap[3*c+2] =z;

					an[3*c+0] =xn;
					an[3*c+1] =yn;
					an[3*c+2] =zn;

					ac[3*c+0] =rgbdata[3*ind+2];
					ac[3*c+1] =rgbdata[3*ind+1];
					ac[3*c+2] =rgbdata[3*ind+0];

					ai[c] = pow(fabs(z),-2);//1.0/(z*z);

					X(0,c)	= x;
					X(1,c)	= y;
					X(2,c)	= z;
					Xn(0,c)	= xn;
					Xn(1,c)	= yn;
					Xn(2,c)	= zn;

					information(c) = ai[c];//1.0/(z*z);
					C(0,c) = rgbdata[3*ind+0];
					C(1,c) = rgbdata[3*ind+1];
					C(2,c) = rgbdata[3*ind+2];
					c++;
				}
			}
		}
	}
	informations[i] = information;

	ArrayData3D<double> * a3d = new ArrayData3D<double>;
	a3d->data	= ap;
	a3d->rows	= count;
	a3dv[i]		= a3d;
	trees3d[i]	= new Tree3d(3, *a3d, nanoflann::KDTreeSingleIndexAdaptorParams(10));
	trees3d[i]->buildIndex();


	int depthedge_count = 0;
	for(unsigned int w = 0; w < width; w++){
		for(unsigned int h = 0; h < height; h++){
			int ind = h*width+w;
			if(maskvec[ind] && edgedata[ind] == 255){
				float z = idepth*float(depthdata[ind]);
				if(z > 0.2){depthedge_count++;}
			}
		}
	}

	if(depthedge_count < 10){
		double * depthedge_ap = new double[3*depthedge_count];
		double * depthedge_ai = new double[3*depthedge_count];
		depthedge_nr_arraypoints[i] = depthedge_count;
		depthedge_arraypoints[i] = depthedge_ap;
		depthedge_arrayinformations[i] = depthedge_ai;

		c = 0;
		for(unsigned int w = 0; w < width; w++){
			for(unsigned int h = 0; h < height; h++){
				if(c == depthedge_count){continue;}
				int ind = h*width+w;
				if(maskvec[ind] && edgedata[ind] == 255){
					float z = idepth*float(depthdata[ind]);
					if(z > 0.2){
						depthedge_ap[3*c+0] = (w - cx) * z * ifx;
						depthedge_ap[3*c+1] = (h - cy) * z * ify;;
						depthedge_ap[3*c+2] = z;
						depthedge_ai[c] = pow(fabs(z),-2);//1.0/(z*z);
						c++;
					}
				}
			}
		}

		ArrayData3D<double> * depthedge_a3d = new ArrayData3D<double>;
		depthedge_a3d->data					= depthedge_ap;
		depthedge_a3d->rows					= depthedge_count;
		depthedge_a3dv[i]					= depthedge_a3d;
		depthedge_trees3d[i]				= new Tree3d(3, *depthedge_a3d, nanoflann::KDTreeSingleIndexAdaptorParams(10));
		depthedge_trees3d[i]->buildIndex();
	}

	//printf("total load time:          %5.5f\n",getTime()-total_load_time_start);
}

void MassRegistrationPPR2::addData(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud){
	double total_load_time_start = getTime();

    nr_matches.push_back(	0);
    matchids.push_back(		std::vector< std::vector<int> >() );
    matchdists.push_back(	std::vector< std::vector<double> >() );
	nr_datas.push_back(		0);

	points.push_back(				Eigen::Matrix<double, 3, Eigen::Dynamic>());
	colors.push_back(				Eigen::Matrix<double, 3, Eigen::Dynamic>());
	normals.push_back(				Eigen::Matrix<double, 3, Eigen::Dynamic>());
	transformed_points.push_back(	Eigen::Matrix<double, 3, Eigen::Dynamic>());
	transformed_normals.push_back(	Eigen::Matrix<double, 3, Eigen::Dynamic>());

	informations.push_back(			Eigen::VectorXd());

	nr_arraypoints.push_back(0);

	arraypoints.push_back(0);
	arraynormals.push_back(0);
	arraycolors.push_back(0);
	arrayinformations.push_back(0);

	trees3d.push_back(0);
	a3dv.push_back(0);
	is_ok.push_back(false);

	unsigned int i = points.size()-1;

	int count = 0;
    for(unsigned int j = 0; j < cloud->points.size(); j+=nomaskstep*nomaskstep){
		if (isValidPoint(cloud->points[j])){count++;}
	}

	printf("count: %i\n",count);

	if(count < 10){
		is_ok[i] = false;
		return;
	}else{
		is_ok[i] = true;
	}



	double * ap = new double[3*count];
	double * an = new double[3*count];
	double * ac = new double[3*count];
	double * ai = new double[3*count];

	nr_datas[i] = count;
	points[i].resize(Eigen::NoChange,count);
	colors[i].resize(Eigen::NoChange,count);
	normals[i].resize(Eigen::NoChange,count);
	transformed_points[i].resize(Eigen::NoChange,count);
	transformed_normals[i].resize(Eigen::NoChange,count);

	nr_arraypoints[i] = count;

	arraypoints[i] = ap;
	arraynormals[i] = an;
	arraycolors[i] = ac;
	arrayinformations[i] = ai;

	Eigen::Matrix<double, 3, Eigen::Dynamic> & X	= points[i];
	Eigen::Matrix<double, 3, Eigen::Dynamic> & C	= colors[i];
	Eigen::Matrix<double, 3, Eigen::Dynamic> & Xn	= normals[i];

	Eigen::VectorXd information (count);

	int c = 0;
    for(unsigned int j = 0; j < cloud->points.size(); j+=nomaskstep*nomaskstep){
		pcl::PointXYZRGBNormal p = cloud->points[j];
		if (isValidPoint(p)){

			float xn = p.normal_x;
			float yn = p.normal_y;
			float zn = p.normal_z;

			float x = p.x;
			float y = p.y;
			float z = p.z;

			//if(j % 1000 == 0){printf("%i -> %f %f %f\n",j,x,y,z);}

			ap[3*c+0] =x;
			ap[3*c+1] =y;
			ap[3*c+2] =z;

			an[3*c+0] =xn;
			an[3*c+1] =yn;
			an[3*c+2] =zn;

			ac[3*c+0] =p.r;
			ac[3*c+1] =p.g;
			ac[3*c+2] =p.b;

			ai[c] = 1/sqrt(x*x+y*y+z*z);

			X(0,c)	= x;
			X(1,c)	= y;
			X(2,c)	= z;
			Xn(0,c)	= xn;
			Xn(1,c)	= yn;
			Xn(2,c)	= zn;

			information(c) = ai[c];//1/sqrt(2+x*x+y*y+z*z);//1.0/(z*z);
			C(0,c) = p.r;
			C(1,c) = p.g;
			C(2,c) = p.b;
			c++;

		}
	}

	informations[i] = information;

    kp_nr_arraypoints.push_back(0);
    kp_arraypoints.push_back(0);
    kp_arraynormals.push_back(0);
    kp_arrayinformations.push_back(0);
    kp_arraydescriptors.push_back(0);

	ArrayData3D<double> * a3d = new ArrayData3D<double>;
	a3d->data	= ap;
	a3d->rows	= count;
	a3dv[i]		= a3d;
	trees3d[i]	= new Tree3d(3, *a3d, nanoflann::KDTreeSingleIndexAdaptorParams(10));
	trees3d[i]->buildIndex();
	if(visualizationLvl > 0){
		printf("total load time:          %5.5f\n",getTime()-total_load_time_start);
	}
}

std::vector<Eigen::Matrix4d> MassRegistrationPPR2::optimize(std::vector<Eigen::Matrix4d> poses){
	bool onetoone = true;
	unsigned int nr_frames = poses.size();
	Eigen::MatrixXd Xo1;

	int optt = 2;
    for(int outer=0; outer < 60; ++outer) {
		if(getTime()-total_time_start > timeout){break;}

		//printf("outer: %i\n",outer);

		std::vector<Eigen::Matrix4d> poses2 = poses;
		if(type == PointToPlane && optt == 2){
			for(unsigned int i = 0; i < nr_frames; i++){
				if(getTime()-total_time_start > timeout){break;}
				if(!is_ok[i]){continue;}

				unsigned int count				= 0;
				double * api					= arraypoints[i];
				double * ani					= arraynormals[i];
				double * aci					= arraycolors[i];
				double * aii					= arrayinformations[i];
				const unsigned int nr_api		= nr_arraypoints[i];

				unsigned int kp_count			= 0;
//				double * kp_api					= kp_arraypoints[i];
//				double * kp_ani					= kp_arraynormals[i];
//				double * kp_aii					= kp_arrayinformations[i];
//				const unsigned int kp_nr_api	= kp_nr_arraypoints[i];

				Eigen::Affine3d rpi = Eigen::Affine3d(poses[i]);
				const double & mi00 = rpi(0,0); const double & mi01 = rpi(0,1); const double & mi02 = rpi(0,2); const double & mi03 = rpi(0,3);
				const double & mi10 = rpi(1,0); const double & mi11 = rpi(1,1); const double & mi12 = rpi(1,2); const double & mi13 = rpi(1,3);
				const double & mi20 = rpi(2,0); const double & mi21 = rpi(2,1); const double & mi22 = rpi(2,2); const double & mi23 = rpi(2,3);

				for(unsigned int j = 0; j < nr_frames; j++){
					if(!is_ok[j]){continue;}
					if(i == j){continue;}

					double * apj					= arraypoints[j];
					double * anj					= arraynormals[j];
					double * acj					= arraycolors[j];
					double * aij					= arrayinformations[j];
					const unsigned int nr_apj		= nr_arraypoints[j];

//					double * kp_apj					= kp_arraypoints[j];
//					double * kp_anj					= kp_arraynormals[j];
//					double * kp_aij					= kp_arrayinformations[j];
//					const unsigned int kp_nr_apj	= kp_nr_arraypoints[j];

					Eigen::Affine3d rpj = Eigen::Affine3d(poses[j]);
					const double & mj00 = rpj(0,0); const double & mj01 = rpj(0,1); const double & mj02 = rpj(0,2); const double & mj03 = rpj(0,3);
					const double & mj10 = rpj(1,0); const double & mj11 = rpj(1,1); const double & mj12 = rpj(1,2); const double & mj13 = rpj(1,3);
					const double & mj20 = rpj(2,0); const double & mj21 = rpj(2,1); const double & mj22 = rpj(2,2); const double & mj23 = rpj(2,3);

					std::vector<int> & matchidj = matchids[j][i];
					unsigned int matchesj = matchidj.size();
					std::vector<int> & matchidi = matchids[i][j];
					unsigned int matchesi = matchidi.size();


                    std::vector<double> & matchdistj = matchdists[j][i];
                    std::vector<double> & matchdisti = matchdists[i][j];

                    for(unsigned int ki = 0; true && ki < matchesi; ki++){
						int kj = matchidi[ki];
						if( kj == -1 ){continue;}
						if( kj >=  matchesj){continue;}
						if(!onetoone || matchidj[kj] == ki){
							Xp_arr[3*count+0] = api[3*ki+0];
							Xp_arr[3*count+1] = api[3*ki+1];
							Xp_arr[3*count+2] = api[3*ki+2];

							Xn_arr[3*count+0] = ani[3*ki+0];
							Xn_arr[3*count+1] = ani[3*ki+1];
							Xn_arr[3*count+2] = ani[3*ki+2];

							const double & info_i = aii[ki];

							const double & dst_x = apj[3*kj+0];
							const double & dst_y = apj[3*kj+1];
							const double & dst_z = apj[3*kj+2];

							const double & dst_nx = anj[3*kj+0];
							const double & dst_ny = anj[3*kj+1];
							const double & dst_nz = anj[3*kj+2];

							const double & info_j = aij[kj];

							Qp_arr[3*count+0] = mj00*dst_x + mj01*dst_y + mj02*dst_z + mj03;
							Qp_arr[3*count+1] = mj10*dst_x + mj11*dst_y + mj12*dst_z + mj13;
							Qp_arr[3*count+2] = mj20*dst_x + mj21*dst_y + mj22*dst_z + mj23;

							Qn_arr[3*count+0] = mj00*dst_nx + mj01*dst_ny + mj02*dst_nz;
							Qn_arr[3*count+1] = mj10*dst_nx + mj11*dst_ny + mj12*dst_nz;
							Qn_arr[3*count+2] = mj20*dst_nx + mj21*dst_ny + mj22*dst_nz;

							rangeW_arr[count] = 1.0/(1.0/info_i+1.0/info_j);

							count++;
						}
					}
/*
                    if(kp_matches.size() > 0 && kp_matches[i].size() > 0 && kp_matches[i][j].size() > 0){
                        std::vector< TestMatch > & matches = kp_matches[i][j];
                        unsigned int nr_matches = matches.size();
                        for(unsigned int k = 0; k < nr_matches; k++){
                            TestMatch & m = matches[k];
                            int ki = m.src;
                            int kj = m.dst;

                            kp_Xp_arr[3*kp_count+0] = kp_api[3*ki+0];
                            kp_Xp_arr[3*kp_count+1] = kp_api[3*ki+1];
                            kp_Xp_arr[3*kp_count+2] = kp_api[3*ki+2];

                            kp_Xn_arr[3*kp_count+0] = kp_ani[3*ki+0];
                            kp_Xn_arr[3*kp_count+1] = kp_ani[3*ki+1];
                            kp_Xn_arr[3*kp_count+2] = kp_ani[3*ki+2];
                            const double & info_i = kp_aii[ki];

                            const double & dst_x = kp_apj[3*kj+0];
                            const double & dst_y = kp_apj[3*kj+1];
                            const double & dst_z = kp_apj[3*kj+2];

                            const double & dst_nx = kp_anj[3*kj+0];
                            const double & dst_ny = kp_anj[3*kj+1];
                            const double & dst_nz = kp_anj[3*kj+2];

                            const double & info_j = kp_aij[kj];

                            kp_Qp_arr[3*kp_count+0] = mj00*dst_x + mj01*dst_y + mj02*dst_z + mj03;
                            kp_Qp_arr[3*kp_count+1] = mj10*dst_x + mj11*dst_y + mj12*dst_z + mj13;
                            kp_Qp_arr[3*kp_count+2] = mj20*dst_x + mj21*dst_y + mj22*dst_z + mj23;

                            kp_Qn_arr[3*kp_count+0] = mj00*dst_nx + mj01*dst_ny + mj02*dst_nz;
                            kp_Qn_arr[3*kp_count+1] = mj10*dst_nx + mj11*dst_ny + mj12*dst_nz;
                            kp_Qn_arr[3*kp_count+2] = mj20*dst_nx + mj21*dst_ny + mj22*dst_nz;

                            kp_rangeW_arr[kp_count] = 1.0/(1.0/info_i+1.0/info_j);
                            kp_count++;
                        }
                    }
*/
				}
				if(count == 0 && kp_count == 0){break;}

				typedef Eigen::Matrix<double, 6, 1> Vector6d;
				typedef Eigen::Matrix<double, 6, 6> Matrix6d;

				Matrix6d ATA;
				Vector6d ATb;

				//std::vector<Eigen::MatrixXd> Xv;
				//if(visualizationLvl == 4){for(unsigned int j = 0; j < nr_frames; j++){Xv.push_back(transformed_points[j]);}sprintf(buf,"image%5.5i.png",imgcount++);show(Xv,false,std::string(buf),imgcount);}



				Eigen::Affine3d p = rpi;
				bool do_inner = true;
				for(int inner=0; inner < 15 && do_inner; ++inner) {

					pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr scloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
					pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr dcloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);

					ATA.setZero ();
					ATb.setZero ();
					const double & m00 = p(0,0); const double & m01 = p(0,1); const double & m02 = p(0,2); const double & m03 = p(0,3);
					const double & m10 = p(1,0); const double & m11 = p(1,1); const double & m12 = p(1,2); const double & m13 = p(1,3);
					const double & m20 = p(2,0); const double & m21 = p(2,1); const double & m22 = p(2,2); const double & m23 = p(2,3);
					double planeNoise = func->getNoise();
					double planeInfo = 1.0/(planeNoise*planeNoise);


					double kpNoise = kpfunc->getNoise();
					double kpInfo = 1.0/(kpNoise*kpNoise);

                    for(unsigned int co = 0; co < count; co++){
						const double & src_x = Xp_arr[3*co+0];
						const double & src_y = Xp_arr[3*co+1];
						const double & src_z = Xp_arr[3*co+2];
						const double & src_nx = Xn_arr[3*co+0];
						const double & src_ny = Xn_arr[3*co+1];
						const double & src_nz = Xn_arr[3*co+2];

						const double & dx = Qp_arr[3*co+0];
						const double & dy = Qp_arr[3*co+1];
						const double & dz = Qp_arr[3*co+2];
						const double & dnx = Qn_arr[3*co+0];
						const double & dny = Qn_arr[3*co+1];
						const double & dnz = Qn_arr[3*co+2];

						const double & rw = rangeW_arr[co];

						const double & sx = m00*src_x + m01*src_y + m02*src_z + m03;
						const double & sy = m10*src_x + m11*src_y + m12*src_z + m13;
						const double & sz = m20*src_x + m21*src_y + m22*src_z + m23;

						const double & nx = m00*src_nx + m01*src_ny + m02*src_nz;
						const double & ny = m10*src_nx + m11*src_ny + m12*src_nz;
						const double & nz = m20*src_nx + m21*src_ny + m22*src_nz;

						const double & angle = nx*dnx+ny*dny+nz*dnz;
						//printf("%f\n",angle);
						//if(angle < 0){exit(0);continue;}

						if(angle < 0){continue;}

						double di = rw*(nx*(sx-dx) + ny*(sy-dy) + nz*(sz-dz));
						//double weight = angle*angle*angle*angle*func->getProb(di)*rw*rw;
						double prob = func->getProb(di);
						double weight = planeInfo*prob*rw*rw;

						if(visualizationLvl == 5 && i == 0){
							pcl::PointXYZRGBNormal p;
							p.x = sx;
							p.y = sy;
							p.z = sz;
							p.b = 0;
							p.g = 255;
							p.r = 0;
							scloud->points.push_back(p);

							pcl::PointXYZRGBNormal p1;
							p1.x = dx;
							p1.y = dy;
							p1.z = dz;
							p1.b = 255.0*prob;
							p1.g = 255.0*prob;
							p1.r = 255.0*prob;
							dcloud->points.push_back(p1);
						}

						const double & a = nz*sy - ny*sz;
						const double & b = nx*sz - nz*sx;
						const double & c = ny*sx - nx*sy;

						ATA.coeffRef (0) += weight * a * a;
						ATA.coeffRef (1) += weight * a * b;
						ATA.coeffRef (2) += weight * a * c;
						ATA.coeffRef (3) += weight * a * nx;
						ATA.coeffRef (4) += weight * a * ny;
						ATA.coeffRef (5) += weight * a * nz;
						ATA.coeffRef (7) += weight * b * b;
						ATA.coeffRef (8) += weight * b * c;
						ATA.coeffRef (9) += weight * b * nx;
						ATA.coeffRef (10) += weight * b * ny;
						ATA.coeffRef (11) += weight * b * nz;
						ATA.coeffRef (14) += weight * c * c;
						ATA.coeffRef (15) += weight * c * nx;
						ATA.coeffRef (16) += weight * c * ny;
						ATA.coeffRef (17) += weight * c * nz;
						ATA.coeffRef (21) += weight * nx * nx;
						ATA.coeffRef (22) += weight * nx * ny;
						ATA.coeffRef (23) += weight * nx * nz;
						ATA.coeffRef (28) += weight * ny * ny;
						ATA.coeffRef (29) += weight * ny * nz;
						ATA.coeffRef (35) += weight * nz * nz;

						const double & d = weight * (nx*dx + ny*dy + nz*dz - nx*sx - ny*sy - nz*sz);

						ATb.coeffRef (0) += a * d;
						ATb.coeffRef (1) += b * d;
						ATb.coeffRef (2) += c * d;
						ATb.coeffRef (3) += nx * d;
						ATb.coeffRef (4) += ny * d;
						ATb.coeffRef (5) += nz * d;
					}

					double wsum = 0;
					double wsx = 0;
					double wsy = 0;
					double wsz = 0;
					for(unsigned int co = 0; co < kp_count; co++){
						const double & src_x = kp_Xp_arr[3*co+0];
						const double & src_y = kp_Xp_arr[3*co+1];
						const double & src_z = kp_Xp_arr[3*co+2];

						const double & src_nx = kp_Xn_arr[3*co+0];
						const double & src_ny = kp_Xn_arr[3*co+1];
						const double & src_nz = kp_Xn_arr[3*co+2];


						const double & dx = kp_Qp_arr[3*co+0];
						const double & dy = kp_Qp_arr[3*co+1];
						const double & dz = kp_Qp_arr[3*co+2];

						const double & dnx = kp_Qn_arr[3*co+0];
						const double & dny = kp_Qn_arr[3*co+1];
						const double & dnz = kp_Qn_arr[3*co+2];

						const double & rw = kp_rangeW_arr[co];

						const double & sx = m00*src_x + m01*src_y + m02*src_z + m03;
						const double & sy = m10*src_x + m11*src_y + m12*src_z + m13;
						const double & sz = m20*src_x + m21*src_y + m22*src_z + m23;

						const double & nx = m00*src_nx + m01*src_ny + m02*src_nz;
						const double & ny = m10*src_nx + m11*src_ny + m12*src_nz;
						const double & nz = m20*src_nx + m21*src_ny + m22*src_nz;

						const double & angle = nx*dnx+ny*dny+nz*dnz;



						if(angle < 0){continue;}

						const double diffX = dx-sx;
						const double diffY = dy-sy;
						const double diffZ = dz-sz;

						double probX = kpfunc->getProb(rw*diffX);
						double probY = kpfunc->getProb(rw*diffY);
						double probZ = kpfunc->getProb(rw*diffZ);
						double prob = probX*probY*probZ/(probX*probY*probZ + (1-probX)*(1-probY)*(1-probZ));

                        if(prob < 0.000001){continue;}

						if(visualizationLvl == 5 && i == 0){
							pcl::PointXYZRGBNormal p;
							p.x = sx;
							p.y = sy;
							p.z = sz;
							p.normal_x = nx;
							p.normal_y = ny;
							p.normal_z = nz;
							p.b = 0;
							p.g = 255;
							p.r = 0;
							scloud->points.push_back(p);

							pcl::PointXYZRGBNormal pn = p;
							pn.x += 0.01*nx;
							pn.y += 0.01*ny;
							pn.z += 0.01*nz;

							pcl::PointXYZRGBNormal p1;
							p1.x = dx;
							p1.y = dy;
							p1.z = dz;
							p1.normal_x = dnx;
							p1.normal_y = dny;
							p1.normal_z = dnz;
							p1.b = 255.0;
							p1.g = 0;
							p1.r = 0;
							dcloud->points.push_back(p1);

							pcl::PointXYZRGBNormal p1n = p1;
							p1n.x += 0.01*dnx;
							p1n.y += 0.01*dny;
							p1n.z += 0.01*dnz;

							char buf [1024];
							sprintf(buf,"line%i",scloud->points.size());
							viewer->addLine<pcl::PointXYZRGBNormal>(p,p1,1.0-prob,prob,0,buf);
							//viewer->addLine<pcl::PointXYZRGBNormal>(p,p1, angle <= 0 , angle > 0 ,0,buf);

//							sprintf(buf,"slineN%i",scloud->points.size());
//							viewer->addLine<pcl::PointXYZRGBNormal>(p1,p1n,1,0,1,buf);

//							sprintf(buf,"dlineN%i",scloud->points.size());
//							viewer->addLine<pcl::PointXYZRGBNormal>(p,pn,1,1,0,buf);
						}

                        //double weight = kpInfo*prob*rw*rw;
                        double weight = prob*rw*rw;
						wsum += weight;

						wsx += weight * sx;
						wsy += weight * sy;
						wsz += weight * sz;

						double wsxsx = weight * sx*sx;
						double wsysy = weight * sy*sy;
						double wszsz = weight * sz*sz;

						ATA.coeffRef (0)  += wsysy + wszsz;//a0 * a0;
						ATA.coeffRef (1)  -= weight * sx*sy;//a0 * a1;
						ATA.coeffRef (2)  -= weight * sz*sx;//a0 * a2;


						ATA.coeffRef (7)  += wsxsx + wszsz;//a1 * a1;
						ATA.coeffRef (8)  -= weight * sy*sz;//a1 * a2;

						ATA.coeffRef (14) += wsxsx + wsysy;//a2 * a2;

						ATb.coeffRef (0) += weight * (sy*diffZ -sz*diffY);//a0 * b;
						ATb.coeffRef (1) += weight * (-sx*diffZ + sz*diffX);//a1 * b;
						ATb.coeffRef (2) += weight * (sx*diffY -sy*diffX);//a2 * b;
						ATb.coeffRef (3) += weight * diffX;//a3 * b;
						ATb.coeffRef (4) += weight * diffY;//a4 * b;
						ATb.coeffRef (5) += weight * diffZ;//a5 * b;
					}


					ATA.coeffRef (4)  -= wsz;//a0 * a4;
					ATA.coeffRef (9)  += wsz;//a1 * a3;
					ATA.coeffRef (5)  += wsy;//a0 * a5;
					ATA.coeffRef (15) -= wsy;//a2 * a3;
					ATA.coeffRef (11) -= wsx;//a1 * a5;
					ATA.coeffRef (16) += wsx;//a2 * a4;
					ATA.coeffRef (21) += wsum;//a3 * a3;
					ATA.coeffRef (28) += wsum;//a4 * a4;
					ATA.coeffRef (35) += wsum;//a5 * a5;

					ATA.coeffRef (6) = ATA.coeff (1);
					ATA.coeffRef (12) = ATA.coeff (2);
					ATA.coeffRef (13) = ATA.coeff (8);
					ATA.coeffRef (18) = ATA.coeff (3);
					ATA.coeffRef (19) = ATA.coeff (9);
					ATA.coeffRef (20) = ATA.coeff (15);
					ATA.coeffRef (24) = ATA.coeff (4);
					ATA.coeffRef (25) = ATA.coeff (10);
					ATA.coeffRef (26) = ATA.coeff (16);
					ATA.coeffRef (27) = ATA.coeff (22);
					ATA.coeffRef (30) = ATA.coeff (5);
					ATA.coeffRef (31) = ATA.coeff (11);
					ATA.coeffRef (32) = ATA.coeff (17);
					ATA.coeffRef (33) = ATA.coeff (23);
					ATA.coeffRef (34) = ATA.coeff (29);

                    for(int d = 0; d < 6; d++){
                        ATA(d,d) += 1;
                    }

					// Solve A*x = b
					Vector6d x = static_cast<Vector6d> (ATA.inverse () * ATb);
					Eigen::Affine3d transformation = Eigen::Affine3d(constructTransformationMatrix(x(0,0),x(1,0),x(2,0),x(3,0),x(4,0),x(5,0)));
					p = transformation*p;

					double change_t = 0;
					double change_r = 0;
					for(unsigned int k = 0; k < 3; k++){
						change_t += transformation(k,3)*transformation(k,3);
						for(unsigned int l = 0; l < 3; l++){
							if(k == l){ change_r += fabs(1-transformation(k,l));}
							else{		change_r += fabs(transformation(k,l));}
						}
					}
					change_t = sqrt(change_t);

					if(visualizationLvl == 5 && i == 0){
						viewer->removeAllPointClouds();
						viewer->addPointCloud<pcl::PointXYZRGBNormal> (scloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(scloud), "scloud");
						//viewer->addPointCloudNormals<pcl::PointXYZRGBNormal>(scloud,1,0.2,"scloudn");

						viewer->addPointCloud<pcl::PointXYZRGBNormal> (dcloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(dcloud), "dcloud");
						//viewer->addPointCloudNormals<pcl::PointXYZRGBNormal>(scloud,1,0.2,"dcloudn");
						viewer->spin();
						viewer->removeAllPointClouds();
						viewer->removeAllShapes();
					}

					if(change_t < stopval && change_r < stopval){do_inner = false;}
				}


				//new_opt_time += getTime()-new_opt_start;
				//std::cout << p.matrix() << std::endl;
				//Eigen::Matrix4d newpose = p.matrix()*poses[i];
				poses[i] = p.matrix();//newpose;
			}
		}else if(type == PointToPlane && optt == 1){
			for(unsigned int i = 0; i < nr_frames; i++){
				if(getTime()-total_time_start > timeout){break;}
				if(!is_ok[i]){continue;}
				unsigned int count = 0;

				Eigen::Matrix<double, 3, Eigen::Dynamic> & tXi    = transformed_points[i];
				Eigen::Matrix<double, 3, Eigen::Dynamic> & Xi     = points[i];
				Eigen::Matrix<double, 3, Eigen::Dynamic> & tXni   = transformed_normals[i];
				Eigen::Matrix<double, 3, Eigen::Dynamic> & Xni    = normals[i];
				Eigen::VectorXd & informationi                    = informations[i];


				for(unsigned int j = 0; j < nr_frames; j++){
					if(!is_ok[j]){continue;}
					Eigen::Matrix<double, 3, Eigen::Dynamic> & tXj    = transformed_points[j];
					Eigen::Matrix<double, 3, Eigen::Dynamic> & tXnj   = transformed_normals[j];
					Eigen::VectorXd & informationj                    = informations[j];

					std::vector<int> & matchidj = matchids[j][i];
					unsigned int matchesj = matchidj.size();
					std::vector<int> & matchidi = matchids[i][j];
					unsigned int matchesi = matchidi.size();

                    std::vector<double> & matchdistj = matchdists[j][i];
                    std::vector<double> & matchdisti = matchdists[i][j];


					for(unsigned int ki = 0; ki < matchesi; ki++){
						int kj = matchidi[ki];
						if( kj == -1 ){continue;}
						if( kj >=  matchesj){continue;}
						if(!onetoone || matchidj[kj] == ki){
							Qp_arr[3*count+0] = tXj(0,kj);
							Qp_arr[3*count+1] = tXj(1,kj);
							Qp_arr[3*count+2] = tXj(2,kj);
							Qn_arr[3*count+0] = tXnj(0,kj);
							Qn_arr[3*count+1] = tXnj(1,kj);
							Qn_arr[3*count+2] = tXnj(2,kj);

							Xp_arr[3*count+0] = tXi(0,ki);
							Xp_arr[3*count+1] = tXi(1,ki);
							Xp_arr[3*count+2] = tXi(2,ki);
							Xn_arr[3*count+0] = tXni(0,ki);
							Xn_arr[3*count+1] = tXni(1,ki);
							Xn_arr[3*count+2] = tXni(2,ki);

							rangeW_arr[count] = 1.0/(1.0/informationi(ki)+1.0/informationj(kj));
							count++;
						}
					}
				}

				if(count == 0){break;}

				typedef Eigen::Matrix<double, 6, 1> Vector6d;
				typedef Eigen::Matrix<double, 6, 6> Matrix6d;

				Matrix6d ATA;
				Vector6d ATb;


				Eigen::Affine3d p = Eigen::Affine3d::Identity();
				bool do_inner = true;
				for(int inner=0; inner < 5 && do_inner; ++inner) {
					ATA.setZero ();
					ATb.setZero ();
					const double & m00 = p(0,0); const double & m01 = p(0,1); const double & m02 = p(0,2); const double & m03 = p(0,3);
					const double & m10 = p(1,0); const double & m11 = p(1,1); const double & m12 = p(1,2); const double & m13 = p(1,3);
					const double & m20 = p(2,0); const double & m21 = p(2,1); const double & m22 = p(2,2); const double & m23 = p(2,3);
					for(unsigned int co = 0; co < count; co++){
						const double & src_x = Xp_arr[3*co+0];
						const double & src_y = Xp_arr[3*co+1];
						const double & src_z = Xp_arr[3*co+2];
						const double & src_nx = Xn_arr[3*co+0];
						const double & src_ny = Xn_arr[3*co+1];
						const double & src_nz = Xn_arr[3*co+2];

						const double & dx = Qp_arr[3*co+0];
						const double & dy = Qp_arr[3*co+1];
						const double & dz = Qp_arr[3*co+2];
						const double & dnx = Qn_arr[3*co+0];
						const double & dny = Qn_arr[3*co+1];
						const double & dnz = Qn_arr[3*co+2];

						const double & rw = rangeW_arr[co];

						const double & sx = m00*src_x + m01*src_y + m02*src_z + m03;
						const double & sy = m10*src_x + m11*src_y + m12*src_z + m13;
						const double & sz = m20*src_x + m21*src_y + m22*src_z + m23;

						const double & nx = m00*src_nx + m01*src_ny + m02*src_nz;
						const double & ny = m10*src_nx + m11*src_ny + m12*src_nz;
						const double & nz = m20*src_nx + m21*src_ny + m22*src_nz;

						if(nx*dnx+ny*dny+nz*dnz < 0){continue;}

						double di = rw*(nx*(sx-dx) + ny*(sy-dy) + nz*(sz-dz));
						double weight = func->getProb(di)*rw*rw;

						const double & a = nz*sy - ny*sz;
						const double & b = nx*sz - nz*sx;
						const double & c = ny*sx - nx*sy;

						ATA.coeffRef (0) += weight * a * a;
						ATA.coeffRef (1) += weight * a * b;
						ATA.coeffRef (2) += weight * a * c;
						ATA.coeffRef (3) += weight * a * nx;
						ATA.coeffRef (4) += weight * a * ny;
						ATA.coeffRef (5) += weight * a * nz;
						ATA.coeffRef (7) += weight * b * b;
						ATA.coeffRef (8) += weight * b * c;
						ATA.coeffRef (9) += weight * b * nx;
						ATA.coeffRef (10) += weight * b * ny;
						ATA.coeffRef (11) += weight * b * nz;
						ATA.coeffRef (14) += weight * c * c;
						ATA.coeffRef (15) += weight * c * nx;
						ATA.coeffRef (16) += weight * c * ny;
						ATA.coeffRef (17) += weight * c * nz;
						ATA.coeffRef (21) += weight * nx * nx;
						ATA.coeffRef (22) += weight * nx * ny;
						ATA.coeffRef (23) += weight * nx * nz;
						ATA.coeffRef (28) += weight * ny * ny;
						ATA.coeffRef (29) += weight * ny * nz;
						ATA.coeffRef (35) += weight * nz * nz;

						const double & d = weight * (nx*dx + ny*dy + nz*dz - nx*sx - ny*sy - nz*sz);

						ATb.coeffRef (0) += a * d;
						ATb.coeffRef (1) += b * d;
						ATb.coeffRef (2) += c * d;
						ATb.coeffRef (3) += nx * d;
						ATb.coeffRef (4) += ny * d;
						ATb.coeffRef (5) += nz * d;
					}

					ATA.coeffRef (6) = ATA.coeff (1);
					ATA.coeffRef (12) = ATA.coeff (2);
					ATA.coeffRef (13) = ATA.coeff (8);
					ATA.coeffRef (18) = ATA.coeff (3);
					ATA.coeffRef (19) = ATA.coeff (9);
					ATA.coeffRef (20) = ATA.coeff (15);
					ATA.coeffRef (24) = ATA.coeff (4);
					ATA.coeffRef (25) = ATA.coeff (10);
					ATA.coeffRef (26) = ATA.coeff (16);
					ATA.coeffRef (27) = ATA.coeff (22);
					ATA.coeffRef (30) = ATA.coeff (5);
					ATA.coeffRef (31) = ATA.coeff (11);
					ATA.coeffRef (32) = ATA.coeff (17);
					ATA.coeffRef (33) = ATA.coeff (23);
					ATA.coeffRef (34) = ATA.coeff (29);

					// Solve A*x = b
					Vector6d x = static_cast<Vector6d> (ATA.inverse () * ATb);
					Eigen::Affine3d transformation = Eigen::Affine3d(constructTransformationMatrix(x(0,0),x(1,0),x(2,0),x(3,0),x(4,0),x(5,0)));
					p = transformation*p;

					double change_t = 0;
					double change_r = 0;
					for(unsigned int k = 0; k < 3; k++){
						change_t += transformation(k,3)*transformation(k,3);
						for(unsigned int l = 0; l < 3; l++){
							if(k == l){ change_r += fabs(1-transformation(k,l));}
							else{		change_r += fabs(transformation(k,l));}
						}
					}
					change_t = sqrt(change_t);
					if(change_t < stopval && change_r < stopval){do_inner = false;}
				}

				//new_opt_time += getTime()-new_opt_start;
				//std::cout << p.matrix() << std::endl;
				Eigen::Matrix4d newpose = p.matrix()*poses[i];
				poses[i] = newpose;
			}
		}else{

			for(unsigned int i = 0; i < nr_frames; i++){
				if(getTime()-total_time_start > timeout){break;}
				if(!is_ok[i]){continue;}
				unsigned int nr_match = 0;
				{
					for(unsigned int j = 0; j < nr_frames; j++){
						if(!is_ok[j]){continue;}
						std::vector<int> & matchidj = matchids[j][i];
						unsigned int matchesj = matchidj.size();
						std::vector<int> & matchidi = matchids[i][j];
						unsigned int matchesi = matchidi.size();


                        std::vector<double> & matchdistj = matchdists[j][i];
                        std::vector<double> & matchdisti = matchdists[i][j];


						for(unsigned int ki = 0; ki < matchesi; ki++){
							int kj = matchidi[ki];
							if( kj == -1 ){continue;}
							if( kj >=  matchesj){continue;}
							if(!onetoone || matchidj[kj] == ki){	nr_match++;}
						}
					}
				}

				Eigen::Matrix3Xd Xp		= Eigen::Matrix3Xd::Zero(3,	nr_match);
				Eigen::Matrix3Xd Xp_ori	= Eigen::Matrix3Xd::Zero(3,	nr_match);
				Eigen::Matrix3Xd Xn		= Eigen::Matrix3Xd::Zero(3,	nr_match);

				Eigen::Matrix3Xd Qp		= Eigen::Matrix3Xd::Zero(3,	nr_match);
				Eigen::Matrix3Xd Qn		= Eigen::Matrix3Xd::Zero(3,	nr_match);
				Eigen::VectorXd  rangeW	= Eigen::VectorXd::Zero(	nr_match);

				int count = 0;

				Eigen::Matrix<double, 3, Eigen::Dynamic> & tXi	= transformed_points[i];
				Eigen::Matrix<double, 3, Eigen::Dynamic> & Xi	= points[i];
				Eigen::Matrix<double, 3, Eigen::Dynamic> & tXni	= transformed_normals[i];
				Eigen::Matrix<double, 3, Eigen::Dynamic> & Xni	= normals[i];
				Eigen::VectorXd & informationi					= informations[i];

				for(unsigned int j = 0; j < nr_frames; j++){
					if(!is_ok[j]){continue;}
					Eigen::Matrix<double, 3, Eigen::Dynamic> & tXj	= transformed_points[j];
					Eigen::Matrix<double, 3, Eigen::Dynamic> & tXnj	= transformed_normals[j];
					Eigen::VectorXd & informationj					= informations[j];

					std::vector<int> & matchidj = matchids[j][i];
					unsigned int matchesj = matchidj.size();
					std::vector<int> & matchidi = matchids[i][j];
					unsigned int matchesi = matchidi.size();


                    std::vector<double> & matchdistj = matchdists[j][i];
                    std::vector<double> & matchdisti = matchdists[i][j];

					for(unsigned int ki = 0; ki < matchesi; ki++){
						int kj = matchidi[ki];
						if( kj == -1 ){continue;}
						if( kj >=  matchesj){continue;}
						if(!onetoone || matchidj[kj] == ki){
							Qp.col(count) = tXj.col(kj);
							Qn.col(count) = tXnj.col(kj);

							Xp_ori.col(count) = Xi.col(ki);
							Xp.col(count) = tXi.col(ki);

							Xn.col(count) = tXni.col(ki);
							rangeW(count) = 1.0/(1.0/informationi(ki)+1.0/informationj(kj));
							count++;
						}
					}
				}

				if(count == 0){break;}

				//showMatches(Xp,Qp);
				for(int inner=0; inner < 5; ++inner) {
					Eigen::MatrixXd residuals;
					switch(type) {
					case PointToPoint:	{residuals = Xp-Qp;} 						break;
					case PointToPlane:	{
						residuals		= Eigen::MatrixXd::Zero(1,	Xp.cols());
						for(int i=0; i<Xp.cols(); ++i) {
							float dx = Xp(0,i)-Qp(0,i);
							float dy = Xp(1,i)-Qp(1,i);
							float dz = Xp(2,i)-Qp(2,i);
							float qx = Qn(0,i);
							float qy = Qn(1,i);
							float qz = Qn(2,i);
							float di = qx*dx + qy*dy + qz*dz;
							residuals(0,i) = di;
						}
					}break;
					default:			{printf("type not set\n");}					break;
					}
					for(unsigned int k=0; k < nr_match; ++k) {residuals.col(k) *= rangeW(k);}

					Eigen::VectorXd  W;
					switch(type) {
					case PointToPoint:	{W = func->getProbs(residuals); } 					break;
					case PointToPlane:	{
						W = func->getProbs(residuals);
						for(int k=0; k<nr_match; ++k) {W(k) = W(k)*float((Xn(0,k)*Qn(0,k) + Xn(1,k)*Qn(1,k) + Xn(2,k)*Qn(2,k)) > 0.0);}
					}	break;
					default:			{printf("type not set\n");} break;
					}

					W = W.array()*rangeW.array()*rangeW.array();
					Xo1 = Xp;
					switch(type) {
					case PointToPoint:	{
						//RigidMotionEstimator::point_to_point(Xp, Qp, W);
						pcl::TransformationFromCorrespondences tfc1;
						for(unsigned int c = 0; c < nr_match; c++){tfc1.add(Eigen::Vector3f(Xp(0,c), Xp(1,c),Xp(2,c)),Eigen::Vector3f(Qp(0,c),Qp(1,c),Qp(2,c)),W(c));}
						Eigen::Affine3d rot = tfc1.getTransformation().cast<double>();
						Xp = rot*Xp;
						Xn = rot.rotation()*Xn;
					}		break;
					case PointToPlane:	{
						point_to_plane2(Xp, Xn, Qp, Qn, W);
					}	break;
					default:  			{printf("type not set\n"); } break;
					}

					double stop1 = (Xp-Xo1).colwise().norm().maxCoeff();
					Xo1 = Xp;
					if(stop1 < 0.001){break; }
				}
				//exit(0);
				pcl::TransformationFromCorrespondences tfc;
				for(unsigned int c = 0; c < nr_match; c++){tfc.add(Eigen::Vector3f(Xp_ori(0,c),Xp_ori(1,c),Xp_ori(2,c)),Eigen::Vector3f(Xp(0,c),Xp(1,c),Xp(2,c)));}
				poses[i] = tfc.getTransformation().cast<double>().matrix();
			}
		}

		Eigen::Matrix4d p0inv = poses[0].inverse();
		for(unsigned int j = 0; j < nr_frames; j++){
			if(!is_ok[j]){continue;}
			poses[j] = p0inv*poses[j];

			Eigen::Matrix<double, 3, Eigen::Dynamic> & tXi	= transformed_points[j];
			Eigen::Matrix<double, 3, Eigen::Dynamic> & Xi	= points[j];
			Eigen::Matrix<double, 3, Eigen::Dynamic> & tXni	= transformed_normals[j];
			Eigen::Matrix<double, 3, Eigen::Dynamic> & Xni	= normals[j];

			Eigen::Matrix4d p = poses[j];
			float m00 = p(0,0); float m01 = p(0,1); float m02 = p(0,2); float m03 = p(0,3);
			float m10 = p(1,0); float m11 = p(1,1); float m12 = p(1,2); float m13 = p(1,3);
			float m20 = p(2,0); float m21 = p(2,1); float m22 = p(2,2); float m23 = p(2,3);

			for(int c = 0; c < Xi.cols(); c++){
				float x = Xi(0,c);
				float y = Xi(1,c);
				float z = Xi(2,c);

				float nx = Xni(0,c);
				float ny = Xni(1,c);
				float nz = Xni(2,c);

				tXi(0,c)		= m00*x + m01*y + m02*z + m03;
				tXi(1,c)		= m10*x + m11*y + m12*z + m13;
				tXi(2,c)		= m20*x + m21*y + m22*z + m23;

				tXni(0,c)		= m00*nx + m01*ny + m02*nz;
				tXni(1,c)		= m10*nx + m11*ny + m12*nz;
				tXni(2,c)		= m20*nx + m21*ny + m22*nz;
			}
		}
		if(isconverged(poses, poses2, stopval, stopval)){break;}
	}

//	exit(0);
	return poses;
}

int popcount_lauradoux(uint64_t *buf, uint32_t size) {
	const uint64_t* data = (uint64_t*) buf;

	const uint64_t m1	= UINT64_C(0x5555555555555555);
	const uint64_t m2	= UINT64_C(0x3333333333333333);
	const uint64_t m4	= UINT64_C(0x0F0F0F0F0F0F0F0F);
	const uint64_t m8	= UINT64_C(0x00FF00FF00FF00FF);
	const uint64_t m16 = UINT64_C(0x0000FFFF0000FFFF);
	const uint64_t h01 = UINT64_C(0x0101010101010101);

	uint32_t bitCount = 0;
	uint32_t i, j;
	uint64_t count1, count2, half1, half2, acc;
	uint64_t x;
	uint32_t limit30 = size - size % 30;

	// 64-bit tree merging (merging3)
	for (i = 0; i < limit30; i += 30, data += 30) {
		acc = 0;
		for (j = 0; j < 30; j += 3) {
			count1	=	data[j];
			count2	=	data[j+1];
			half1	=	data[j+2];
			half2	=	data[j+2];
			half1	&=	m1;
			half2	= (half2	>> 1) & m1;
			count1 -= (count1 >> 1) & m1;
			count2 -= (count2 >> 1) & m1;
			count1 +=	half1;
			count2 +=	half2;
			count1	= (count1 & m2) + ((count1 >> 2) & m2);
			count1 += (count2 & m2) + ((count2 >> 2) & m2);
			acc		+= (count1 & m4) + ((count1 >> 4) & m4);
		}
		acc = (acc & m8) + ((acc >>	8)	& m8);
		acc = (acc			 +	(acc >> 16)) & m16;
		acc =	acc			 +	(acc >> 32);
		bitCount += (uint32_t)acc;
	}

	for (i = 0; i < size - limit30; i++) {
		x = data[i];
		x =	x			 - ((x >> 1)	& m1);
		x = (x & m2) + ((x >> 2)	& m2);
		x = (x			 +	(x >> 4)) & m4;
		bitCount += (uint32_t)((x * h01) >> 56);
	}
	return bitCount;
}

void MassRegistrationPPR2::rematchKeyPoints(std::vector<Eigen::Matrix4d> poses, std::vector<Eigen::Matrix4d> prev_poses, bool first){
	printf("rematchKeyPoints\n");
	double total_rematchKeyPoints_time_start = getTime();
	unsigned int nr_frames = poses.size();

	int max_kps = 0;
    for(unsigned int i = 0; i < nr_frames; i++){
        printf("max_kps: %i kp_nr_arraypoints: %i\n",max_kps, kp_nr_arraypoints[i]);
        max_kps = std::max(max_kps,kp_nr_arraypoints[i]);
    }
	printf("max_kps: %i\n",max_kps);

    if(max_kps == 0){return;}

	double *	best_src	= new double[max_kps];
    double *	best_src2	= new double[max_kps];
	double *	best_src_e	= new double[max_kps];
	double *	best_src_f	= new double[max_kps];
	int *		best_src_id	= new int[max_kps];

	double *	best_dst	= new double[max_kps];
	double *	best_dst_e	= new double[max_kps];
	double *	best_dst_f	= new double[max_kps];
	int *		best_dst_id	= new int[max_kps];

	for(unsigned int i = 0; i < nr_frames; i++){
		kp_matches.resize(nr_frames);
		for(unsigned int j = 0; j < nr_frames; j++){
			kp_matches[i].resize(nr_frames);
		}
	}


	int ignores = 0;
bool useOrb = true;
if(useOrb){
	for(unsigned int i = 0; i < nr_frames; i++){
		const unsigned int src_nr_ap = kp_nr_arraypoints[i];
		if(src_nr_ap == 0){continue;}
		double * src_ap = kp_arraypoints[i];
		double * src_an = kp_arraynormals[i];
		double * src_ai = kp_arrayinformations[i];
		uint64_t * src_data = kp_arraydescriptors[i];

		for(unsigned int j = 0; j < nr_frames; j++){
			if(i == j){continue;}

			const unsigned int dst_nr_ap = kp_nr_arraypoints[j];
			if(dst_nr_ap == 0){continue;}

            for(int k = 0; k < src_nr_ap; k++){
                best_src[k]		= 999999999999;
                best_src2[k]		= 999999999999;
				best_src_id[k]	= -1;
			}
            for(int k = 0; k < dst_nr_ap; k++){
                best_dst[k]		= 999999999999;
				best_dst_id[k]	= -1;
			}


			double * dst_ap = kp_arraypoints[j];
			double * dst_an = kp_arraynormals[j];
			double * dst_ai = kp_arrayinformations[j];
			uint64_t * dst_data = kp_arraydescriptors[j];

			Eigen::Affine3d rp = Eigen::Affine3d(poses[j].inverse()*poses[i]);

			if(!first){
				Eigen::Affine3d prev_rp = Eigen::Affine3d(prev_poses[j].inverse()*prev_poses[i]);

				Eigen::Affine3d diff = prev_rp.inverse()*rp;

				double change_trans = 0;
				double change_rot = 0;
				double dt = 0;
				for(unsigned int k = 0; k < 3; k++){
					dt += diff(k,3)*diff(k,3);
					for(unsigned int l = 0; l < 3; l++){
						if(k == l){ change_rot += fabs(1-diff(k,l));}
						else{		change_rot += fabs(diff(k,l));}
					}
				}
				change_trans += sqrt(dt);

				if(change_trans < 1*stopval && change_rot < 1*stopval){ignores++;continue;}
			}

			const double & m00 = rp(0,0); const double & m01 = rp(0,1); const double & m02 = rp(0,2); const double & m03 = rp(0,3);
			const double & m10 = rp(1,0); const double & m11 = rp(1,1); const double & m12 = rp(1,2); const double & m13 = rp(1,3);
			const double & m20 = rp(2,0); const double & m21 = rp(2,1); const double & m22 = rp(2,2); const double & m23 = rp(2,3);

            float di = 10.0;
            float pi = 10.0;

			#pragma omp parallel for num_threads(8)
			for(unsigned int src_k = 0; src_k < src_nr_ap; ++src_k) {
				const double & src_x = src_ap[3*src_k+0];
				const double & src_y = src_ap[3*src_k+1];
				const double & src_z = src_ap[3*src_k+2];

				double sx = m00*src_x + m01*src_y + m02*src_z + m03;
				double sy = m10*src_x + m11*src_y + m12*src_z + m13;
				double sz = m20*src_x + m21*src_y + m22*src_z + m23;

				unsigned int src_k4 = src_k*4;
				const uint64_t s1 = src_data[src_k4+0];
				const uint64_t s2 = src_data[src_k4+1];
				const uint64_t s3 = src_data[src_k4+2];
				const uint64_t s4 = src_data[src_k4+3];

				uint64_t xordata [4];
				for(unsigned int dst_k = 0; dst_k < dst_nr_ap; ++dst_k) {
					double dx = sx-dst_ap[3*dst_k+0];
					double dy = sy-dst_ap[3*dst_k+1];
					double dz = sz-dst_ap[3*dst_k+2];

                    double p_dist = sqrt(dx*dx+dy*dy+dz*dz);

					unsigned int dst_k4 = dst_k*4;
					const uint64_t d1 = dst_data[dst_k4+0];
					const uint64_t d2 = dst_data[dst_k4+1];
					const uint64_t d3 = dst_data[dst_k4+2];
					const uint64_t d4 = dst_data[dst_k4+3];

					xordata[0] = s1 ^ d1;
					xordata[1] = s2 ^ d2;
					xordata[2] = s3 ^ d3;
					xordata[3] = s4 ^ d4;

					int cnt = popcount_lauradoux(xordata, 4);
					double f_dist = float(cnt)/256.0f;

                    float d = f_dist*di + p_dist*pi;

                    if(d < best_src[src_k]){

                        best_src2[src_k] = best_src[src_k];
                        best_src_id[src_k] = dst_k;
                        best_src[src_k] = d;
                    }else if(d < best_src2[src_k]){
                        best_src2[src_k] = d;
                    }



					if(d < best_dst[dst_k]){
						best_dst_id[dst_k] = src_k;
						best_dst[dst_k] = d;
					}

				}
			}

			std::vector< TestMatch > matches;
            double threshold = 2.0;
            double ratiothreshold = 0.8;

			int nr_matches = 0;
			for(unsigned int src_k = 0; src_k < src_nr_ap; ++src_k) {
				int dst_k = best_src_id[src_k];

                if(best_dst_id[dst_k] != src_k){continue;}//One to one
                if(best_src[src_k] > threshold){continue;}//threshold
                if(best_src[src_k]/best_src2[src_k] < ratiothreshold){continue;}//ratiothreshold

				if(dst_an[3*dst_k+0] > 1 || src_an[3*src_k+0] > 1){continue;}

				nr_matches++;

				matches.push_back(TestMatch(src_k, dst_k,best_src[src_k]));
			}
			kp_matches[i][j] = matches;
		}
	}
}


bool useSurf = false;
if(useSurf){
    for(unsigned int i = 0; i < nr_frames; i++){
        const unsigned int src_nr_ap = kp_nr_arraypoints[i];
        if(src_nr_ap == 0){continue;}
        double * src_ap = kp_arraypoints[i];
        double * src_an = kp_arraynormals[i];
        double * src_ai = kp_arrayinformations[i];
        float* src_data = (float*)(kp_arraydescriptors[i]);

        for(unsigned int j = 0; j < nr_frames; j++){
            if(i == j){continue;}

            const unsigned int dst_nr_ap = kp_nr_arraypoints[j];
            if(dst_nr_ap == 0){continue;}

            for(int k = 0; k < src_nr_ap; k++){
                best_src[k]		= 999999999999;
                best_src_id[k]	= -1;
            }
            for(int k = 0; k < dst_nr_ap; k++){
                best_dst[k]		= 999999999999;
                best_dst_id[k]	= -1;
            }


            double * dst_ap = kp_arraypoints[j];
            double * dst_an = kp_arraynormals[j];
            double * dst_ai = kp_arrayinformations[j];
            float* dst_data = (float*)(kp_arraydescriptors[j]);

            Eigen::Affine3d rp = Eigen::Affine3d(poses[j].inverse()*poses[i]);

            if(!first){
                Eigen::Affine3d prev_rp = Eigen::Affine3d(prev_poses[j].inverse()*prev_poses[i]);

                Eigen::Affine3d diff = prev_rp.inverse()*rp;

                double change_trans = 0;
                double change_rot = 0;
                double dt = 0;
                for(unsigned int k = 0; k < 3; k++){
                    dt += diff(k,3)*diff(k,3);
                    for(unsigned int l = 0; l < 3; l++){
                        if(k == l){ change_rot += fabs(1-diff(k,l));}
                        else{		change_rot += fabs(diff(k,l));}
                    }
                }
                change_trans += sqrt(dt);

                if(change_trans < 1*stopval && change_rot < 1*stopval){ignores++;continue;}
            }

            const double & m00 = rp(0,0); const double & m01 = rp(0,1); const double & m02 = rp(0,2); const double & m03 = rp(0,3);
            const double & m10 = rp(1,0); const double & m11 = rp(1,1); const double & m12 = rp(1,2); const double & m13 = rp(1,3);
            const double & m20 = rp(2,0); const double & m21 = rp(2,1); const double & m22 = rp(2,2); const double & m23 = rp(2,3);

            #pragma omp parallel for num_threads(8)
            for(unsigned int src_k = 0; src_k < src_nr_ap; ++src_k) {
                const double & src_x = src_ap[3*src_k+0];
                const double & src_y = src_ap[3*src_k+1];
                const double & src_z = src_ap[3*src_k+2];

                double sx = m00*src_x + m01*src_y + m02*src_z + m03;
                double sy = m10*src_x + m11*src_y + m12*src_z + m13;
                double sz = m20*src_x + m21*src_y + m22*src_z + m23;



                for(unsigned int dst_k = 0; dst_k < dst_nr_ap; ++dst_k) {
                    double dx = sx-dst_ap[3*dst_k+0];
                    double dy = sy-dst_ap[3*dst_k+1];
                    double dz = sz-dst_ap[3*dst_k+2];


                    double f_dist = 0;

                    for(unsigned int f_k = 0; f_k < 128; ++f_k) {
                        double diff = src_data[src_k*128 + f_k] - dst_data[dst_k*128 + f_k];
                        f_dist += diff*diff;
                    }

                    float d = f_dist;//*di + p_dist*pi;

                    if(d < best_src[src_k]){
                        best_src_id[src_k] = dst_k;
                        best_src[src_k] = d;
                    }

                    if(d < best_dst[dst_k]){
                        best_dst_id[dst_k] = src_k;
                        best_dst[dst_k] = d;
                    }

                }
            }

            std::vector< TestMatch > matches;
            double threshold = 0.25;
            int nr_matches = 0;
            for(unsigned int src_k = 0; src_k < src_nr_ap; ++src_k) {
                int dst_k = best_src_id[src_k];

                if(best_dst_id[dst_k] != src_k || best_src[src_k] > threshold){continue;}//One to one

                if(dst_an[3*dst_k+0] > 1 || src_an[3*src_k+0] > 1){continue;}

                nr_matches++;

                matches.push_back(TestMatch(src_k, dst_k,best_src[src_k]));
            }
            kp_matches[i][j] = matches;
        }
    }
}
if(false){
    for(unsigned int i = 0; i < nr_frames; i++){
        if(frameid[i] == -1){continue;}
        for(unsigned int j = 0; j < nr_frames; j++){
            if(frameid[j] == -1){continue;}
            if(kp_matches[i][j].size() == 0){continue;}
            unsigned char * src_data = (unsigned char *)(model->frames[frameid[i]]->rgb.data);
            unsigned char * dst_data = (unsigned char *)(model->frames[frameid[j]]->rgb.data);

            cv::Mat img;
            img.create(480,2*640,CV_8UC3);
            unsigned char * imgdata = (unsigned char *)img.data;
            for(int w = 0; w < 640; w++){
                for(int h = 0; h < 480; h++){
                    int ind = h*640+w;

                    int ind2 = h*2*640+w;
                    int ind3 = h*2*640+w+640;

                    imgdata[3*ind2+0] = src_data[3*ind+0];
                    imgdata[3*ind2+1] = src_data[3*ind+1];
                    imgdata[3*ind2+2] = src_data[3*ind+2];

                    imgdata[3*ind3+0] = dst_data[3*ind+0];
                    imgdata[3*ind3+1] = dst_data[3*ind+1];
                    imgdata[3*ind3+2] = dst_data[3*ind+2];
                }
            }

            std::vector<cv::KeyPoint> & src_keypoints = model->all_keypoints[i];
            std::vector<cv::KeyPoint> & dst_keypoints = model->all_keypoints[j];

            std::vector< TestMatch > matches = kp_matches[i][j];
            for(unsigned int k = 0; k < matches.size(); ++k) {
                cv::KeyPoint & spt = src_keypoints[matches[k].src];
                cv::KeyPoint & dpt = dst_keypoints[matches[k].dst];

                //printf("%i -> ")

                cv::line(img, spt.pt, cv::Point(dpt.pt.x+640,dpt.pt.y), cv::Scalar(255,0,255));
            }

            cv::imshow( "matches", img );
            cv::waitKey(0);

            //for(int j = 0; j < width*height; j++){maskdata[j] = 255*maskvec[j];}
        }
    }
}

	delete[] best_src;
	delete[] best_src_e;
	delete[] best_src_f;
	delete[] best_src_id;
	delete[] best_dst;
	delete[] best_dst_e;
	delete[] best_dst_f;
	delete[] best_dst_id;

	if(visualizationLvl > 0){printf("total rematchKeyPoints time:          %5.5f\n",getTime()-total_rematchKeyPoints_time_start);}

}

MassFusionResults MassRegistrationPPR2::getTransforms(std::vector<Eigen::Matrix4d> poses){
	if(visualizationLvl > 0){printf("start MassRegistrationPPR2::getTransforms(std::vector<Eigen::Matrix4d> poses)\n");}

	unsigned int nr_frames = informations.size();
	if(poses.size() != nr_frames){
        printf("ERROR: poses.size()  = %i != informations.size() %i\n",poses.size(), informations.size());
		return MassFusionResults();
	}

	fast_opt = false;
	if(fast_opt){
		printf("debugging... setting nr frames to 3: %s :: %i\n",__FILE__,__LINE__);
		nr_frames = 3;
	}

	for(unsigned int i = 0; i < nr_frames; i++){
		Eigen::Matrix4d p = poses[i];
		float m00 = p(0,0); float m01 = p(0,1); float m02 = p(0,2); float m03 = p(0,3);
		float m10 = p(1,0); float m11 = p(1,1); float m12 = p(1,2); float m13 = p(1,3);
		float m20 = p(2,0); float m21 = p(2,1); float m22 = p(2,2); float m23 = p(2,3);

		if(!is_ok[i]){continue;}

        matchids[i].resize(nr_frames);
        matchdists[i].resize(nr_frames);

		Eigen::Matrix<double, 3, Eigen::Dynamic> & X	= points[i];
		Eigen::Matrix<double, 3, Eigen::Dynamic> & Xn	= normals[i];
		Eigen::Matrix<double, 3, Eigen::Dynamic> & tX	= transformed_points[i];
		Eigen::Matrix<double, 3, Eigen::Dynamic> & tXn	= transformed_normals[i];
		int count = nr_datas[i];
		for(int c = 0; c < count; c++){
			float x = X(0,c);
			float y = X(1,c);
			float z = X(2,c);
			float xn = Xn(0,c);
			float yn = Xn(1,c);
			float zn = Xn(2,c);

			tX(0,c)		= m00*x + m01*y + m02*z + m03;
			tX(1,c)		= m10*x + m11*y + m12*z + m13;
			tX(2,c)		= m20*x + m21*y + m22*z + m23;
			tXn(0,c)	= m00*xn + m01*yn + m02*zn;
			tXn(1,c)	= m10*xn + m11*yn + m12*zn;
			tXn(2,c)	= m20*xn + m21*yn + m22*zn;
		}

	}

	func->reset();
	kpfunc->reset();

	int imgcount = 0;
	char buf [1024];
	if(visualizationLvl > 0){
		std::vector<Eigen::MatrixXd> Xv;
		for(unsigned int j = 0; j < nr_frames; j++){Xv.push_back(transformed_points[j]);}
		sprintf(buf,"image%5.5i.png",imgcount++);
		show(Xv,false,std::string(buf),imgcount);
	}

	rematch_time = 0;
	residuals_time = 0;
	opt_time = 0;
	computeModel_time = 0;
	setup_matches_time = 0;
	setup_equation_time = 0;
	setup_equation_time2 = 0;
	solve_equation_time = 0;
	total_time_start = getTime();



	bool first = true;
	std::vector<Eigen::Matrix4d> poses0 = poses;
	//rematchKeyPoints(poses,poses0,first);

    for(int funcupdate=0; funcupdate < 100; ++funcupdate) {
		if(getTime()-total_time_start > timeout){break;}
		if(visualizationLvl == 2){std::vector<Eigen::MatrixXd> Xv;for(unsigned int j = 0; j < nr_frames; j++){Xv.push_back(transformed_points[j]);}sprintf(buf,"image%5.5i.png",imgcount++);show(Xv,false,std::string(buf),imgcount);}

        for(int rematching=0; rematching < 10; ++rematching) {
			if(visualizationLvl == 3){std::vector<Eigen::MatrixXd> Xv;for(unsigned int j = 0; j < nr_frames; j++){Xv.push_back(transformed_points[j]);}sprintf(buf,"image%5.5i.png",imgcount++);show(Xv,false,std::string(buf),imgcount);}
			std::vector<Eigen::Matrix4d> poses1 = poses;

			double rematch_time_start = getTime();
            rematch(poses,poses0,first);
			//rematchKeyPoints(poses,poses0,first);
			first = false;
			poses0 = poses;
			rematch_time += getTime()-rematch_time_start;

            for(int lala = 0; lala < 10; lala++){
                if(visualizationLvl > 0){
					printf("funcupdate: %i rematching: %i lala: %i\n",funcupdate,rematching,lala);
					printf("total_time:          %5.5f\n",getTime()-total_time_start);
					printf("rematch_time:        %5.5f\n",rematch_time);
					printf("compM residuals_time:%5.5f\n",residuals_time);
					printf("computeModel:        %5.5f\n",computeModel_time);
					printf("opt_time:            %5.5f\n",opt_time);
				}
				if(visualizationLvl == 4){std::vector<Eigen::MatrixXd> Xv;for(unsigned int j = 0; j < nr_frames; j++){Xv.push_back(transformed_points[j]);}sprintf(buf,"image%5.5i.png",imgcount++);show(Xv,false,std::string(buf),imgcount);}
				std::vector<Eigen::Matrix4d> poses2b = poses;

				double residuals_time_start = getTime();
                Eigen::MatrixXd all_residuals = getAllResiduals(poses);

//				Eigen::MatrixXd all_KPresiduals = getAllKpResiduals(poses);
				residuals_time += getTime()-residuals_time_start;

				double computeModel_time_start = getTime();
                func->computeModel(all_residuals);
//				kpfunc->computeModel(all_KPresiduals);
                stopval = func->getNoise()*0.1;
//                printf("reg: %f noise:%f stopval: %f\n",func->regularization,func->noiseval,stopval);

//                double stopval1 = func->getNoise()*0.1;
//                double stopval2 = kpfunc->getNoise()*0.1;
//                if(func->noiseval > 0 && kpfunc->noiseval > 0){ stopval = std::min(stopval1,stopval2);}
//                else if(func->noiseval > 0){                    stopval = func->getNoise()*0.1;}
//                else if(kpfunc->noiseval > 0){                  stopval = kpfunc->getNoise()*0.1;}
//                else{                                           break;}
//                printf("reg: %f noise:%f stopval: %f\n",func->regularization,func->noiseval,stopval);
//                printf("kpreg: %f kpnoise:%f stopval: %f\n",kpfunc->regularization,kpfunc->noiseval,stopval);

				computeModel_time += getTime()-computeModel_time_start;

				double opt_time_start = getTime();
				poses = optimize(poses);
				opt_time += getTime()-opt_time_start;

				if(isconverged(poses, poses2b, stopval, stopval)){break;}
			}
            if(isconverged(poses, poses1, stopval, stopval)){break;}
		}

		double noise_before = func->getNoise();
        func->update();
		double noise_after = func->getNoise();
        double ratio = noise_after/noise_before;

		double kpnoise_before = kpfunc->getNoise();
		kpfunc->update();
		double kpnoise_after = kpfunc->getNoise();
        double kpratio = kpnoise_after/kpnoise_before;

        double reg1 = func->regularization;
        double reg2 = kpfunc->regularization;
        func->regularization = std::max(reg1,reg2);
        kpfunc->regularization = std::max(reg1,reg2);

		printf("func->noiseval: %f func->regularization: %f\n",func->noiseval,func->regularization);
		//printf("ratio: %f kpratio: %f\n",ratio,kpratio);
        //if(func->noiseval >= kpfunc->noiseval && 0.01*func->noiseval   < func->regularization){break;}
        //if(func->noiseval < kpfunc->noiseval  && 0.01*kpfunc->noiseval < kpfunc->regularization){break;}
        //if(ratio > 0.99){break;}
		if(func->noiseval > 10.0*func->regularization && ratio > 0.75){break;}
	}

	printf("total_time:          %5.5f\n",getTime()-total_time_start);
	printf("rematch_time:        %5.5f\n",rematch_time);
	printf("compM residuals_time:%5.5f\n",residuals_time);
	printf("computeModel:        %5.5f\n",computeModel_time);
	printf("opt_time:            %5.5f\n",opt_time);
//	printf("setup_matches_time:  %5.5f\n",setup_matches_time);
//	printf("setup_equation_time: %5.5f\n",setup_equation_time);
//	printf("solve_equation_time: %5.5f\n",solve_equation_time);

	if(visualizationLvl > 0){
		std::vector<Eigen::MatrixXd> Xv;
		for(unsigned int j = 0; j < nr_frames; j++){Xv.push_back(transformed_points[j]);}
		sprintf(buf,"image%5.5i.png",imgcount++);
		show(Xv,false,std::string(buf),imgcount);
	}

	Eigen::Matrix4d firstinv = poses.front().inverse();
	for(int i = 0; i < nr_frames; i++){poses[i] = firstinv*poses[i];}

	printf("stop MassRegistrationPPR2::getTransforms(std::vector<Eigen::Matrix4d> guess)\n");
	//exit(0);
	return MassFusionResults(poses,-1);
}

}
