#include "registration/MassRegistrationPPR.h"

#include <iostream>
#include <fstream>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace reglib
{

MassRegistrationPPR::MassRegistrationPPR(double startreg, bool visualize){
	type					= PointToPlane;
	//type					= PointToPoint;
	use_PPR_weight			= true;
	use_features			= true;
	normalize_matchweights	= true;

	DistanceWeightFunction2PPR2 * dwf = new DistanceWeightFunction2PPR2();
	dwf->update_size		= true;
	dwf->startreg			= startreg;
	dwf->debugg_print		= false;
	func					= dwf;

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
}
MassRegistrationPPR::~MassRegistrationPPR(){

	for(unsigned int i = 0; i < arraypoints.size(); i++){delete[] arraypoints[i];}
	for(unsigned int i = 0; i < arraynormals.size(); i++){delete[] arraynormals[i];}
	for(unsigned int i = 0; i < arraycolors.size(); i++){delete[] arraycolors[i];}
	for(unsigned int i = 0; i < arrayinformations.size(); i++){delete[] arrayinformations[i];}

	for(unsigned int i = 0; i < trees3d.size(); i++){delete trees3d[i];}
	for(unsigned int i = 0; i < a3dv.size(); i++){delete a3dv[i];}
	delete func;

	delete[] Qp_arr;
	delete[] Qn_arr;
	delete[] Xp_arr;
	delete[] Xn_arr;
	delete[] rangeW_arr;
}

void MassRegistrationPPR::addModelData(Model * model_, bool submodels){
	printf("addModelData\n");

	if(submodels){
		for(unsigned int i = 0; i < model_->submodels.size(); i++){
			addData(model_->submodels[i]->getPCLnormalcloud(1,false));
		}
	}else{
		//setData(model_->frames,model_->modelmasks);

//		for(unsigned int i = 0; i < model->submodels.size(); i++){
//			addData(model->submodels[i]->getPCLnormalcloud(1,false));
//		}

		unsigned int nr_frames = model_->frames.size();
		for(unsigned int i = 0; i < nr_frames; i++){
			addData(model_->frames[i], model_->modelmasks[i]);
		}
	}
}


void MassRegistrationPPR::setData(std::vector< pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr > all_clouds){
	double total_load_time_start = getTime();
	unsigned int nr_frames = all_clouds.size();

	if(arraypoints.size() > 0){
		for(unsigned int i = 0; i < arraypoints.size(); i++){
			delete[] arraypoints[i];
		}
	}

	if(a3dv.size() > 0){
		for(unsigned int i = 0; i < a3dv.size(); i++){
			delete a3dv[i];
		}
	}

	if(trees3d.size() > 0){
		for(unsigned int i = 0; i < trees3d.size(); i++){
			delete trees3d[i];
		}
	}

	nr_matches.resize(nr_frames);
	matchids.resize(nr_frames);
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

void MassRegistrationPPR::setData(std::vector<RGBDFrame*> frames_,std::vector<ModelMask *> mmasks_){
	double total_load_time_start = getTime();
	//	frames = frames_;
	//	mmasks = mmasks_;

	unsigned int nr_frames = frames_.size();

	if(arraypoints.size() > 0){
		for(unsigned int i = 0; i < arraypoints.size(); i++){
			delete[] arraypoints[i];
		}
	}

	if(a3dv.size() > 0){
		for(unsigned int i = 0; i < a3dv.size(); i++){
			delete a3dv[i];
		}
	}

	if(trees3d.size() > 0){
		for(unsigned int i = 0; i < trees3d.size(); i++){
			delete trees3d[i];
		}
	}

	for(unsigned int i = 0; i < nr_frames; i++){
        //printf("loading data for %i, nomask %i\n",i,nomask);
		addData(frames_[i], mmasks_[i]);
	}
	/*
	for(unsigned int i = 0; i < nr_frames; i++){
		printf("loading data for %i, nomask %i\n",i,nomask);
		addData(frames[i], mmasks[i]);
	}
*/
	/*
	nr_matches.resize(nr_frames);
	matchids.resize(nr_frames);
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
*/
	/*
	for(unsigned int i = 0; i < nr_frames; i++){
		printf("loading data for %i, nomask %i\n",i,nomask);
		bool * maskvec		= mmasks[i]->maskvec;
		unsigned char  * rgbdata		= (unsigned char	*)(frames[i]->rgb.data);
		unsigned short * depthdata		= (unsigned short	*)(frames[i]->depth.data);
		float		   * normalsdata	= (float			*)(frames[i]->normals.data);

		Camera * camera				= frames[i]->camera;
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
				if(maskvec[ind] || nomask){
					float z = idepth*float(depthdata[ind]);
					float xn = normalsdata[3*ind+0];
					if(z > 0.2 && xn != 2){count++;}
				}
			}
		}

		printf("count: %i\n",count);

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
		Eigen::Matrix<double, 3, Eigen::Dynamic> & tX	= transformed_points[i];
		Eigen::Matrix<double, 3, Eigen::Dynamic> & tXn	= transformed_normals[i];
		Eigen::VectorXd information (count);

		int c = 0;
		for(unsigned int w = 0; w < width; w+=maskstep){
			for(unsigned int h = 0; h < height;h+=maskstep){
				if(c == count){continue;}
				int ind = h*width+w;
				if(maskvec[ind] || nomask){
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

						ai[c] = pow(fabs(z),-1);//1.0/(z*z);

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
	}
*/
    //printf("total load time:          %5.5f\n",getTime()-total_load_time_start);
}

void MassRegistrationPPR::rematch(std::vector<Eigen::Matrix4d> poses, std::vector<Eigen::Matrix4d> prev_poses, bool first){
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

			double * ap = arraypoints[i];
			const unsigned int nr_ap = nr_arraypoints[i];

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

					//printf("%f %f\n",change_trans,change_rot);

					if(change_trans < 1*stopval && change_rot < 1*stopval){ignores++;continue;}
				}

				const double & m00 = rp(0,0); const double & m01 = rp(0,1); const double & m02 = rp(0,2); const double & m03 = rp(0,3);
				const double & m10 = rp(1,0); const double & m11 = rp(1,1); const double & m12 = rp(1,2); const double & m13 = rp(1,3);
				const double & m20 = rp(2,0); const double & m21 = rp(2,1); const double & m22 = rp(2,2); const double & m23 = rp(2,3);

				std::vector<int> & matchid = matchids[i][j];
				matchid.resize(nr_ap);
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
				}
				nr_matches[i] += matchid.size();
			}
		}
		//printf("ignores: %i rematches: %i\n",ignores,nr_frames*(nr_frames-1)-ignores);
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


Eigen::MatrixXd MassRegistrationPPR::getAllResiduals(std::vector<Eigen::Matrix4d> poses){
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

						float tx = m00*src_x + m01*src_y + m02*src_z + m03;
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

void MassRegistrationPPR::clearData(){}

void MassRegistrationPPR::addData(RGBDFrame* frame, ModelMask * mmask){
	double total_load_time_start = getTime();
	frames.push_back(frame);
	mmasks.push_back(mmask);

	nr_matches.push_back(	0);
	matchids.push_back(		std::vector< std::vector<int> >() );
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

	unsigned int i = frames.size()-1;
	//printf("loading data for %i, nomask %i\n",i,nomask);
	bool * maskvec		= mmasks[i]->maskvec;
	unsigned char  * rgbdata		= (unsigned char	*)(frames[i]->rgb.data);
	unsigned short * depthdata		= (unsigned short	*)(frames[i]->depth.data);
	float		   * normalsdata	= (float			*)(frames[i]->normals.data);

	Camera * camera				= frames[i]->camera;
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
			if(maskvec[ind] || nomask){
				float z = idepth*float(depthdata[ind]);
				float xn = normalsdata[3*ind+0];
				if(z > 0.2 && xn != 2){count++;}
			}
		}
	}

//	printf("count: %i\n",count);

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
	//matchids[i].resize(nr_frames);
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
	Eigen::Matrix<double, 3, Eigen::Dynamic> & tX	= transformed_points[i];
	Eigen::Matrix<double, 3, Eigen::Dynamic> & tXn	= transformed_normals[i];
	Eigen::VectorXd information (count);

	int c = 0;
	for(unsigned int w = 0; w < width; w+=maskstep){
		for(unsigned int h = 0; h < height;h+=maskstep){
			if(c == count){continue;}
			int ind = h*width+w;
			if(maskvec[ind] || nomask){
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
	//printf("total load time:          %5.5f\n",getTime()-total_load_time_start);
}

void MassRegistrationPPR::addData(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud){
	double total_load_time_start = getTime();

	nr_matches.push_back(	0);
	matchids.push_back(		std::vector< std::vector<int> >() );
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
	for(unsigned int j = 0; j < cloud->points.size(); j+=maskstep){
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
	for(unsigned int j = 0; j < cloud->points.size(); j+=maskstep){
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

std::vector<Eigen::Matrix4d> MassRegistrationPPR::optimize(std::vector<Eigen::Matrix4d> poses){
	bool onetoone = true;
	unsigned int nr_frames = poses.size();
	Eigen::MatrixXd Xo1;

	int optt = 2;
	for(int outer=0; outer < 30; ++outer) {
		if(getTime()-total_time_start > timeout){break;}

		//printf("outer: %i\n",outer);

		std::vector<Eigen::Matrix4d> poses2 = poses;
		if(type == PointToPlane && optt == 2){
			for(unsigned int i = 0; i < nr_frames; i++){
				if(getTime()-total_time_start > timeout){break;}
				if(!is_ok[i]){continue;}

				unsigned int count = 0;
				double * api = arraypoints[i];
				double * ani = arraynormals[i];
				double * aci = arraycolors[i];
				double * aii = arrayinformations[i];
				const unsigned int nr_api = nr_arraypoints[i];

				Eigen::Affine3d rpi = Eigen::Affine3d(poses[i]);

				for(unsigned int j = 0; j < nr_frames; j++){
					if(!is_ok[j]){continue;}
					if(i == j){continue;}

					double * apj = arraypoints[j];
					double * anj = arraynormals[j];
					double * acj = arraycolors[j];
					double * aij = arrayinformations[j];
					const unsigned int nr_apj = nr_arraypoints[j];

					std::vector<int> & matchidj = matchids[j][i];
					unsigned int matchesj = matchidj.size();
					std::vector<int> & matchidi = matchids[i][j];
					unsigned int matchesi = matchidi.size();



					Eigen::Affine3d rpj = Eigen::Affine3d(poses[j]);
					const double & mj00 = rpj(0,0); const double & mj01 = rpj(0,1); const double & mj02 = rpj(0,2); const double & mj03 = rpj(0,3);
					const double & mj10 = rpj(1,0); const double & mj11 = rpj(1,1); const double & mj12 = rpj(1,2); const double & mj13 = rpj(1,3);
					const double & mj20 = rpj(2,0); const double & mj21 = rpj(2,1); const double & mj22 = rpj(2,2); const double & mj23 = rpj(2,3);

					for(unsigned int ki = 0; ki < matchesi; ki++){
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
				}

				if(count == 0){break;}

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
						double weight = prob*rw*rw;

						if(visualizationLvl == 5){
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

					if(visualizationLvl == 5){
						viewer->removeAllPointClouds();
						viewer->addPointCloud<pcl::PointXYZRGBNormal> (scloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(scloud), "scloud");
						viewer->addPointCloud<pcl::PointXYZRGBNormal> (dcloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(dcloud), "dcloud");
						viewer->spin();
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
	return poses;
}

MassFusionResults MassRegistrationPPR::getTransforms(std::vector<Eigen::Matrix4d> poses){
	if(visualizationLvl > 0){printf("start MassRegistrationPPR::getTransforms(std::vector<Eigen::Matrix4d> poses)\n");}

	unsigned int nr_frames = informations.size();
	if(poses.size() != nr_frames){
		printf("ERROR: poses.size() != informations.size()\n");
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

	for(int funcupdate=0; funcupdate < 100; ++funcupdate) {
		if(getTime()-total_time_start > timeout){break;}
		if(visualizationLvl == 2){std::vector<Eigen::MatrixXd> Xv;for(unsigned int j = 0; j < nr_frames; j++){Xv.push_back(transformed_points[j]);}sprintf(buf,"image%5.5i.png",imgcount++);show(Xv,false,std::string(buf),imgcount);}

		for(int rematching=0; rematching < 20; ++rematching) {
			if(visualizationLvl == 3){std::vector<Eigen::MatrixXd> Xv;for(unsigned int j = 0; j < nr_frames; j++){Xv.push_back(transformed_points[j]);}sprintf(buf,"image%5.5i.png",imgcount++);show(Xv,false,std::string(buf),imgcount);}
			std::vector<Eigen::Matrix4d> poses1 = poses;

			double rematch_time_start = getTime();
			rematch(poses,poses0,first);
			first = false;
			poses0 = poses;
			rematch_time += getTime()-rematch_time_start;

			for(int lala = 0; lala < 1; lala++){
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
				residuals_time += getTime()-residuals_time_start;

				double computeModel_time_start = getTime();
				func->computeModel(all_residuals);
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
		if(fabs(1.0 - noise_after/noise_before) < 0.01){break;}
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

	printf("stop MassRegistrationPPR::getTransforms(std::vector<Eigen::Matrix4d> guess)\n");
	//exit(0);
	return MassFusionResults(poses,-1);
}

}
