#include "modelupdater/ModelUpdater.h"

#include <unordered_map>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "core/Util.h"


namespace gc
{
#include "graphcuts/graph.cpp"
#include "graphcuts/maxflow.cpp"
}

using namespace std;

namespace reglib
{

std::vector<int> ModelUpdater::getPartition(std::vector< std::vector< float > > & scores, int dims, int nr_todo, double timelimit){return partition_graph(scores);}

ModelUpdater::ModelUpdater(){
	occlusion_penalty = 10;
	massreg_timeout = 60;
	model = 0;
	show_init_lvl = 0;//init show
	show_refine_lvl = 0;//refine show
	show_scoring = false;//fuse scoring show
}

ModelUpdater::ModelUpdater(Model * model_){
	occlusion_penalty = 10;
	massreg_timeout = 60;
	show_init_lvl = 0;//init show
	show_refine_lvl = 0;//refine show
	show_scoring = false;//fuse scoring show
	model = model_;
}
ModelUpdater::~ModelUpdater(){}

FusionResults ModelUpdater::registerModel(Model * model2, Eigen::Matrix4d guess, double uncertanity){return FusionResults();}

void ModelUpdater::fuse(Model * model2, Eigen::Matrix4d guess, double uncertanity){
	for(unsigned int i = 0; i < model2->frames.size();i++){
		model->frames.push_back(model2->frames[i]);
		model->modelmasks.push_back(model2->modelmasks[i]);
		model->relativeposes.push_back(guess*model2->relativeposes[i]);
	}
}

UpdatedModels ModelUpdater::fuseData(FusionResults * f, Model * model1,Model * model2){return UpdatedModels();}

bool ModelUpdater::isRefinementNeeded(){
	vector<vector < OcclusionScore > > occlusionScores;
	occlusionScores.resize(model->frames.size());
	for(unsigned int i = 0; i < model->frames.size(); i++){occlusionScores[i].resize(model->frames.size());}
	for(unsigned int i = 0; i < model->frames.size(); i++){
		for(unsigned int j = i+1; j < model->frames.size(); j++){
			Eigen::Matrix4d relative_pose = model->relativeposes[i].inverse() * model->relativeposes[j];
			occlusionScores[j][i]		= computeOcclusionScore(model->frames[j], model->modelmasks[j],model->frames[i], model->modelmasks[i], relative_pose,1,false);
			occlusionScores[i][j]		= computeOcclusionScore(model->frames[i], model->modelmasks[i],model->frames[j], model->modelmasks[j], relative_pose.inverse(),1,false);
		}
	}
	std::vector<std::vector < float > > scores = getScores(occlusionScores);
	std::vector<int> partition = getPartition(scores,2,5,2);
	bool failed = false;
	for(unsigned int i = 0; i < partition.size(); i++){failed = failed | (partition[i]!=0);}
	return failed;
}

OcclusionScore ModelUpdater::computeOcclusionScore(vector<superpoint> & spvec, Matrix4d cp, RGBDFrame* cf, ModelMask* cm, int step,  bool debugg){
	OcclusionScore oc;

	unsigned char  * dst_maskdata		= (unsigned char	*)(cm->mask.data);
	unsigned char  * dst_rgbdata		= (unsigned char	*)(cf->rgb.data);
	unsigned short * dst_depthdata		= (unsigned short	*)(cf->depth.data);
	float		   * dst_normalsdata	= (float			*)(cf->normals.data);

	float m00 = cp(0,0); float m01 = cp(0,1); float m02 = cp(0,2); float m03 = cp(0,3);
	float m10 = cp(1,0); float m11 = cp(1,1); float m12 = cp(1,2); float m13 = cp(1,3);
	float m20 = cp(2,0); float m21 = cp(2,1); float m22 = cp(2,2); float m23 = cp(2,3);

	Camera * dst_camera				= cf->camera;
	const unsigned int dst_width	= dst_camera->width;
	const unsigned int dst_height	= dst_camera->height;
	const float dst_idepth			= dst_camera->idepth_scale;
	const float dst_cx				= dst_camera->cx;
	const float dst_cy				= dst_camera->cy;
	const float dst_fx				= dst_camera->fx;
	const float dst_fy				= dst_camera->fy;
	const float dst_ifx				= 1.0/dst_camera->fx;
	const float dst_ify				= 1.0/dst_camera->fy;
	const unsigned int dst_width2	= dst_camera->width  - 2;
	const unsigned int dst_height2	= dst_camera->height - 2;

	int nr_data = spvec.size();

	std::vector<float> residuals;
	std::vector<int> debugg_src_inds;
	std::vector<int> debugg_dst_inds;
	std::vector<float> weights;
	residuals.reserve(nr_data);
	if(debugg){
		debugg_src_inds.reserve(nr_data);
		debugg_dst_inds.reserve(nr_data);
	}
	weights.reserve(nr_data);

	for(unsigned int ind = 0; ind < spvec.size();ind+=step){
		superpoint & sp = spvec[ind];

		float src_x = sp.point(0);
		float src_y = sp.point(1);
		float src_z = sp.point(2);

		float src_nx = sp.normal(0);
		float src_ny = sp.normal(1);
		float src_nz = sp.normal(2);

		float point_information = sp.point_information;


		float tx	= m00*src_x + m01*src_y + m02*src_z + m03;
		float ty	= m10*src_x + m11*src_y + m12*src_z + m13;
		float tz	= m20*src_x + m21*src_y + m22*src_z + m23;

		float itz	= 1.0/tz;
		float dst_w	= dst_fx*tx*itz + dst_cx;
		float dst_h	= dst_fy*ty*itz + dst_cy;

		if((dst_w > 0) && (dst_h > 0) && (dst_w < dst_width2) && (dst_h < dst_height2)){
			unsigned int dst_ind = unsigned(dst_h+0.5) * dst_width + unsigned(dst_w+0.5);

			float dst_z = dst_idepth*float(dst_depthdata[dst_ind]);
			float dst_nx = dst_normalsdata[3*dst_ind+0];
			if(dst_z > 0 && dst_nx != 2){
				//if(dst_detdata[dst_ind] != 0){continue;}
				float dst_ny = dst_normalsdata[3*dst_ind+1];
				float dst_nz = dst_normalsdata[3*dst_ind+2];

				float dst_x = (float(dst_w) - dst_cx) * dst_z * dst_ifx;
				float dst_y = (float(dst_h) - dst_cy) * dst_z * dst_ify;

				float tnx	= m00*src_nx + m01*src_ny + m02*src_nz;
				float tny	= m10*src_nx + m11*src_ny + m12*src_nz;
				float tnz	= m20*src_nx + m21*src_ny + m22*src_nz;

				double d = mysign(dst_z-tz)*fabs(tnx*(dst_x-tx) + tny*(dst_y-ty) + tnz*(dst_z-tz));
				double dst_noise = dst_z * dst_z;
				double point_noise = 1.0/sqrt(point_information);

				double compare_mul = sqrt(dst_noise*dst_noise + point_noise*point_noise);
				d *= compare_mul;

				double dist_dst = sqrt(dst_x*dst_x+dst_y*dst_y+dst_z*dst_z);
				double angle_dst = fabs((dst_x*dst_nx+dst_y*dst_ny+dst_z*dst_nz)/dist_dst);

				residuals.push_back(d);
				weights.push_back(angle_dst*angle_dst*angle_dst);
				if(debugg){
					debugg_src_inds.push_back(ind);
					debugg_dst_inds.push_back(dst_ind);
				}
			}
		}
	}

	//	DistanceWeightFunction2PPR2 * func = new DistanceWeightFunction2PPR2();
	//	func->maxp			= 1.0;
	//	func->update_size	= true;
	//	func->zeromean      = true;
	//	func->startreg		= 0.0001;
	//	func->debugg_print	= debugg;
	//	func->bidir			= true;
	//	func->maxnoise      = pred;
	//	func->reset();

	DistanceWeightFunction2 * func = new DistanceWeightFunction2();
	func->f = THRESHOLD;
	func->p = 0.02;

	Eigen::MatrixXd X = Eigen::MatrixXd::Zero(1,residuals.size());
	for(unsigned int i = 0; i < residuals.size(); i++){X(0,i) = residuals[i];}
	func->computeModel(X);

	Eigen::VectorXd  W = func->getProbs(X);

	delete func;

	for(unsigned int i = 0; i < residuals.size(); i++){
		float r = residuals[i];
		float weight = W(i);
		float ocl = 0;
		if(r > 0){ocl += 1-weight;}
		oc.score		+= weight*weights.at(i);
		oc.occlusions	+= ocl*weights.at(i);
	}


	if(debugg){
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr dcloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
		dcloud->points.resize(dst_width*dst_height);
		for(unsigned int dst_w = 0; dst_w < dst_width; dst_w++){
			for(unsigned int dst_h = 0; dst_h < dst_height;dst_h++){
				unsigned int dst_ind = dst_h*dst_width+dst_w;
				float z = dst_idepth*float(dst_depthdata[dst_ind]);
				if(z > 0){
					float x = (float(dst_w) - dst_cx) * z * dst_ifx;
					float y = (float(dst_h) - dst_cy) * z * dst_ify;
					dcloud->points[dst_ind].x = x;
					dcloud->points[dst_ind].y = y;
					dcloud->points[dst_ind].z = z;
					dcloud->points[dst_ind].r = dst_rgbdata[3*dst_ind+2];
					dcloud->points[dst_ind].g = dst_rgbdata[3*dst_ind+1];
					dcloud->points[dst_ind].b = dst_rgbdata[3*dst_ind+0];
					if(dst_maskdata[dst_ind] == 255){
						dcloud->points[dst_ind].r = 255;
						dcloud->points[dst_ind].g = 000;
						dcloud->points[dst_ind].b = 255;
					}
				}
			}
		}

		oc.print();
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr scloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
		for(unsigned int i = 0; i < spvec.size(); i++){
			superpoint & sp = spvec[i];
			pcl::PointXYZRGBNormal p;
			p.x = sp.point(0);
			p.y = sp.point(1);
			p.z = sp.point(2);

			float tx	= m00*p.x + m01*p.y + m02*p.z + m03;
			float ty	= m10*p.x + m11*p.y + m12*p.z + m13;
			float tz	= m20*p.x + m21*p.y + m22*p.z + m23;

			p.x = tx;
			p.y = ty;
			p.z = tz;

			p.normal_x = sp.normal(0);
			p.normal_y = sp.normal(1);
			p.normal_z = sp.normal(2);
			p.r = sp.feature(0);
			p.g = sp.feature(1);
			p.b = sp.feature(2);

			scloud->points.push_back(p);
		}

		//		viewer->removeAllPointClouds();
		//		viewer->addPointCloud<pcl::PointXYZRGBNormal> (scloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(scloud), "scloud");
		//		viewer->addPointCloud<pcl::PointXYZRGBNormal> (dcloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(dcloud), "dcloud");
		//		viewer->spin();

		viewer->removeAllPointClouds();
		viewer->removeAllShapes();

		for(unsigned int i = 0; i < residuals.size(); i++){
			float r = residuals[i];
			float weight = W(i);
			float ocl = 0;
			if(r > 0){ocl += 1-weight;}
			if(debugg){
				unsigned int src_ind = debugg_src_inds[i];
				unsigned int dst_ind = debugg_dst_inds[i];
				if(ocl > 0.01 || weight > 0.01){
					scloud->points[src_ind].r = 255.0*ocl*weights.at(i);
					scloud->points[src_ind].g = 255.0*weight*weights.at(i);
					scloud->points[src_ind].b = 0;

					if(i % 300 == 0){
						char buf [1024];
						sprintf(buf,"line%i",i);
						viewer->addLine<pcl::PointXYZRGBNormal> (scloud->points[src_ind], dcloud->points[dst_ind],buf);
					}
				}else{
					scloud->points[src_ind].x = 0;
					scloud->points[src_ind].y = 0;
					scloud->points[src_ind].z = 0;
				}
			}
		}

		viewer->removeAllPointClouds();
		viewer->addPointCloud<pcl::PointXYZRGBNormal> (scloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(scloud), "scloud");
		viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "scloud");

		viewer->addPointCloud<pcl::PointXYZRGBNormal> (dcloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(dcloud), "dcloud");
		viewer->spin();
		viewer->removeAllPointClouds();
		viewer->removeAllShapes();

	}
	//printf("stop :: %s::%i\n",__FILE__,__LINE__);
	return oc;
}

OcclusionScore ModelUpdater::computeOcclusionScore(Model * mod, vector<Matrix4d> cp, vector<RGBDFrame*> cf, vector<ModelMask*> cm, Matrix4d rp, int step,  bool debugg){
	OcclusionScore ocs;
	for(unsigned int i = 0; i < cp.size(); i++){
		ocs.add(computeOcclusionScore(mod->points,cp[i].inverse()*rp,cf[i],cm[i],step,debugg));
	}
	return ocs;
}


OcclusionScore ModelUpdater::computeOcclusionScore(Model * model1, Model * model2, Matrix4d rp, int step, bool debugg){
	OcclusionScore ocs;
	//	ocs.add(computeOcclusionScore(model1, model2->rep_relativeposes,model2->rep_frames,model2->rep_modelmasks,rp.inverse(),step,debugg));
	//	ocs.add(computeOcclusionScore(model2, model1->rep_relativeposes,model1->rep_frames,model1->rep_modelmasks,rp,step,debugg));
	ocs.add(computeOcclusionScore(model1, model2->relativeposes,model2->frames,model2->modelmasks,rp.inverse(),step,debugg));
	ocs.add(computeOcclusionScore(model2, model1->relativeposes,model1->frames,model1->modelmasks,rp,step,debugg));
	return ocs;
}

vector<vector < OcclusionScore > > ModelUpdater::computeOcclusionScore(vector<Model *> models, vector<Matrix4d> rps, int step, bool debugg){
	unsigned int nr_models = models.size();
	vector<vector < OcclusionScore > > occlusionScores;
	occlusionScores.resize(nr_models);
	for(unsigned int i = 0; i < nr_models; i++){occlusionScores[i].resize(nr_models);}

	for(unsigned int i = 0; i < nr_models; i++){
		for(unsigned int j = i+1; j < nr_models; j++){
			//printf("%5.5f %5.5f\n",double(i+1)/double(nr_models),double(j+1)/double(nr_models));
			Eigen::Matrix4d relative_pose = rps[i].inverse() * rps[j];
			occlusionScores[i][j] = computeOcclusionScore(models[i],models[j], relative_pose, step, debugg);
			occlusionScores[j][i] = occlusionScores[i][j];
		}
	}

	return occlusionScores;
}

double ModelUpdater::computeOcclusionScoreCosts(vector<Model *> models){
	double total = 0;

	unsigned int nr_models = models.size();
	for(unsigned int i = 0; i < nr_models; i++){
		for(unsigned int j = 0; j < nr_models; j++){
			if(i == j){continue;}
			total += models[j]->rep_frames.size() * models[i]->points.size();
		}
	}

	return total;
}

void ModelUpdater::addModelsToVector(vector<Model *> & models, vector<Matrix4d> & rps, Model * model, Matrix4d rp){
	if(model->frames.size() > 0){
		models.push_back(model);
		rps.push_back(rp);
	}

	for(unsigned int i = 0; i < model->submodels.size(); i++){
		addModelsToVector(models,rps,model->submodels[i], rp*model->submodels_relativeposes[i]);
	}
}

pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr ModelUpdater::getPCLnormalcloud(vector<superpoint> & points){
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	for(unsigned int i = 0; i < points.size(); i++){
		superpoint & sp = points[i];
		pcl::PointXYZRGBNormal p;
		p.x = sp.point(0);
		p.y = sp.point(1);
		p.z = sp.point(2);

		p.normal_x = sp.normal(0);
		p.normal_y = sp.normal(1);
		p.normal_z = sp.normal(2);
		p.r = sp.feature(0);
		p.g = sp.feature(1);
		p.b = sp.feature(2);

		cloud_ptr->points.push_back(p);
	}
	return cloud_ptr;
}

bool ModelUpdater::refineIfNeeded(){

	if(isRefinementNeeded()){
		MassRegistrationPPR * massreg = new MassRegistrationPPR(0.1);
		massreg->timeout = massreg_timeout;
		massreg->viewer = viewer;
		massreg->visualizationLvl = 1;
		massreg->maskstep = 4;
		massreg->nomaskstep = std::max(1,int(0.5+1.0*double(model->frames.size())));

		massreg->nomask = true;
		massreg->stopval = 0.001;

		massreg->setData(model->frames,model->modelmasks);
		MassFusionResults mfr = massreg->getTransforms(model->relativeposes);
		model->relativeposes = mfr.poses;

		isRefinementNeeded();
	}


	MassRegistrationPPR * massreg = new MassRegistrationPPR(0.1);
	massreg->timeout = massreg_timeout;
	massreg->viewer = viewer;
	massreg->visualizationLvl = 1;
	massreg->maskstep = 4;
	massreg->nomaskstep = std::max(1,int(0.5+1.0*double(model->frames.size())));

	massreg->nomask = true;
	massreg->stopval = 0.001;

	massreg->setData(model->frames,model->modelmasks);
	MassFusionResults mfr = massreg->getTransforms(model->relativeposes);
	model->relativeposes = mfr.poses;

	vector<superpoint> spvec = getSuperPoints(model->relativeposes,model->frames,model->modelmasks,1,false);
	vector<Matrix4d> cp;
	vector<RGBDFrame*> cf;
	vector<ModelMask*> cm;
	for(unsigned int i = 0; i < model->relativeposes.size(); i++){
		cp.push_back(model->relativeposes[i]);
		cf.push_back(model->frames[i]);
		cm.push_back(model->modelmasks[i]);
	}
	getGoodCompareFrames(cp,cf,cm);

	vector<superpoint> spvec2 = getSuperPoints(cp,cf,cm,1,true);
	return true;
}

void ModelUpdater::makeInitialSetup(){

	if(model->relativeposes.size() <= 1){
		model->points = getSuperPoints(model->relativeposes,model->frames,model->modelmasks,1,false);

		vector<Matrix4d> cp;
		vector<RGBDFrame*> cf;
		vector<ModelMask*> cm;
		for(unsigned int i = 0; i < model->relativeposes.size(); i++){
			cp.push_back(model->relativeposes[i]);
			cf.push_back(model->frames[i]);
			cm.push_back(model->modelmasks[i]);
		}
		getGoodCompareFrames(cp,cf,cm);

		model->rep_relativeposes = cp;
		model->rep_frames = cf;
		model->rep_modelmasks = cm;
		return ;
	}

	MassRegistrationPPR2 * massreg = new MassRegistrationPPR2(0.0);
	massreg->timeout = 4*massreg_timeout;
	massreg->viewer = viewer;
	massreg->visualizationLvl = show_init_lvl;

	massreg->maskstep = 1;//std::max(1,int(0.25*double(model->frames.size())));
	massreg->nomaskstep = std::max(5,int(0.5+0.3*double(model->frames.size())));//std::max(1,int(0.5+1.0*double(model->frames.size())));
	massreg->nomask = true;
	massreg->stopval = 0.0005;

	massreg->setData(model->frames,model->modelmasks);
	MassFusionResults mfr = massreg->getTransforms(model->relativeposes);

	model->relativeposes = mfr.poses;



	model->points = getSuperPoints(model->relativeposes,model->frames,model->modelmasks,1,false);
	//printf("getSuperPoints done\n");
	vector<Matrix4d> cp;
	vector<RGBDFrame*> cf;
	vector<ModelMask*> cm;
	for(unsigned int i = 0; i < model->relativeposes.size(); i++){
		cp.push_back(model->relativeposes[i]);
		cf.push_back(model->frames[i]);
		cm.push_back(model->modelmasks[i]);
	}

	getGoodCompareFrames(cp,cf,cm);

	model->rep_relativeposes = cp;
	model->rep_frames = cf;
	model->rep_modelmasks = cm;

	//printf("model->rep_relativeposes: %i\n",model->rep_relativeposes.size());
}

void ModelUpdater::addSuperPoints(vector<superpoint> & spvec, Matrix4d p, RGBDFrame* frame, ModelMask* modelmask, int type, bool debugg){
	bool * maskvec = modelmask->maskvec;
	unsigned char  * rgbdata		= (unsigned char	*)(frame->rgb.data);
	unsigned short * depthdata		= (unsigned short	*)(frame->depth.data);
	float		   * normalsdata	= (float			*)(frame->normals.data);

	unsigned int frameid = frame->id;

	Matrix4d ip = p.inverse();

	float m00 = p(0,0); float m01 = p(0,1); float m02 = p(0,2); float m03 = p(0,3);
	float m10 = p(1,0); float m11 = p(1,1); float m12 = p(1,2); float m13 = p(1,3);
	float m20 = p(2,0); float m21 = p(2,1); float m22 = p(2,2); float m23 = p(2,3);

	float im00 = ip(0,0); float im01 = ip(0,1); float im02 = ip(0,2); float im03 = ip(0,3);
	float im10 = ip(1,0); float im11 = ip(1,1); float im12 = ip(1,2); float im13 = ip(1,3);
	float im20 = ip(2,0); float im21 = ip(2,1); float im22 = ip(2,2); float im23 = ip(2,3);

	Camera * camera				= frame->camera;
	const unsigned int width	= camera->width;
	const unsigned int height	= camera->height;

	const unsigned int dst_width2	= camera->width  - 2;
	const unsigned int dst_height2	= camera->height - 2;

	const float idepth			= camera->idepth_scale;
	const float cx				= camera->cx;
	const float cy				= camera->cy;
	const float fx				= camera->fx;
	const float fy				= camera->fy;
	const float ifx				= 1.0/camera->fx;
	const float ify				= 1.0/camera->fy;

	//bool * isfused = new bool[width*height];
	std::vector<bool> isfused;
	//printf("wh: %i %i -> %i\n",width,height,width*height);
	isfused.resize(width*height);
	for(unsigned int i = 0; i < width*height; i++){isfused[i] = false;}


	for(unsigned int ind = 0; ind < spvec.size();ind++){
		superpoint & sp = spvec[ind];

		float src_x = sp.point(0);
		float src_y = sp.point(1);
		float src_z = sp.point(2);

		float src_nx = sp.normal(0);
		float src_ny = sp.normal(1);
		float src_nz = sp.normal(2);

		float src_r = sp.feature(0);
		float src_g = sp.feature(1);
		float src_b = sp.feature(2);

		double point_information = sp.point_information;
		double feature_information = sp.feature_information;

		float tx	= im00*src_x + im01*src_y + im02*src_z + im03;
		float ty	= im10*src_x + im11*src_y + im12*src_z + im13;
		float tz	= im20*src_x + im21*src_y + im22*src_z + im23;

		float itz	= 1.0/tz;
		float dst_w	= fx*tx*itz + cx;
		float dst_h	= fy*ty*itz + cy;

		if((dst_w > 0) && (dst_h > 0) && (dst_w < dst_width2) && (dst_h < dst_height2)){
			unsigned int dst_ind = unsigned(dst_h+0.5) * width + unsigned(dst_w+0.5);

			float dst_z = idepth*float(depthdata[dst_ind]);
			float dst_nx = normalsdata[3*dst_ind+0];
			if(!isfused[dst_ind] && dst_z > 0 && dst_nx != 2){
				float dst_ny = normalsdata[3*dst_ind+1];
				float dst_nz = normalsdata[3*dst_ind+2];

				float dst_x = (float(dst_w) - cx) * dst_z * ifx;
				float dst_y = (float(dst_h) - cy) * dst_z * ify;

				double dst_noise = dst_z * dst_z;

				double point_noise = 1.0/sqrt(point_information);

				float tnx	= im00*src_nx + im01*src_ny + im02*src_nz;
				float tny	= im10*src_nx + im11*src_ny + im12*src_nz;
				float tnz	= im20*src_nx + im21*src_ny + im22*src_nz;

				double d = fabs(tnx*(dst_x-tx) + tny*(dst_y-ty) + tnz*(dst_z-tz));

				double compare_mul = sqrt(dst_noise*dst_noise + point_noise*point_noise);
				d *= compare_mul;

				double surface_angle = tnx*dst_nx+tny*dst_ny+tnz*dst_nz;

				if(d < 0.01 && surface_angle > 0.5){//If close, according noises, and angle of the surfaces similar: FUSE
					float px	= m00*dst_x + m01*dst_y + m02*dst_z + m03;
					float py	= m10*dst_x + m11*dst_y + m12*dst_z + m13;
					float pz	= m20*dst_x + m21*dst_y + m22*dst_z + m23;

					float pnx	= m00*dst_nx + m01*dst_ny + m02*dst_nz;
					float pny	= m10*dst_nx + m11*dst_ny + m12*dst_nz;
					float pnz	= m20*dst_nx + m21*dst_ny + m22*dst_nz;

					float pb = rgbdata[3*dst_ind+0];
					float pg = rgbdata[3*dst_ind+1];
					float pr = rgbdata[3*dst_ind+2];

					Vector3f	pxyz	(px	,py	,pz );
					Vector3f	pnxyz	(pnx,pny,pnz);
					Vector3f	prgb	(pr	,pg	,pb );
					float		weight	= 1.0/(dst_noise*dst_noise);
					superpoint sp2 = superpoint(pxyz,pnxyz,prgb, weight, weight, frameid);
					sp.merge(sp2);
					isfused[dst_ind] = true;
				}
			}
		}
	}

	int nr_fused = 0;
	int nr_mask = 0;
	for(unsigned int w = 0; w < width; w++){
		for(unsigned int h = 0; h < height;h++){
			int ind = h*width+w;
			nr_fused += isfused[ind];
			nr_mask += maskvec[ind] > 0;
		}
	}

	for(unsigned int w = 0; w < width; w++){
		for(unsigned int h = 0; h < height;h++){
			int ind = h*width+w;
			if(!isfused[ind] && maskvec[ind]){
				float z = idepth*float(depthdata[ind]);
				float nx = normalsdata[3*ind+0];

				if(z > 0 && nx != 2){
					float ny = normalsdata[3*ind+1];
					float nz = normalsdata[3*ind+2];

					float x = (w - cx) * z * ifx;
					float y = (h - cy) * z * ify;

					double noise = z * z;

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
					float		weight	= 1.0/(noise*noise);
					spvec.push_back(superpoint(pxyz,pnxyz,prgb, weight, weight, frameid));
				}
			}
		}
	}

	//delete[] isfused;

	if(debugg){
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud = getPCLnormalcloud(spvec);
		viewer->removeAllPointClouds();
		viewer->addPointCloud<pcl::PointXYZRGBNormal> (cloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(cloud), "cloud");
		viewer->spin();
	}
}

vector<superpoint> ModelUpdater::getSuperPoints(vector<Matrix4d> cp, vector<RGBDFrame*> cf, vector<ModelMask*> cm, int type, bool debugg){
	vector<superpoint> spvec;
	for(unsigned int i = 0; i < cp.size(); i++){addSuperPoints(spvec, cp[i], cf[i], cm[i], type,debugg);}
	return spvec;
}

double ModelUpdater::getCompareUtility(Matrix4d p, RGBDFrame* frame, ModelMask* mask, vector<Matrix4d> & cp, vector<RGBDFrame*> & cf, vector<ModelMask*> & cm){
	unsigned char  * src_maskdata		= (unsigned char	*)(mask->mask.data);
	unsigned char  * src_rgbdata		= (unsigned char	*)(frame->rgb.data);
	unsigned short * src_depthdata		= (unsigned short	*)(frame->depth.data);
	float		   * src_normalsdata	= (float			*)(frame->normals.data);

	Camera * src_camera				= frame->camera;
	const unsigned int src_width	= src_camera->width;
	const unsigned int src_height	= src_camera->height;
	const float src_idepth			= src_camera->idepth_scale;
	const float src_cx				= src_camera->cx;
	const float src_cy				= src_camera->cy;
	const float src_ifx				= 1.0/src_camera->fx;
	const float src_ify				= 1.0/src_camera->fy;

	std::vector<float> test_x;
	std::vector<float> test_y;
	std::vector<float> test_z;
	std::vector<float> test_nx;
	std::vector<float> test_ny;
	std::vector<float> test_nz;

	std::vector<int> & testw = mask->testw;
	std::vector<int> & testh = mask->testh;

	for(unsigned int ind = 0; ind < testw.size();ind++){
		unsigned int src_w = testw[ind];
		unsigned int src_h = testh[ind];

		int src_ind = src_h*src_width+src_w;
		if(src_maskdata[src_ind] == 255){
			float src_z = src_idepth*float(src_depthdata[src_ind]);
			float src_nx = src_normalsdata[3*src_ind+0];
			if(src_z > 0 && src_nx != 2){
				float src_ny = src_normalsdata[3*src_ind+1];
				float src_nz = src_normalsdata[3*src_ind+2];

				float src_x = (float(src_w) - src_cx) * src_z * src_ifx;
				float src_y = (float(src_h) - src_cy) * src_z * src_ify;

				test_x.push_back(src_x);
				test_y.push_back(src_y);
				test_z.push_back(src_z);
				test_nx.push_back(src_nx);
				test_ny.push_back(src_ny);
				test_nz.push_back(src_nz);
			}
		}
	}

	bool debugg = false;
	unsigned int nrdata_start = test_x.size();

	for(unsigned int c = 0; c < cp.size();c++){
		RGBDFrame* dst = cf[c];
		if(dst == frame){continue;}
		ModelMask* dst_modelmask = cm[c];
		unsigned char  * dst_maskdata		= (unsigned char	*)(dst_modelmask->mask.data);
		unsigned char  * dst_rgbdata		= (unsigned char	*)(dst->rgb.data);
		unsigned short * dst_depthdata		= (unsigned short	*)(dst->depth.data);
		float		   * dst_normalsdata	= (float			*)(dst->normals.data);

		Matrix4d rp = cp[c].inverse()*p;

		float m00 = rp(0,0); float m01 = rp(0,1); float m02 = rp(0,2); float m03 = rp(0,3);
		float m10 = rp(1,0); float m11 = rp(1,1); float m12 = rp(1,2); float m13 = rp(1,3);
		float m20 = rp(2,0); float m21 = rp(2,1); float m22 = rp(2,2); float m23 = rp(2,3);

		Camera * dst_camera				= dst->camera;
		const unsigned int dst_width	= dst_camera->width;
		const unsigned int dst_height	= dst_camera->height;
		const float dst_idepth			= dst_camera->idepth_scale;
		const float dst_cx				= dst_camera->cx;
		const float dst_cy				= dst_camera->cy;
		const float dst_fx				= dst_camera->fx;
		const float dst_fy				= dst_camera->fy;
		const float dst_ifx				= 1.0/dst_camera->fx;
		const float dst_ify				= 1.0/dst_camera->fy;
		const unsigned int dst_width2	= dst_camera->width  - 2;
		const unsigned int dst_height2	= dst_camera->height - 2;
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
		if(debugg){
			viewer->removeAllPointClouds();

			pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2 = dst->getPCLcloud();
			viewer->addPointCloud<pcl::PointXYZRGBNormal> (cloud2, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(cloud2), "cloud2");
		}

		for(unsigned int ind = 0; ind < test_x.size();ind++){
			float src_nx = test_nx[ind];
			float src_ny = test_ny[ind];
			float src_nz = test_nz[ind];

			float src_x = test_x[ind];
			float src_y = test_y[ind];
			float src_z = test_z[ind];

			float tx	= m00*src_x + m01*src_y + m02*src_z + m03;
			float ty	= m10*src_x + m11*src_y + m12*src_z + m13;
			float tz	= m20*src_x + m21*src_y + m22*src_z + m23;
			float itz	= 1.0/tz;
			float dst_w	= dst_fx*tx*itz + dst_cx;
			float dst_h	= dst_fy*ty*itz + dst_cy;

			pcl::PointXYZRGBNormal point;
			if(debugg){
				point.x = tx;
				point.y = ty;
				point.z = tz;
				point.r = 255;
				point.g = 0;
				point.b = 0;
			}


			if((dst_w > 0) && (dst_h > 0) && (dst_w < dst_width2) && (dst_h < dst_height2)){
				unsigned int dst_ind = unsigned(dst_h+0.5) * dst_width + unsigned(dst_w+0.5);

				float dst_z = dst_idepth*float(dst_depthdata[dst_ind]);
				float dst_nx = dst_normalsdata[3*dst_ind+0];
				if(dst_z > 0 && dst_nx != 2){
					float dst_ny = dst_normalsdata[3*dst_ind+1];
					float dst_nz = dst_normalsdata[3*dst_ind+2];

					float dst_x = (float(dst_w) - dst_cx) * dst_z * dst_ifx;
					float dst_y = (float(dst_h) - dst_cy) * dst_z * dst_ify;

					double dst_noise = dst_z * dst_z;
					double point_noise = src_z * src_z;

					float tnx	= m00*src_nx + m01*src_ny + m02*src_nz;
					float tny	= m10*src_nx + m11*src_ny + m12*src_nz;
					float tnz	= m20*src_nx + m21*src_ny + m22*src_nz;

					double d = fabs(tnx*(dst_x-tx) + tny*(dst_y-ty) + tnz*(dst_z-tz));

					double compare_mul = sqrt(dst_noise*dst_noise + point_noise*point_noise);
					d *= compare_mul;

					double surface_angle = tnx*dst_nx+tny*dst_ny+tnz*dst_nz;

					if(d < 0.01 && surface_angle > 0.5){//If close, according noises, and angle of the surfaces similar: FUSE
						//if(d < 0.01){//If close, according noises, and angle of the surfaces similar: FUSE
						test_nx[ind] = test_nx.back(); test_nx.pop_back();
						test_ny[ind] = test_ny.back(); test_ny.pop_back();
						test_nz[ind] = test_nz.back(); test_nz.pop_back();
						test_x[ind] = test_x.back(); test_x.pop_back();
						test_y[ind] = test_y.back(); test_y.pop_back();
						test_z[ind] = test_z.back(); test_z.pop_back();
						ind--;
						if(debugg){
							point.r = 0;
							point.g = 255;
							point.b = 0;
						}
					}
				}
			}
			if(debugg){
				cloud->points.push_back(point);
			}
		}

		if(debugg){
			viewer->addPointCloud<pcl::PointXYZRGBNormal> (cloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(cloud), "cloud");
			viewer->spin();
			viewer->removeAllPointClouds();
			viewer->addPointCloud<pcl::PointXYZRGBNormal> (cloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(cloud), "cloud");
			viewer->spin();
		}
	}
	return double(test_x.size())/double(nrdata_start);
}

void ModelUpdater::getGoodCompareFrames(vector<Matrix4d> & cp, vector<RGBDFrame*> & cf, vector<ModelMask*> & cm){
	while(true){
		double worst = 1;
		int worst_id = -1;
		for(unsigned int i = 0; i < cp.size(); i++){
			double score = getCompareUtility(cp[i], cf[i], cm[i], cp, cf, cm);
			if(score < worst){
				worst = score;
				worst_id = i;
			}
		}

		if(worst_id < 0){break;}
		if(worst < 0.1){//worst frame can to 90% be represented by other frames
			cp[worst_id] = cp.back();
			cp.pop_back();

			cf[worst_id] = cf.back();
			cf.pop_back();

			cm[worst_id] = cm.back();
			cm.pop_back();
		}else{break;}
	}
}

void ModelUpdater::refine(double reg,bool useFullMask, int visualization){


	vector<vector < OcclusionScore > > ocs = computeOcclusionScore(model->submodels,model->submodels_relativeposes,1,false);

	std::vector<std::vector < float > > scores = getScores(ocs);


	double sumscore_bef = 0;
	for(unsigned int i = 0; i < scores.size(); i++){
		for(unsigned int j = 0; j < scores.size(); j++){
			sumscore_bef += scores[i][j];
		}
	}

	MassFusionResults mfr;
	MassRegistrationPPR * massreg = new MassRegistrationPPR(reg);
	massreg->timeout = massreg_timeout;
	massreg->viewer = viewer;
	massreg->visualizationLvl = 0;
	massreg->maskstep = 4;//std::max(1,int(0.5+0.02*double(model->frames.size())));
	massreg->nomaskstep = std::max(1,int(0.5+1.0*double(model->frames.size())));
	printf("maskstep: %i nomaskstep: %i\n",massreg->maskstep,massreg->nomaskstep);
	massreg->nomask = true;
	massreg->stopval = 0.001;
	double step = model->submodels_relativeposes.size()*model->submodels_relativeposes.size();
	step *= 0.25;
	step += 0.5;
	massreg->maskstep = std::max(1.5,step);
	massreg->addModelData(model);
	mfr = massreg->getTransforms(model->submodels_relativeposes);
	vector<vector < OcclusionScore > > ocs2 = computeOcclusionScore(model->submodels,mfr.poses,1,false);
	std::vector<std::vector < float > > scores2 = getScores(ocs2);

	double sumscore_aft = 0;
	for(unsigned int i = 0; i < scores2.size(); i++){
		for(unsigned int j = 0; j < scores2.size(); j++){
			sumscore_aft += scores2[i][j];
		}
	}

	printf("bef %f after %f\n",sumscore_bef,sumscore_aft);
	if(sumscore_aft >= sumscore_bef){
		model->submodels_relativeposes = mfr.poses;
	}
}

void ModelUpdater::show(bool stop){
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = model->getPCLcloud(1,false);
	viewer->removeAllPointClouds();
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(cloud), "cloud");
	if(stop){	viewer->spin();}
	else{		viewer->spinOnce();}
	viewer->removeAllPointClouds();
}

//Backproject and prune occlusions
void ModelUpdater::pruneOcclusions(){}

void ModelUpdater::computeResiduals(std::vector<float> & residuals, std::vector<float> & weights,
									RGBDFrame * src, cv::Mat src_mask, ModelMask * src_modelmask,
									RGBDFrame * dst, cv::Mat dst_mask, ModelMask * dst_modelmask,
									Eigen::Matrix4d p, bool debugg){

	unsigned char  * src_maskdata		= (unsigned char	*)(src_modelmask->mask.data);
	unsigned char  * src_rgbdata		= (unsigned char	*)(src->rgb.data);
	unsigned short * src_depthdata		= (unsigned short	*)(src->depth.data);
	float		   * src_normalsdata	= (float			*)(src->normals.data);

	unsigned char  * dst_maskdata		= (unsigned char	*)(dst_modelmask->mask.data);
	unsigned char  * dst_rgbdata		= (unsigned char	*)(dst->rgb.data);
	unsigned short * dst_depthdata		= (unsigned short	*)(dst->depth.data);
	float		   * dst_normalsdata	= (float			*)(dst->normals.data);

	float m00 = p(0,0); float m01 = p(0,1); float m02 = p(0,2); float m03 = p(0,3);
	float m10 = p(1,0); float m11 = p(1,1); float m12 = p(1,2); float m13 = p(1,3);
	float m20 = p(2,0); float m21 = p(2,1); float m22 = p(2,2); float m23 = p(2,3);

	Camera * src_camera				= src->camera;
	const unsigned int src_width	= src_camera->width;
	const unsigned int src_height	= src_camera->height;
	const float src_idepth			= src_camera->idepth_scale;
	const float src_cx				= src_camera->cx;
	const float src_cy				= src_camera->cy;
	const float src_ifx				= 1.0/src_camera->fx;
	const float src_ify				= 1.0/src_camera->fy;

	Camera * dst_camera				= dst->camera;
	const unsigned int dst_width	= dst_camera->width;
	const unsigned int dst_height	= dst_camera->height;
	const float dst_idepth			= dst_camera->idepth_scale;
	const float dst_cx				= dst_camera->cx;
	const float dst_cy				= dst_camera->cy;
	const float dst_fx				= dst_camera->fx;
	const float dst_fy				= dst_camera->fy;

	const unsigned int dst_width2	= dst_camera->width  - 2;
	const unsigned int dst_height2	= dst_camera->height - 2;

	std::vector<int> & testw = src_modelmask->testw;
	std::vector<int> & testh = src_modelmask->testh;

	unsigned int test_nrdata = testw.size();
	for(unsigned int ind = 0; ind < test_nrdata;ind++){
		unsigned int src_w = testw[ind];
		unsigned int src_h = testh[ind];

		int src_ind = src_h*src_width+src_w;
		if(src_maskdata[src_ind] == 255){// && p.z > 0 && !isnan(p.normal_x)){
			float z = src_idepth*float(src_depthdata[src_ind]);
			float nx = src_normalsdata[3*src_ind+0];

			if(z > 0 && nx != 2){
				float ny = src_normalsdata[3*src_ind+1];
				float nz = src_normalsdata[3*src_ind+2];

				float x = (float(src_w) - src_cx) * z * src_ifx;
				float y = (float(src_h) - src_cy) * z * src_ify;

				float tx	= m00*x + m01*y + m02*z + m03;
				float ty	= m10*x + m11*y + m12*z + m13;
				float tz	= m20*x + m21*y + m22*z + m23;
				//float tnx	= m00*nx + m01*ny + m02*nz;
				//float tny	= m10*nx + m11*ny + m12*nz;
				float tnz	= m20*nx + m21*ny + m22*nz;

				float itz	= 1.0/tz;
				float dst_w	= dst_fx*tx*itz + dst_cx;
				float dst_h	= dst_fy*ty*itz + dst_cy;

				if((dst_w > 0) && (dst_h > 0) && (dst_w < dst_width2) && (dst_h < dst_height2)){
					unsigned int dst_ind = unsigned(dst_h+0.5) * dst_width + unsigned(dst_w+0.5);

					float dst_z = dst_idepth*float(dst_depthdata[dst_ind]);
					if(dst_z > 0){
						float diff_z = (dst_z-tz)/(z*z+dst_z*dst_z);//if tz < dst_z then tz infront and diff_z > 0
						residuals.push_back(diff_z);
						weights.push_back(1.0);
					}
				}
			}
		}
	}
}

void ModelUpdater::recomputeScores(){
	//	printf("recomputeScores\n");
	std::vector<std::vector < OcclusionScore > > ocs = getOcclusionScores(model->relativeposes,model->frames,model->modelmasks,false);
	model->scores = getScores(ocs);

	model->total_scores = 0;
	for(unsigned int i = 0; i < model->scores.size(); i++){
		for(unsigned int j = 0; j < model->scores.size(); j++){
			model->total_scores += model->scores[i][j];
		}
	}
}

void ModelUpdater::getAreaWeights(Matrix4d p, RGBDFrame* frame1, double * weights1, double * overlaps1, double * total1, RGBDFrame* frame2, double * weights2, double * overlaps2, double * total2){
	unsigned char  * src_rgbdata		= (unsigned char	*)(frame1->rgb.data);
	unsigned short * src_depthdata		= (unsigned short	*)(frame1->depth.data);
	float		   * src_normalsdata	= (float			*)(frame1->normals.data);

	Camera * src_camera				= frame1->camera;
	const unsigned int src_width	= src_camera->width;
	const unsigned int src_height	= src_camera->height;
	const float src_idepth			= src_camera->idepth_scale;
	const float src_cx				= src_camera->cx;
	const float src_cy				= src_camera->cy;
	const float src_ifx				= 1.0/src_camera->fx;
	const float src_ify				= 1.0/src_camera->fy;

	unsigned char  * dst_rgbdata		= (unsigned char	*)(frame2->rgb.data);
	unsigned short * dst_depthdata		= (unsigned short	*)(frame2->depth.data);
	float		   * dst_normalsdata	= (float			*)(frame2->normals.data);

	Camera * dst_camera				= frame2->camera;
	const unsigned int dst_width	= dst_camera->width;
	const unsigned int dst_height	= dst_camera->height;
	const float dst_idepth			= dst_camera->idepth_scale;
	const float dst_cx				= dst_camera->cx;
	const float dst_cy				= dst_camera->cy;
	const float dst_ifx				= 1.0/dst_camera->fx;
	const float dst_ify				= 1.0/dst_camera->fy;
	const float dst_fx				= dst_camera->fx;
	const float dst_fy				= dst_camera->fy;

	const unsigned int dst_width2	= dst_camera->width  - 2;
	const unsigned int dst_height2	= dst_camera->height - 2;

	float m00 = p(0,0); float m01 = p(0,1); float m02 = p(0,2); float m03 = p(0,3);
	float m10 = p(1,0); float m11 = p(1,1); float m12 = p(1,2); float m13 = p(1,3);
	float m20 = p(2,0); float m21 = p(2,1); float m22 = p(2,2); float m23 = p(2,3);

	bool debugg = false;
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr src_cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr dst_cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	if(debugg){

		src_cloud->points.resize(src_width*src_height);
		for(unsigned int src_w = 0; src_w < src_width;src_w++){
			for(unsigned int src_h = 0; src_h < src_height;src_h++){
				int src_ind = src_h*src_width+src_w;
				float src_z = src_idepth*float(src_depthdata[src_ind]);
				float src_nx = src_normalsdata[3*src_ind+0];
				if(src_z > 0 && src_nx != 2){
					float src_ny = src_normalsdata[3*src_ind+1];
					float src_nz = src_normalsdata[3*src_ind+2];

					float src_x = (float(src_w) - src_cx) * src_z * src_ifx;
					float src_y = (float(src_h) - src_cy) * src_z * src_ify;


					float tx	= m00*src_x + m01*src_y + m02*src_z + m03;
					float ty	= m10*src_x + m11*src_y + m12*src_z + m13;
					float tz	= m20*src_x + m21*src_y + m22*src_z + m23;

					pcl::PointXYZRGBNormal point;
					point.x = tx;
					point.y = ty;
					point.z = tz;
					point.r = 255;
					point.g = 0;
					point.b = 0;
					src_cloud->points[src_ind] = point;
				}
			}
		}

		dst_cloud->points.resize(dst_width*dst_height);
		for(unsigned int dst_w = 0; dst_w < dst_width;dst_w++){
			for(unsigned int dst_h = 0; dst_h < dst_height;dst_h++){
				int dst_ind = dst_h*dst_width+dst_w;
				float dst_z = dst_idepth*float(dst_depthdata[dst_ind]);
				float dst_nx = dst_normalsdata[3*dst_ind+0];
				if(dst_z > 0 && dst_nx != 2){
					float dst_ny = dst_normalsdata[3*dst_ind+1];
					float dst_nz = dst_normalsdata[3*dst_ind+2];

					float dst_x = (float(dst_w) - dst_cx) * dst_z * dst_ifx;
					float dst_y = (float(dst_h) - dst_cy) * dst_z * dst_ify;

					pcl::PointXYZRGBNormal point;
					point.x = dst_x;
					point.y = dst_y;
					point.z = dst_z;
					point.r = 0;
					point.g = 0;
					point.b = 255;
					dst_cloud->points[dst_ind] = point;
				}
			}
		}
	}

	for(unsigned int src_w = 0; src_w < src_width;src_w++){
		for(unsigned int src_h = 0; src_h < src_height;src_h++){
			int src_ind = src_h*src_width+src_w;
			float src_z = src_idepth*float(src_depthdata[src_ind]);
			float src_nx = src_normalsdata[3*src_ind+0];
			if(src_z > 0 && src_nx != 2){
				float src_ny = src_normalsdata[3*src_ind+1];
				float src_nz = src_normalsdata[3*src_ind+2];

				float src_x = (float(src_w) - src_cx) * src_z * src_ifx;
				float src_y = (float(src_h) - src_cy) * src_z * src_ify;


				float tx	= m00*src_x + m01*src_y + m02*src_z + m03;
				float ty	= m10*src_x + m11*src_y + m12*src_z + m13;
				float tz	= m20*src_x + m21*src_y + m22*src_z + m23;
				float itz	= 1.0/tz;
				float dst_w	= dst_fx*tx*itz + dst_cx;
				float dst_h	= dst_fy*ty*itz + dst_cy;

				if((dst_w > 0) && (dst_h > 0) && (dst_w < dst_width2) && (dst_h < dst_height2)){
					unsigned int dst_ind = unsigned(dst_h+0.5) * dst_width + unsigned(dst_w+0.5);

					float dst_z = dst_idepth*float(dst_depthdata[dst_ind]);
					float dst_nx = dst_normalsdata[3*dst_ind+0];
					if(dst_z > 0 && dst_nx != 2){
						float dst_ny = dst_normalsdata[3*dst_ind+1];
						float dst_nz = dst_normalsdata[3*dst_ind+2];

						float dst_x = (float(dst_w) - dst_cx) * dst_z * dst_ifx;
						float dst_y = (float(dst_h) - dst_cy) * dst_z * dst_ify;

						double dst_noise = dst_z * dst_z;
						double point_noise = src_z * src_z;

						float tnx	= m00*src_nx + m01*src_ny + m02*src_nz;
						float tny	= m10*src_nx + m11*src_ny + m12*src_nz;
						float tnz	= m20*src_nx + m21*src_ny + m22*src_nz;

						double d = (tnx*(dst_x-tx) + tny*(dst_y-ty) + tnz*(dst_z-tz)) / (src_z*src_z + dst_z*dst_z);
						//double compare_mul = 1/sqrt(dst_noise*dst_noise + point_noise*point_noise);
						//d *= src_z;//compare_mul;
						//double surface_angle = tnx*dst_nx+tny*dst_ny+tnz*dst_nz;

						if(fabs(d) < 0.01){//If close, according noises, and angle of the surfaces similar: FUSE
							overlaps1[src_ind] += weights2[dst_ind];
							overlaps2[dst_ind] += weights1[src_ind];

							total1[src_ind] += weights2[dst_ind];
							total2[dst_ind] += weights1[src_ind];
							if(debugg){
								dst_cloud->points[dst_ind].r = 0;
								dst_cloud->points[dst_ind].g = 255;
								dst_cloud->points[dst_ind].b = 0;
								src_cloud->points[src_ind].r = 0;
								src_cloud->points[src_ind].g = 255;
								src_cloud->points[src_ind].b = 0;
							}
						}else if(tz < dst_z){
							total1[src_ind] += weights2[dst_ind];
							total2[dst_ind] += weights1[src_ind];
							if(debugg){
								src_cloud->points[src_ind].r = 255;
								src_cloud->points[src_ind].g = 0;
								src_cloud->points[src_ind].b = 255;
							}
						}
					}
				}
			}
		}
	}

	if(debugg){
		viewer->removeAllPointClouds();
		viewer->addPointCloud<pcl::PointXYZRGBNormal> (src_cloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(src_cloud), "scloud");
		viewer->addPointCloud<pcl::PointXYZRGBNormal> (dst_cloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(dst_cloud), "dcloud");
		viewer->spin();
	}
}

double interpolate( double val, double y0, double x0, double y1, double x1 ) {return (val-x0)*(y1-y0)/(x1-x0) + y0;}
double base( double val ) {
	if ( val <= -0.75 ) return 0;
	else if ( val <= -0.25 ) return interpolate( val, 0.0, -0.75, 1.0, -0.25 );
	else if ( val <= 0.25 ) return 1.0;
	else if ( val <= 0.75 ) return interpolate( val, 1.0, 0.25, 0.0, 0.75 );
	else return 0.0;
}
double red(		double gray )	{ return base( gray - 0.5 ); }
double green(	double gray )	{ return base( gray );}
double blue(	double gray )	{ return base( gray + 0.5 );}

cv::Mat getImageFromArray(unsigned int width, unsigned int height, double * arr){
	double maxval = 0;
	for(unsigned int i = 0; i < width*height; i++){maxval = std::max(maxval,arr[i]);}
	maxval = 1;

	cv::Mat m;
	m.create(height,width,CV_8UC3);
	unsigned char * data = m.data;
	for(unsigned int i = 0; i < width*height; i++){
		double d = arr[i]/maxval;
		if(d < 0){
			data[3*i+0] = 255.0;//blue(d);
			data[3*i+1] = 0;//green(d);
			data[3*i+2] = 255.0;//red(d);
		}else{
			data[3*i+0] = 255.0*d;//blue(d);
			data[3*i+1] = 255.0*d;//green(d);
			data[3*i+2] = 255.0*d;//red(d);
		}
	}

	//	cv::namedWindow( "getImageFromArray", cv::WINDOW_AUTOSIZE );
	//	cv::imshow( "getImageFromArray",m);
	//	cv::waitKey(0);

	return m;
}

void ModelUpdater::computeOcclusionAreas(vector<Matrix4d> cp, vector<RGBDFrame*> cf, vector<ModelMask*> cm){

	//double ** weights_old = new double*[];
	std::vector<double **> weights_old;
	double ** wo = new double*[cf.size()];
	for(unsigned int i = 0; i < cf.size(); i++){
		unsigned int nr_pixels = cf[i]->camera->width * cf[i]->camera->height;
		wo[i] = new double[nr_pixels];
		for(unsigned int j = 0; j < nr_pixels; j++){
			wo[i][j] = 1;
		}
	}

	weights_old.push_back(wo);

	double ** overlaps = new double*[cf.size()];
	double ** total = new double*[cf.size()];
	for(unsigned int i = 0; i < cf.size(); i++){
		unsigned int nr_pixels = cf[i]->camera->width * cf[i]->camera->height;
		overlaps[i] = new double[nr_pixels];
		total[i] = new double[nr_pixels];
	}







	for(unsigned int iter = 0; iter < 10; iter++){
		for(unsigned int i = 0; i < cf.size(); i++){
			unsigned int nr_pixels = cf[i]->camera->width * cf[i]->camera->height;
			for(unsigned int j = 0; j < nr_pixels; j++){
				overlaps[i][j] = 0;
				total[i][j] = 0;
			}
		}
		printf("iter %i\n",iter);
		for(unsigned int i = 0; i < cf.size(); i++){
			for(unsigned int j = 0; j < cf.size(); j++){
				if(i == j){continue;}
				//weightedocclusioncounts
				Eigen::Matrix4d p = cp[i].inverse() * cp[j];
				getAreaWeights(p.inverse(), cf[i], wo[i], overlaps[i], total[i], cf[j], wo[j], overlaps[j], total[j]);
			}
		}

		wo = new double*[cf.size()];
		for(unsigned int i = 0; i < cf.size(); i++){
			unsigned int nr_pixels = cf[i]->camera->width * cf[i]->camera->height;
			wo[i] = new double[nr_pixels];
			for(unsigned int j = 0; j < nr_pixels; j++){
				if(total[i][j] < 0.00001){
					wo[i][j] = 0.0;
				}else{
					double fails	= occlusion_penalty*(total[i][j]-overlaps[i][j]);
					double others	= overlaps[i][j];
					wo[i][j] = others/(fails+others);
				}
			}
		}
		weights_old.push_back(wo);
	}

	for(unsigned int i = 0; i < cf.size(); i++){
		unsigned int nr_pixels = cf[i]->camera->width * cf[i]->camera->height;
		double sum = 0;
		for(unsigned int j = 0; j < nr_pixels; j++){
			sum += wo[i][j];
		}
		sum /= double(nr_pixels);
		for(unsigned int j = 0; j < nr_pixels; j++){
			if(total[i][j] == 0){
				wo[i][j] = sum;
			}
		}
	}
	for(unsigned int i = 0; i < cf.size(); i++){
		for(unsigned int j = 0; j < weights_old.size(); j++){
			getImageFromArray(cf[i]->camera->width, cf[i]->camera->height, weights_old[j][i]);
		}
	}
}


void ModelUpdater::getDynamicWeights(bool isbg, Matrix4d p, RGBDFrame* frame1, double * overlaps, double * occlusions, RGBDFrame* frame2, cv::Mat mask, int offset1, int offset2, std::vector< std::vector<int> > & interframe_connectionId, std::vector< std::vector<double> > & interframe_connectionStrength, double debugg){
	unsigned char  * src_rgbdata		= (unsigned char	*)(frame1->rgb.data);
	unsigned short * src_depthdata		= (unsigned short	*)(frame1->depth.data);
	float		   * src_normalsdata	= (float			*)(frame1->normals.data);

	unsigned char * src_detdata = (unsigned char*)(frame1->det_dilate.data);
	unsigned char * dst_detdata = (unsigned char*)(frame2->det_dilate.data);

	Camera * src_camera				= frame1->camera;
	const unsigned int src_width	= src_camera->width;
	const unsigned int src_height	= src_camera->height;
	const float src_idepth			= src_camera->idepth_scale;
	const float src_cx				= src_camera->cx;
	const float src_cy				= src_camera->cy;
	const float src_ifx				= 1.0/src_camera->fx;
	const float src_ify				= 1.0/src_camera->fy;

	unsigned char  * dst_rgbdata		= (unsigned char	*)(frame2->rgb.data);
	unsigned short * dst_depthdata		= (unsigned short	*)(frame2->depth.data);
	float		   * dst_normalsdata	= (float			*)(frame2->normals.data);
	unsigned char  * dst_maskdata       = (unsigned char	*)(mask.data);

	Camera * dst_camera				= frame2->camera;
	const unsigned int dst_width	= dst_camera->width;
	const unsigned int dst_height	= dst_camera->height;
	const float dst_idepth			= dst_camera->idepth_scale;
	const float dst_cx				= dst_camera->cx;
	const float dst_cy				= dst_camera->cy;
	const float dst_ifx				= 1.0/dst_camera->fx;
	const float dst_ify				= 1.0/dst_camera->fy;
	const float dst_fx				= dst_camera->fx;
	const float dst_fy				= dst_camera->fy;

	const unsigned int dst_width2	= dst_camera->width  - 2;
	const unsigned int dst_height2	= dst_camera->height - 2;

	float m00 = p(0,0); float m01 = p(0,1); float m02 = p(0,2); float m03 = p(0,3);
	float m10 = p(1,0); float m11 = p(1,1); float m12 = p(1,2); float m13 = p(1,3);
	float m20 = p(2,0); float m21 = p(2,1); float m22 = p(2,2); float m23 = p(2,3);

	//bool debugg = true;
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr src_cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr dst_cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	if(debugg){


		//		cv::namedWindow( "src_det_dilate", cv::WINDOW_AUTOSIZE );	cv::imshow( "src_det_dilate",	frame1->det_dilate);
		//		cv::namedWindow( "dst_det_dilate", cv::WINDOW_AUTOSIZE );	cv::imshow( "dst_det_dilate",	frame2->det_dilate);
		//		cv::waitKey(0);

		src_cloud->points.resize(src_width*src_height);
		for(unsigned int src_w = 0; src_w < src_width;src_w++){
			for(unsigned int src_h = 0; src_h < src_height;src_h++){
				int src_ind = src_h*src_width+src_w;
				float src_z = src_idepth*float(src_depthdata[src_ind]);
				float src_nx = src_normalsdata[3*src_ind+0];
				if(src_z > 0 && src_nx != 2){
					float src_ny = src_normalsdata[3*src_ind+1];
					float src_nz = src_normalsdata[3*src_ind+2];

					float src_x = (float(src_w) - src_cx) * src_z * src_ifx;
					float src_y = (float(src_h) - src_cy) * src_z * src_ify;


					float tx	= m00*src_x + m01*src_y + m02*src_z + m03;
					float ty	= m10*src_x + m11*src_y + m12*src_z + m13;
					float tz	= m20*src_x + m21*src_y + m22*src_z + m23;

					pcl::PointXYZRGBNormal point;
					point.x = tx;
					point.y = ty;
					point.z = tz;
					//point.r = 255*src_detdata[src_ind] != 0;
					//                    point.g = 0;
					//                    point.b = 0;

					point.r = src_detdata[src_ind];
					point.g = 0;
					point.b = 0;

					src_cloud->points[src_ind] = point;
				}
			}
		}

		dst_cloud->points.resize(dst_width*dst_height);
		for(unsigned int dst_w = 0; dst_w < dst_width;dst_w++){
			for(unsigned int dst_h = 0; dst_h < dst_height;dst_h++){
				int dst_ind = dst_h*dst_width+dst_w;
				float dst_z = dst_idepth*float(dst_depthdata[dst_ind]);
				float dst_nx = dst_normalsdata[3*dst_ind+0];
				if(dst_z > 0 && dst_nx != 2){
					float dst_ny = dst_normalsdata[3*dst_ind+1];
					float dst_nz = dst_normalsdata[3*dst_ind+2];

					float dst_x = (float(dst_w) - dst_cx) * dst_z * dst_ifx;
					float dst_y = (float(dst_h) - dst_cy) * dst_z * dst_ify;

					pcl::PointXYZRGBNormal point;
					point.x = dst_x;
					point.y = dst_y;
					point.z = dst_z;
					point.r = 0;
					point.g = 0;
					point.b = dst_detdata[dst_ind];
					dst_cloud->points[dst_ind] = point;
				}
			}
		}


		//		viewer->removeAllPointClouds();
		//		viewer->addPointCloud<pcl::PointXYZRGBNormal> (src_cloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(src_cloud), "scloud");
		//		viewer->addPointCloud<pcl::PointXYZRGBNormal> (dst_cloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(dst_cloud), "dcloud");
		//		viewer->spin();
	}

	int informations = 0;

	for(unsigned int src_w = 0; src_w < src_width;src_w++){
		for(unsigned int src_h = 0; src_h < src_height;src_h++){
			int src_ind = src_h*src_width+src_w;
			float src_z = src_idepth*float(src_depthdata[src_ind]);
			float src_nx = src_normalsdata[3*src_ind+0];
			if(src_z > 0 && src_nx != 2){
				float src_ny = src_normalsdata[3*src_ind+1];
				float src_nz = src_normalsdata[3*src_ind+2];

				float src_x = (float(src_w) - src_cx) * src_z * src_ifx;
				float src_y = (float(src_h) - src_cy) * src_z * src_ify;

				float tz	= m20*src_x + m21*src_y + m22*src_z + m23;
				if(tz < 0){continue;}

				float tx	= m00*src_x + m01*src_y + m02*src_z + m03;
				float ty	= m10*src_x + m11*src_y + m12*src_z + m13;
				float itz	= 1.0/tz;
				float dst_w	= dst_fx*tx*itz + dst_cx;
				float dst_h	= dst_fy*ty*itz + dst_cy;

				if((dst_w > 0) && (dst_h > 0) && (dst_w < dst_width2) && (dst_h < dst_height2)){
					unsigned int dst_ind = unsigned(dst_h+0.5) * dst_width + unsigned(dst_w+0.5);

					float dst_z = dst_idepth*float(dst_depthdata[dst_ind]);
					float dst_nx = dst_normalsdata[3*dst_ind+0];
					if(dst_z > 0 && dst_nx != 2){
						float dst_ny = dst_normalsdata[3*dst_ind+1];
						float dst_nz = dst_normalsdata[3*dst_ind+2];

						float dst_x = (float(dst_w) - dst_cx) * dst_z * dst_ifx;
						float dst_y = (float(dst_h) - dst_cy) * dst_z * dst_ify;

						double dst_noise = dst_z * dst_z;
						double point_noise = src_z * src_z;

						float tnx	= m00*src_nx + m01*src_ny + m02*src_nz;
						float tny	= m10*src_nx + m11*src_ny + m12*src_nz;
						float tnz	= m20*src_nx + m21*src_ny + m22*src_nz;

						double d = (tnx*(dst_x-tx) + tny*(dst_y-ty) + tnz*(dst_z-tz)) / (src_z*src_z + dst_z*dst_z);
						//double compare_mul = 1/sqrt(dst_noise*dst_noise + point_noise*point_noise);
						//d *= src_z;//compare_mul;
						double surface_angle = tnx*dst_nx+tny*dst_ny+tnz*dst_nz;

						if(fabs(d) <= 0.01){//If close, according noises, and angle of the surfaces similar: FUSE
							if(surface_angle > 0.8 && dst_maskdata[dst_ind] > 0){
								if(interframe_connectionId.size() > 0){
									double p_same = 0.99;
									double not_p_same = 1-p_same;
									double weight = -log(not_p_same);

									interframe_connectionId[src_ind+offset1].push_back(dst_ind+offset2);
									interframe_connectionStrength[src_ind+offset1].push_back(weight);
								}
								if(dst_detdata[dst_ind] == 0 && src_detdata[src_ind] == 0){
									overlaps[src_ind]++;
									if(debugg){
										informations++;
										dst_cloud->points[dst_ind].r = 0;
										dst_cloud->points[dst_ind].g = 255;
										dst_cloud->points[dst_ind].b = 0;
										src_cloud->points[src_ind].r = 0;
										src_cloud->points[src_ind].g = 255;
										src_cloud->points[src_ind].b = 0;
									}
								}
							}
						}else if(tz < dst_z && fabs(d) > 0.02 && dst_detdata[dst_ind] == 0 && src_detdata[src_ind] == 0){

							occlusions[src_ind] ++;
							if(debugg){
								informations++;
								src_cloud->points[src_ind].r = 255;
								src_cloud->points[src_ind].g = 0;
								src_cloud->points[src_ind].b = 255;
							}
						}
					}
				}
			}
		}
	}

	if(debugg){printf("informations: %i\n",informations);}
	if(debugg && informations > 0){
		viewer->removeAllPointClouds();
		viewer->addPointCloud<pcl::PointXYZRGBNormal> (src_cloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(src_cloud), "scloud");
		viewer->addPointCloud<pcl::PointXYZRGBNormal> (dst_cloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(dst_cloud), "dcloud");
		viewer->spin();
	}
}

std::vector< std::vector<float> > getImageProbs(reglib::RGBDFrame * frame, int blursize = 5){
	cv::Mat src						= frame->rgb.clone();
	unsigned short * depthdata		= (unsigned short	*)(frame->depth.data);
	const float idepth				= frame->camera->idepth_scale;

	cv::GaussianBlur( src, src, cv::Size(blursize,blursize), 0, 0, cv::BORDER_DEFAULT );

	unsigned char * srcdata = (unsigned char *)src.data;
	unsigned int width = src.cols;
	unsigned int height = src.rows;

	std::vector<float> dxc;
	dxc.resize(width*height);
	for(unsigned int i = 0; i < width*height;i++){dxc[i] = 0;}

	std::vector<float> dyc;
	dyc.resize(width*height);
	for(unsigned int i = 0; i < width*height;i++){dyc[i] = 0;}

	std::vector<double> src_dxdata;
	src_dxdata.resize(width*height);
	for(unsigned int i = 0; i < width*height;i++){src_dxdata[i] = 0;}

	std::vector<double> src_dydata;
	src_dydata.resize(width*height);
	for(unsigned int i = 0; i < width*height;i++){src_dydata[i] = 0;}

	std::vector<bool> maxima_dxdata;
	maxima_dxdata.resize(width*height);
	for(unsigned int i = 0; i < width*height;i++){maxima_dxdata[i] = 0;}

	std::vector<bool> maxima_dydata;
	maxima_dydata.resize(width*height);
	for(unsigned int i = 0; i < width*height;i++){maxima_dydata[i] = 0;}


	for(unsigned int w = 0; w < width;w++){
		for(unsigned int h = 0; h < height;h++){
			int ind = h*width+w;
			src_dxdata[ind] = 0;
			src_dydata[ind] = 0;
		}
	}

	unsigned int chans = 3;
	for(unsigned int c = 0; c < chans;c++){
		for(unsigned int w = 1; w < width;w++){
			for(unsigned int h = 1; h < height;h++){
				int ind = h*width+w;
				int dir		= 1;
				src_dxdata[ind] += fabs(float(srcdata[chans*ind+c] - srcdata[chans*(ind-1)+c]) / 255.0)/3.0;

				dir		= width;
				src_dydata[ind] += fabs(float(srcdata[chans*ind+c] - srcdata[chans*(ind-width)+c]) / 255.0)/3.0;
			}
		}

	}

	for(unsigned int w = 1; w < width-1;w++){
		for(unsigned int h = 1; h < height-1;h++){
			int ind = h*width+w;
			maxima_dxdata[ind] = (src_dxdata[ind] >= src_dxdata[ind-1]     && src_dxdata[ind] > src_dxdata[ind+1]);
			maxima_dydata[ind] = (src_dydata[ind] >= src_dydata[ind-width] && src_dydata[ind] > src_dydata[ind+width]);
		}
	}

	std::vector< std::vector<double> > probs;
	for(unsigned int c = 0; c < chans;c++){
		std::vector<double> Xvec;
		int dir;
		for(unsigned int w = 1; w < width-1;w++){
			for(unsigned int h = 1; h < height-1;h++){
				int ind = h*width+w;
				dir		= 1;
				Xvec.push_back(fabs(srcdata[chans*ind+c] - srcdata[chans*(ind-dir)+c]));

				dir		= width;
				Xvec.push_back(fabs(srcdata[chans*ind+c] - srcdata[chans*(ind-dir)+c]));
			}
		}

		Eigen::MatrixXd X = Eigen::MatrixXd::Zero(1,Xvec.size());
		for(unsigned int i = 0; i < Xvec.size();i++){X(0,i) = Xvec[i];}

		double stdval = 0;
		for(unsigned int i = 0; i < Xvec.size();i++){stdval += X(0,i)*X(0,i);}
		stdval = sqrt(stdval/double(Xvec.size()));

		DistanceWeightFunction2PPR2 * func = new DistanceWeightFunction2PPR2();
		func->zeromean				= true;
		func->maxp					= 0.99999;
		func->startreg				= 0.5;
		func->debugg_print			= false;
		//func->bidir					= true;
		func->maxd					= 256.0;
		func->histogram_size		= 256;
		func->fixed_histogram_size	= true;
		func->startmaxd				= func->maxd;
		func->starthistogram_size	= func->histogram_size;
		func->blurval				= 0.005;
		func->stdval2				= stdval;
		func->maxnoise				= stdval;
		func->reset();
		func->computeModel(X);

		std::vector<double> dx;  dx.resize(width*height);	for(unsigned int i = 0; i < width*height;i++){dx[i] = 0.5;}
		std::vector<double> dy;	 dy.resize(width*height);	for(unsigned int i = 0; i < width*height;i++){dy[i] = 0.5;}
		for(unsigned int w = 1; w < width;w++){
			for(unsigned int h = 1; h < height-1;h++){
				int ind = h*width+w;

				dir		= 1;
				dx[ind] = func->getProb(fabs(srcdata[chans*ind+c] - srcdata[chans*(ind-dir)+c]));

				dir		= width;
				dy[ind] = func->getProb(fabs(srcdata[chans*ind+c] - srcdata[chans*(ind-dir)+c]));
			}
		}
		delete func;

		probs.push_back(dx);
		probs.push_back(dy);
	}

	{
		std::vector<double> Xvec;
		for(unsigned int w = 1; w < width-1;w++){
			for(unsigned int h = 1; h < height-1;h++){
				int ind = h*width+w;
				float z = idepth*float(depthdata[ind]);

				if(w > 1){
					int dir = -1;
					int other2 = ind+2*dir;
					int other = ind+dir;

					float z3 = idepth*float(depthdata[other2]);
					float z2 = idepth*float(depthdata[other]);
					z2 = 2*z2-z3;

					if(z2 > 0 || z > 0){Xvec.push_back((z-z2)/(z*z+z2*z2));}
				}

				if(h > 1){
					int dir = -width;
					int other2 = ind+2*dir;
					int other = ind+dir;

					float z3 = idepth*float(depthdata[other2]);
					float z2 = idepth*float(depthdata[other]);
					z2 = 2*z2-z3;

					if(z2 > 0 || z > 0){Xvec.push_back((z-z2)/(z*z+z2*z2));}
				}
			}
		}

		Eigen::MatrixXd X = Eigen::MatrixXd::Zero(1,Xvec.size());
		for(unsigned int i = 0; i < Xvec.size();i++){X(0,i) = Xvec[i];}

		double stdval = 0;
		for(unsigned int i = 0; i < Xvec.size();i++){stdval += X(0,i)*X(0,i);}
		stdval = sqrt(stdval/double(Xvec.size()));

		DistanceWeightFunction2PPR2 * funcZ = new DistanceWeightFunction2PPR2();
		funcZ->zeromean				= true;
		funcZ->startreg				= 0.002;
		funcZ->debugg_print			= false;
		//funcZ->bidir				= true;
		funcZ->maxp					= 0.999999;
		funcZ->maxd					= 0.1;
		funcZ->histogram_size		= 100;
		funcZ->fixed_histogram_size	= true;
		funcZ->startmaxd			= funcZ->maxd;
		funcZ->starthistogram_size	= funcZ->histogram_size;
		funcZ->blurval				= 0.5;
		funcZ->stdval2				= stdval;
		funcZ->maxnoise				= stdval;
		funcZ->reset();
		funcZ->computeModel(X);

		for(unsigned int w = 1; w < width-1;w++){
			for(unsigned int h = 1; h < height-1;h++){
				int ind = h*width+w;
				float z = idepth*float(depthdata[ind]);

				if(w > 1){
					int dir = -1;
					int other2 = ind+2*dir;
					int other = ind+dir;

					float z3 = idepth*float(depthdata[other2]);
					float z2 = idepth*float(depthdata[other]);
					z2 = 2*z2-z3;

					if(z2 > 0 || z > 0){Xvec.push_back((z-z2)/(z*z+z2*z2));}
				}

				if(h > 1){
					int dir = -width;
					int other2 = ind+2*dir;
					int other = ind+dir;

					float z3 = idepth*float(depthdata[other2]);
					float z2 = idepth*float(depthdata[other]);
					z2 = 2*z2-z3;

					if(z2 > 0 || z > 0){Xvec.push_back((z-z2)/(z*z+z2*z2));}
				}
			}
		}

		std::vector<double> dx;  dx.resize(width*height);	for(unsigned int i = 0; i < width*height;i++){dx[i] = 0.5;}
		std::vector<double> dy;	 dy.resize(width*height);	for(unsigned int i = 0; i < width*height;i++){dy[i] = 0.5;}

		for(unsigned int w = 1; w < width-1;w++){
			for(unsigned int h = 1; h < height-1;h++){
				int ind = h*width+w;
				float z = idepth*float(depthdata[ind]);

				if(w > 1){
					int dir = -1;
					int other2 = ind+2*dir;
					int other = ind+dir;


					float z3 = idepth*float(depthdata[other2]);
					float z2 = idepth*float(depthdata[other]);

					float dz = fabs(z-z2);
					if(z3 > 0){dz = std::min(dz,fabs(z- (2*z2-z3)));}
					if(z2 > 0 || z > 0){dx[ind] = funcZ->getProb(dz/(z*z+z2*z2));}
				}

				if(h > 1){
					int dir = -width;

					int other2 = ind+2*dir;
					int other = ind+dir;

					float z3 = idepth*float(depthdata[other2]);
					float z2 = idepth*float(depthdata[other]);

					float dz = fabs(z-z2);
					if(z3 > 0){dz = std::min(dz,fabs(z- (2*z2-z3)));}
					if(z2 > 0 || z > 0){dy[ind] = funcZ->getProb(dz/(z*z+z2*z2));}
				}
			}
		}

		delete funcZ;
		probs.push_back(dx);
		probs.push_back(dy);
	}


	for(unsigned int w = 0; w < width;w++){
		for(unsigned int h = 0; h < height;h++){
			int ind = h*width+w;

			float probX = 0;
			float probY = 0;

			if(w > 0 && w < width-1){
				float ax = 0.5;
				float bx = 0.5;
				for(unsigned int p = 0; p < probs.size()-2; p+=2){
					float pr = probs[p][ind];
					ax *= pr;
					bx *= 1.0-pr;
				}
				float px = ax/(ax+bx);
				float current = 0;
				if(!frame->det_dilate.data[ind]){	current = (1-px) * float(maxima_dxdata[ind]);}
				else{								current = std::max(float(1-probs[probs.size()-2][ind]),0.8f*(1-px) * float(maxima_dxdata[ind]));}
				probX = 1-current;
			}

			if(h > 0 && h < height-1){
				float ay = 0.5;
				float by = 0.5;
				for(unsigned int p = 1; p < probs.size()-2; p+=2){
					float pr = probs[p][ind];
					ay *= pr;
					by *= 1.0-pr;
				}
				float py = ay/(ay+by);
				float current = 0;
				if(!frame->det_dilate.data[ind]){	current = (1-py) * float(maxima_dydata[ind]);}
				else{								current = std::max(float(1-probs[probs.size()-1][ind]),0.8f*(1-py) * float(maxima_dydata[ind]));}
				probY = 1-current;
			}

			dxc[ind] = std::min(probX,probY);
			dyc[ind] = std::min(probX,probY);
		}
	}


	std::vector< std::vector<float> > probs2;
	probs2.push_back(dxc);
	probs2.push_back(dyc);

	cv::Mat edges;
	edges.create(height,width,CV_32FC3);
	float * edgesdata = (float *)edges.data;

	for(unsigned int i = 0; i < width*height;i++){
		edgesdata[3*i+0] = 0;
		edgesdata[3*i+1] = dxc[i];
		edgesdata[3*i+2] = dyc[i];
	}

	//		cv::namedWindow( "src", cv::WINDOW_AUTOSIZE );          cv::imshow( "src",	src);
	//		cv::namedWindow( "edges", cv::WINDOW_AUTOSIZE );          cv::imshow( "edges",	edges);
	//		cv::waitKey(0);

	return probs2;
}

std::vector<int> doInference(std::vector<double> & prior, std::vector< std::vector<int> > & connectionId, std::vector< std::vector<double> > & connectionStrength){
	double start_inf = getTime();

	unsigned int nr_data = prior.size();
	unsigned int nr_edges = 0;
	for(unsigned int j = 0; j < nr_data;j++){
		nr_edges += connectionId[j].size();
	}

	gc::Graph<double,double,double> * g = new gc::Graph<double,double,double>(nr_data,nr_edges);
	for(unsigned int i = 0; i < nr_data;i++){
		g -> add_node();
		double p_fg = prior[i];
		if(p_fg < 0){
			g -> add_tweights( i, 0, 0 );
			continue;
		}
		double p_bg = 1-p_fg;
		double weightFG = -log(p_fg);
		double weightBG = -log(p_bg);
		g -> add_tweights( i, weightFG, weightBG );
	}

	for(unsigned int i = 0; i < nr_data;i++){
		for(unsigned int j = 0; j < connectionId[i].size();j++){
			double weight = connectionStrength[i][j];
			g -> add_edge( i, connectionId[i][j], weight, weight );
		}
	}

	g -> maxflow();

	std::vector<int> labels;
	labels.resize(nr_data);
	for(unsigned int i = 0; i < nr_data;i++){labels[i] = g->what_segment(i);}
	delete g;

	double end_inf = getTime();
	printf("nr data: %i nr edges: %i inference time: %10.10fs\n",nr_data,nr_edges,end_inf-start_inf);

	return labels;
}
/*
void ModelUpdater::getDynamicWeights(bool store_distance, std::vector<double> & dvec, std::vector<double> & nvec, DistanceWeightFunction2 * dfunc, DistanceWeightFunction2 * nfunc, Matrix4d p, RGBDFrame* frame1, double * overlaps, double * occlusions, RGBDFrame* frame2,  int offset1, int offset2, std::vector< std::vector<int> > & interframe_connectionId, std::vector< std::vector<float> > & interframe_connectionStrength, double debugg){
	std::vector<superpoint> framesp1_test		= frame1->getSuperPoints(Eigen::Matrix4d::Identity(),10,false);
	std::vector<ReprojectionResult> rr_vec_test	= frame2->getReprojections(framesp1_test,p,0,false);

	double inlierratio = double(rr_vec_test.size())/double(framesp1_test.size());

	if( inlierratio < 0.01 ){return ;}
	std::vector<superpoint> framesp1		= frame1->getSuperPoints();
	std::vector<superpoint> framesp2		= frame2->getSuperPoints();//p);

	getDynamicWeights(store_distance,dvec,nvec,dfunc,nfunc,p,frame1,framesp1_test,framesp1,overlaps,occlusions,frame2,framesp2,offset1,offset2,interframe_connectionId, interframe_connectionStrength, debugg);
}

void ModelUpdater::getDynamicWeights(bool store_distance, std::vector<double> & dvec, std::vector<double> & nvec, DistanceWeightFunction2 * dfunc, DistanceWeightFunction2 * nfunc, Matrix4d p, RGBDFrame* frame1, std::vector<superpoint> & tframesp1_test, std::vector<superpoint> & tframesp1, double * overlaps, double * occlusions, RGBDFrame* frame2, std::vector<superpoint> & tframesp2,  int offset1, int offset2, std::vector< std::vector<int> > & interframe_connectionId, std::vector< std::vector<float> > & interframe_connectionStrength, double debugg){
	double totalstartTime = getTime();

	std::string currentString;
	double startTime;
	startTime = getTime();
	unsigned char * src_detdata = (unsigned char*)(frame1->det_dilate.data);
	unsigned char * dst_detdata = (unsigned char*)(frame2->det_dilate.data);

	std::vector<superpoint> & framesp1_test		= tframesp1_test;//frame1->getSuperPoints(Eigen::Matrix4d::Identity(),10,false);
	std::vector<ReprojectionResult> rr_vec_test	= frame2->getReprojections(framesp1_test,p,0,false);
	double inlierratio = double(rr_vec_test.size())/double(framesp1_test.size());

	currentString = "part1";
	if(benchtime.count(currentString) == 0){benchtime[currentString] = 0;}
	benchtime[currentString] += getTime() - startTime;
	startTime = getTime();

	if( inlierratio < 0.01 ){return ;}

	std::vector<superpoint> & framesp1		= tframesp1;//frame1->getSuperPoints();
	std::vector<superpoint> & framesp2		= tframesp2;//frame2->getSuperPoints();

	currentString = "part2";
	if(benchtime.count(currentString) == 0){benchtime[currentString] = 0;}
	benchtime[currentString] += getTime() - startTime;
	startTime = getTime();

	std::vector<ReprojectionResult> rr_vec	= frame2->getReprojections(framesp1,p,0,false);
	inlierratio = double(rr_vec.size())/double(framesp1.size());
	unsigned long nr_rr = rr_vec.size();

	currentString = "part3";
	if(benchtime.count(currentString) == 0){benchtime[currentString] = 0;}
	benchtime[currentString] += getTime() - startTime;
	startTime = getTime();

	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr src_cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr dst_cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	if(debugg){
		src_cloud = getPointCloudFromVector(framesp1,3,255,0,255);
		dst_cloud = getPointCloudFromVector(framesp2,3,0,0,255);
	}

	double threshold = 0;
	if(!store_distance){
		threshold = pow(20.0*dfunc->getNoise(),2);
	}

	int totsum = 0;
	for(unsigned long ind = 0; ind < nr_rr;ind++){
		ReprojectionResult & rr = rr_vec[ind];
		unsigned int src_ind = rr.src_ind;
		superpoint & src_p = framesp1[src_ind];
		if(src_p.point(2) < 0.0){continue;}

		totsum++;
		unsigned int dst_ind = rr.dst_ind;
		superpoint & dst_p = framesp2[dst_ind];
		double src_variance = 1.0/src_p.point_information;
		double dst_variance = 1.0/dst_p.point_information;
		double total_variance = src_variance+dst_variance;
		double total_stdiv = sqrt(total_variance);
		double d = rr.residualZ/total_stdiv;
		double surface_angle = rr.angle;
		if(store_distance){
			dvec.push_back(d);
			nvec.push_back(1-surface_angle);
		}else{
			double dE2 = rr.residualE2/total_stdiv;
			double p_overlap_angle = nfunc->getProb(1-surface_angle);
			double p_overlap = dfunc->getProb(d);
			double p_occlusion = dfunc->getProbInfront(d);

			p_overlap *= p_overlap_angle;

			if(p_overlap > 0.001 && offset1 >= 0 && offset2 >= 0 && dE2 < threshold){
				interframe_connectionId[offset1+src_ind].push_back(offset2+dst_ind);
				interframe_connectionStrength[offset1+src_ind].push_back(-log(1-p_overlap));
			}

			if(debugg){
				src_cloud->points[src_ind].r = 255.0 * p_overlap;
				src_cloud->points[src_ind].g = 255.0 * p_occlusion;
				src_cloud->points[src_ind].b = 0;
			}

			if(dst_detdata[dst_ind] != 0){continue;}
			if(src_detdata[src_ind] != 0){continue;}

			overlaps[src_ind] = std::min(0.9,std::max(overlaps[src_ind],p_overlap));
			occlusions[src_ind] = std::min(0.9,std::max(occlusions[src_ind],p_occlusion));
		}
	}
	currentString = "part4";
	if(benchtime.count(currentString) == 0){benchtime[currentString] = 0;}
	benchtime[currentString] += getTime() - startTime;
	startTime = getTime();

	currentString = "total_getDyn";
	if(benchtime.count(currentString) == 0){benchtime[currentString] = 0;}
	benchtime[currentString] += getTime() - totalstartTime;
	startTime = getTime();

	if(debugg && totsum > 0){
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr src_cloud = getPointCloudFromVector(framesp1,3,255,0,255);
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr dst_cloud = getPointCloudFromVector(framesp2,3,0,0,255);
		viewer->removeAllPointClouds();
		viewer->addPointCloud<pcl::PointXYZRGBNormal> (src_cloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(src_cloud), "scloud");
		viewer->addPointCloud<pcl::PointXYZRGBNormal> (dst_cloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(dst_cloud), "dcloud");
		viewer->spin();
	}
}
*/

void ModelUpdater::getDynamicWeights(bool store_distance, std::vector<double> & dvec, std::vector<double> & nvec, DistanceWeightFunction2 * dfunc, DistanceWeightFunction2 * nfunc, Matrix4d p, RGBDFrame* frame1, double * overlaps, double * occlusions, double * notocclusions, RGBDFrame* frame2,  int offset1, int offset2, std::vector< std::vector<int> > & interframe_connectionId, std::vector< std::vector<float> > & interframe_connectionStrength, double debugg){
	std::vector<superpoint> framesp1_test		= frame1->getSuperPoints(Eigen::Matrix4d::Identity(),10,false);
	std::vector<ReprojectionResult> rr_vec_test	= frame2->getReprojections(framesp1_test,p,0,false);

	double inlierratio = double(rr_vec_test.size())/double(framesp1_test.size());

	if( inlierratio < 0.01 ){return ;}
	std::vector<superpoint> framesp1		= frame1->getSuperPoints();
	std::vector<superpoint> framesp2		= frame2->getSuperPoints();//p);

	getDynamicWeights(store_distance,dvec,nvec,dfunc,nfunc,p,frame1,framesp1_test,framesp1,overlaps,occlusions,notocclusions,frame2,framesp2,offset1,offset2,interframe_connectionId, interframe_connectionStrength, debugg);
}

void ModelUpdater::getDynamicWeights(bool store_distance, std::vector<double> & dvec, std::vector<double> & nvec, DistanceWeightFunction2 * dfunc, DistanceWeightFunction2 * nfunc, Matrix4d p, RGBDFrame* frame1, std::vector<superpoint> & tframesp1_test, std::vector<superpoint> & tframesp1, double * overlaps, double * occlusions, double * notocclusions, RGBDFrame* frame2, std::vector<superpoint> & tframesp2,  int offset1, int offset2, std::vector< std::vector<int> > & interframe_connectionId, std::vector< std::vector<float> > & interframe_connectionStrength, double debugg){
	double totalstartTime = getTime();

	std::string currentString;
	double startTime;
	startTime = getTime();
	unsigned char * src_detdata = (unsigned char*)(frame1->det_dilate.data);
	unsigned char * dst_detdata = (unsigned char*)(frame2->det_dilate.data);

	std::vector<superpoint> & framesp1_test		= tframesp1_test;//frame1->getSuperPoints(Eigen::Matrix4d::Identity(),10,false);
	std::vector<ReprojectionResult> rr_vec_test	= frame2->getReprojections(framesp1_test,p,0,false);
	double inlierratio = double(rr_vec_test.size())/double(framesp1_test.size());

	currentString = "part1";
	if(benchtime.count(currentString) == 0){benchtime[currentString] = 0;}
	benchtime[currentString] += getTime() - startTime;
	startTime = getTime();

	if( inlierratio < 0.01 ){return ;}

	std::vector<superpoint> & framesp1		= tframesp1;//frame1->getSuperPoints();
	std::vector<superpoint> & framesp2		= tframesp2;//frame2->getSuperPoints();

	currentString = "part2";
	if(benchtime.count(currentString) == 0){benchtime[currentString] = 0;}
	benchtime[currentString] += getTime() - startTime;
	startTime = getTime();

	std::vector<ReprojectionResult> rr_vec	= frame2->getReprojections(framesp1,p,0,false);
	inlierratio = double(rr_vec.size())/double(framesp1.size());
	unsigned long nr_rr = rr_vec.size();

	currentString = "part3";
	if(benchtime.count(currentString) == 0){benchtime[currentString] = 0;}
	benchtime[currentString] += getTime() - startTime;
	startTime = getTime();

	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr src_cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr dst_cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	if(debugg){
		src_cloud = getPointCloudFromVector(framesp1,3,255,0,255);
		dst_cloud = getPointCloudFromVector(framesp2,3,0,0,255);
	}

	double threshold = 0;
	if(!store_distance){
		threshold = pow(20.0*dfunc->getNoise(),2);
	}

	int totsum = 0;
	for(unsigned long ind = 0; ind < nr_rr;ind++){
		ReprojectionResult & rr = rr_vec[ind];
		unsigned int src_ind = rr.src_ind;
		superpoint & src_p = framesp1[src_ind];
		if(src_p.point(2) < 0.0){continue;}

		totsum++;
		unsigned int dst_ind = rr.dst_ind;
		superpoint & dst_p = framesp2[dst_ind];
		double src_variance = 1.0/src_p.point_information;
		double dst_variance = 1.0/dst_p.point_information;
		double total_variance = src_variance+dst_variance;
		double total_stdiv = sqrt(total_variance);
		double d = rr.residualZ/total_stdiv;
		double surface_angle = rr.angle;
		if(store_distance){
			dvec.push_back(d);
			nvec.push_back(1-surface_angle);
		}else{
			double dE2 = rr.residualE2/total_stdiv;
			double p_overlap_angle = nfunc->getProb(1-surface_angle);
			double p_overlap = dfunc->getProb(d);
			double p_occlusion = dfunc->getProbInfront(d);//std::max(1 - 1e-6,std::max(1e-6,dfunc->getProbInfront(d)));
			double p_behind = 1-p_overlap-p_occlusion;

			p_overlap *= p_overlap_angle;

			if(p_overlap > 0.001 && offset1 >= 0 && offset2 >= 0 && dE2 < threshold){
				interframe_connectionId[offset1+src_ind].push_back(offset2+dst_ind);
				interframe_connectionStrength[offset1+src_ind].push_back(-log(1-p_overlap));
			}

			if(debugg){
				src_cloud->points[src_ind].r = 255.0 * p_overlap;
				src_cloud->points[src_ind].g = 255.0 * p_occlusion;
				src_cloud->points[src_ind].b = 0;
			}

			if(dst_detdata[dst_ind] != 0){continue;}
			if(src_detdata[src_ind] != 0){continue;}

			double olp = overlaps[src_ind];
			double nolp = 1-olp;
			nolp *= (1-p_overlap);
			//overlaps[src_ind] = std::min(0.9999,std::max(olp,p_overlap));
			overlaps[src_ind] = std::min(0.999999999,1-nolp);
			occlusions[src_ind] += p_occlusion;//std::min(0.9,std::max(occlusions[src_ind],p_occlusion));
			notocclusions[src_ind]++;

			//overlaps[src_ind] = std::min(0.9999,std::max(overlaps[src_ind],p_overlap));
			//double prev = 1.0-occlusions[src_ind];
			//notocclusions[src_ind] *= 1.0-p_occlusion;//std::min(0.9,std::max(occlusions[src_ind],p_occlusion));


			//occlusions[src_ind] *= p_occlusion;//std::min(0.9,std::max(occlusions[src_ind],p_occlusion));
			//notocclusions[src_ind] *= 1.0-p_occlusion;//std::min(0.9,std::max(occlusions[src_ind],p_occlusion));
		}
	}
	currentString = "part4";
	if(benchtime.count(currentString) == 0){benchtime[currentString] = 0;}
	benchtime[currentString] += getTime() - startTime;
	startTime = getTime();

	currentString = "total_getDyn";
	if(benchtime.count(currentString) == 0){benchtime[currentString] = 0;}
	benchtime[currentString] += getTime() - totalstartTime;
	startTime = getTime();

	if(false && !store_distance){
		cv::Mat overlapsimg;
		overlapsimg.create(frame1->camera->height,frame1->camera->width,CV_32FC3);
		float * overlapsimgdata = (float*)overlapsimg.data;
		cv::Mat occlusionsimg;
		occlusionsimg.create(frame1->camera->height,frame1->camera->width,CV_32FC3);
		float * occlusionsimgdata = (float*)occlusionsimg.data;
		for(unsigned long ind = 0; ind < frame1->camera->height*frame1->camera->width;ind++){
			overlapsimgdata[3*ind+0] = overlaps[ind];
			overlapsimgdata[3*ind+1] = overlaps[ind];
			overlapsimgdata[3*ind+2] = overlaps[ind];

			occlusionsimgdata[3*ind+0] = occlusions[ind]/std::max(1.0,notocclusions[ind]);
			occlusionsimgdata[3*ind+1] = occlusionsimgdata[3*ind+0];
			occlusionsimgdata[3*ind+2] = occlusionsimgdata[3*ind+0];
		}
		cv::namedWindow( "rgb",				cv::WINDOW_AUTOSIZE );	cv::imshow( "rgb",			frame1->rgb );
		cv::namedWindow( "occlusionsimg",	cv::WINDOW_AUTOSIZE );	cv::imshow( "occlusionsimg",occlusionsimg );
		cv::namedWindow( "overlapsimg",		cv::WINDOW_AUTOSIZE );	cv::imshow( "overlapsimg",	overlapsimg );
		cv::waitKey(0);
	}

	if(debugg && totsum > 0){
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr src_cloud = getPointCloudFromVector(framesp1,3,255,0,255);
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr dst_cloud = getPointCloudFromVector(framesp2,3,0,0,255);
		viewer->removeAllPointClouds();
		viewer->addPointCloud<pcl::PointXYZRGBNormal> (src_cloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(src_cloud), "scloud");
		viewer->addPointCloud<pcl::PointXYZRGBNormal> (dst_cloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(dst_cloud), "dcloud");
		viewer->spin();
	}
}

int setupPriors(int method,float current_occlusions, float current_overlaps, float bg_occlusions, float bg_overlaps, bool valid,
				float & prior_moving, float & prior_dynamic, float & prior_static, float & foreground_weight, float & background_weight){
	float bias = 0.1;
	float maxdiff = 0.001;

	foreground_weight = 0;
	background_weight = 0;
	prior_moving    = 0.0;
	prior_dynamic   = 0.0;
	prior_static    = 0.0;

	if(method == 0){
		if(valid){
			float p_moving_or_dynamic = std::max((bg_occlusions-current_occlusions)*(1.0f-current_overlaps),0.0f);
			float p_moving  = 0.5f*p_moving_or_dynamic + current_occlusions;
			float p_dynamic = 0.5f*p_moving_or_dynamic + std::max((bg_occlusions-current_occlusions)		*   current_overlaps,0.0f);
			float p_static  = std::max(bg_overlaps-p_moving-p_dynamic,0.0f);

			float leftover = 1.0f-p_moving-p_dynamic-p_static;
			float notMoving = current_overlaps;

			float p_moving_leftover   =							 0.5*leftover*(1-notMoving); //(1-notMoving)*leftover/4.0;
			float p_dynamic_leftover  = 0.5*leftover*notMoving + 0.5*leftover*(1-notMoving); //(1-notMoving)*leftover/4.0;
			float p_static_leftover   = 0.5*leftover*notMoving;

			prior_moving    = p_moving+p_moving_leftover;
			prior_dynamic   = p_dynamic+p_dynamic_leftover;
			prior_static    = p_static+p_static_leftover;

			double p_fg = std::min( 1.0f-maxdiff ,std::max( maxdiff , prior_moving+prior_dynamic))-bias;
			double p_bg = std::min( 1.0f-maxdiff ,std::max( maxdiff , prior_static))+bias;
			double norm = p_fg + p_bg;
			p_fg /= norm;
			p_bg /= norm;


			if(norm > 0){
				foreground_weight = -log(p_fg);
				background_weight = -log(p_bg);
			}
		}
	}

	if(method == 1){
		if(valid){

			float minprob = 0.01;
			float bias = 0.001;
			float overlap_same_prob = 0.95;

			//Prob behind all bg
			float prob_behind = 1-(bg_overlaps+bg_occlusions);
			float leftover = 1-prob_behind;

			float p_moving_or_dynamic = std::max((bg_occlusions-current_occlusions)*(1.0f-current_overlaps),0.0f);
			float p_moving  = 0.5f*p_moving_or_dynamic + current_occlusions;
			float p_dynamic = 0.5f*p_moving_or_dynamic + std::max((bg_occlusions-current_occlusions)		*   current_overlaps,0.0f);

			float p_static  = std::max(overlap_same_prob*bg_overlaps-p_moving-p_dynamic,0.0f);

			leftover = 1.0f-p_moving-p_dynamic-p_static;
			float notMoving = current_overlaps;

			float p_moving_leftover   =	0.5*leftover*(1-notMoving);
			float p_dynamic_leftover  = 0.5*leftover;
			float p_static_leftover   = 0.5*leftover*notMoving;

			prior_moving    = (1.0-4.0*minprob-bias)*((1.0-prob_behind)*(p_moving+p_moving_leftover)	+0.25*prob_behind)+		minprob;
			prior_dynamic   = (1.0-4.0*minprob-bias)*((1.0-prob_behind)*(p_dynamic+p_dynamic_leftover)	+0.25*prob_behind)+		minprob;
			prior_static    = (1.0-4.0*minprob-bias)*((1.0-prob_behind)*(p_static+p_static_leftover)	+0.50*prob_behind)+ 2.0*minprob + bias;

			foreground_weight = -log(prior_moving+prior_dynamic);
			background_weight = -log(prior_static);
		}
	}

	if(method == 2){
		if(valid){

			float minprob = 0.1;
			float bias = 0.00;
			float overlap_same_prob = 0.7;

			//Prob behind all bg
			float prob_behind = 1-(bg_overlaps+bg_occlusions);
			float leftover = 1-prob_behind;

			float p_moving_or_dynamic = std::max((bg_occlusions-current_occlusions)*(1.0f-current_overlaps),0.0f);
			float p_moving  = 0.5f*p_moving_or_dynamic + current_occlusions;
			float p_dynamic = 0.5f*p_moving_or_dynamic + std::max((bg_occlusions-current_occlusions)		*   current_overlaps,0.0f);

			float p_static  = std::max(overlap_same_prob*bg_overlaps-p_moving-p_dynamic,0.0f);

			leftover = 1.0f-p_moving-p_dynamic-p_static;
			float notMoving = current_overlaps;

			float p_moving_leftover   =	0.5*leftover*(1-notMoving);
			float p_dynamic_leftover  = 0.5*leftover;
			float p_static_leftover   = 0.5*leftover*notMoving;

			prior_moving    = p_moving+p_moving_leftover;
			prior_dynamic   = (1.0-prob_behind)*(p_dynamic+p_dynamic_leftover);
			prior_static    = (1.0-prob_behind)*(p_static+p_static_leftover);

			double norm = prior_moving+prior_dynamic+prior_static;
			prior_dynamic   += (1.0-norm)*0.5;
			prior_static    += (1.0-norm)*0.5;

			//			prior_moving    = (1.0-3.0*minprob-bias)*prior_moving  + minprob;
			//			prior_dynamic   = (1.0-3.0*minprob-bias)*prior_dynamic + minprob;
			//			prior_static    = (1.0-3.0*minprob-bias)*prior_static  + minprob + bias;

			//			prior_moving    = (1.0-3.0*minprob-bias)*prior_moving  + minprob;
			//			prior_dynamic   = (1.0-3.0*minprob-bias)*prior_dynamic + minprob;
			//			prior_static    = (1.0-3.0*minprob-bias)*prior_static  + minprob + bias;

			double prob_fg = prior_dynamic;
			double prob_bg = prior_moving+prior_static;
			//double norm = prob_fg+prob_bg;

			foreground_weight = -log((1.0-2.0*minprob)*prob_fg+minprob);//+norm*0.5);
			background_weight = -log((1.0-2.0*minprob)*prob_bg+minprob);//+norm*0.5);

			//			prior_moving    += norm/3.0;
			//			prior_dynamic   += norm/3.0;
			//			prior_static    += norm/3.0;
		}
	}

	if(method == 3){
		if(valid){
			float minprob = 0.01;
			float bias = 0.01;
			float overlap_same_prob = 0.75;

			//Prob behind all bg
			float prob_behind = 1-(bg_overlaps+bg_occlusions);

			float p_moving_or_dynamic = std::max((bg_occlusions-current_occlusions)*(1.0f-current_overlaps),0.0f);
			float p_moving  = 0.5f*p_moving_or_dynamic + current_occlusions;
			float p_dynamic = 0.5f*p_moving_or_dynamic + std::max((bg_occlusions-current_occlusions)		*   current_overlaps,0.0f);

			float p_static  = std::max(overlap_same_prob*bg_overlaps-p_moving-p_dynamic,0.0f);

			float leftover				= 1.0f-p_moving-p_dynamic-p_static;
			float notMoving				= 1.0f-current_occlusions;//current_overlaps;

			float p_moving_leftover   =	0.5*leftover*(1-notMoving);
			float p_dynamic_leftover  = 0.5*leftover;
			float p_static_leftover   = 0.5*leftover*notMoving;

			prior_moving  = p_moving+p_moving_leftover;
			prior_dynamic = (1.0-prob_behind)*(p_dynamic+p_dynamic_leftover);
			prior_static  = (1.0-prob_behind)*(p_static+p_static_leftover);

			float p_dynamic_or_static = 1.0f-p_moving-p_dynamic-p_static;
			prior_dynamic += 0.5*p_dynamic_or_static;
			prior_static  += 0.5*p_dynamic_or_static;

			foreground_weight = -log((1.0-2.0*minprob-bias)*(prior_moving+prior_dynamic) + minprob		 );
			background_weight = -log((1.0-2.0*minprob-bias)* prior_static				 + minprob + bias);

			prior_moving  = (1.0-3.0*minprob-bias)* prior_moving	+ minprob;
			prior_dynamic = (1.0-3.0*minprob-bias)* prior_dynamic	+ minprob;
			prior_static  = (1.0-3.0*minprob-bias)* prior_static	+ minprob + bias;
		}else{
			prior_static = prior_dynamic = prior_moving = 1.0/3.0;
			foreground_weight = -log(0.5);
			background_weight = -log(0.5);
		}
	}
	return 0;
}

void ModelUpdater::computeMovingDynamicStatic(std::vector<cv::Mat> & movemask, std::vector<cv::Mat> & dynmask, vector<Matrix4d> bgcp, vector<RGBDFrame*> bgcf, vector<Matrix4d> poses, vector<RGBDFrame*> frames, bool debugg, std::string savePath){
	static int segment_run_counter = -1;
	segment_run_counter++;

	double computeMovingDynamicStatic_startTime = getTime();

	SegmentationResults sr;

	int tot_nr_pixels = 0;
	std::vector<int> offsets;

	for(unsigned int i = 0; i < frames.size(); i++){
		offsets.push_back(tot_nr_pixels);
		unsigned int nr_pixels = frames[i]->camera->width * frames[i]->camera->height;
		tot_nr_pixels += nr_pixels;
	}

	std::vector<unsigned char> labels;
	labels.resize(tot_nr_pixels);

	//Graph...
	std::vector< std::vector<int> > interframe_connectionId;
	std::vector< std::vector<float> > interframe_connectionStrength;
	interframe_connectionId.resize(tot_nr_pixels);
	interframe_connectionStrength.resize(tot_nr_pixels);

	int current_point         = 0;
	float * priors            = new float[3*tot_nr_pixels];
	float * prior_weights     = new float[2*tot_nr_pixels];
	bool * valids             = new bool[tot_nr_pixels];

	std::vector<double> dvec;
	std::vector<double> nvec;
	DistanceWeightFunction2 * dfunc;
	DistanceWeightFunction2 * nfunc;

	double startTime = getTime();

	std::vector< std::vector<superpoint> > framesp_test;
	std::vector< std::vector<superpoint> > framesp;
	for(unsigned int i = 0; i < frames.size(); i++){
		framesp_test.push_back(frames[i]->getSuperPoints(Eigen::Matrix4d::Identity(),10,false));
		framesp.push_back(frames[i]->getSuperPoints());
	}
	printf("frames init time: %5.5fs\n",getTime()-startTime);

	startTime = getTime();
	std::vector< std::vector<superpoint> > bgsp;
	for(unsigned int i = 0; i < bgcf.size(); i++){
		bgsp.push_back(bgcf[i]->getSuperPoints());
	}
	printf("bg init time:     %5.5fs\n",getTime()-startTime);

	startTime = getTime();
	for(unsigned int i = 0; i < frames.size(); i++){
		std::vector<superpoint> & framesp1_test = framesp_test[i];
		std::vector<superpoint> & framesp1		= framesp[i];
		for(unsigned int j = 0; j < frames.size(); j++){
			if(i == j){continue;}
			Eigen::Matrix4d p = poses[i].inverse() * poses[j];
			std::vector<superpoint> & framesp2 = framesp[j];
			getDynamicWeights(true,dvec,nvec,dfunc,nfunc,p.inverse(),frames[i],framesp1_test,framesp1, 0, 0, 0, frames[j],framesp2,offsets[i],offsets[j],interframe_connectionId,interframe_connectionStrength,false);
		}
	}

	double dstdval = 0;
	for(unsigned int i = 0; i < dvec.size(); i++){dstdval += dvec[i]*dvec[i];}
	dstdval = sqrt(dstdval/double(dvec.size()-1));

	GeneralizedGaussianDistribution * dggdnfunc	= new GeneralizedGaussianDistribution(true,true,false,true,true);
	dggdnfunc->nr_refineiters					= 4;
	DistanceWeightFunction2PPR3 * dfuncTMP		= new DistanceWeightFunction2PPR3(dggdnfunc);
	dfunc = dfuncTMP;
	dfuncTMP->startreg				= 0.00;
	dfuncTMP->max_under_mean		= false;
	dfuncTMP->debugg_print			= false;
	dfuncTMP->bidir					= true;
	dfuncTMP->zeromean				= false;
	dfuncTMP->maxp					= 0.9999;
	dfuncTMP->maxd					= 0.5;
	dfuncTMP->histogram_size		= 1000;
	dfuncTMP->fixed_histogram_size	= false;
	dfuncTMP->startmaxd				= dfuncTMP->maxd;
	dfuncTMP->starthistogram_size	= dfuncTMP->histogram_size;
	dfuncTMP->blurval				= 0.5;
	dfuncTMP->maxnoise				= dstdval;
	dfuncTMP->compute_infront		= true;
	dfuncTMP->ggd					= true;
	dfuncTMP->reset();

	if(savePath.size() != 0){
		dfuncTMP->savePath = std::string(savePath)+"/segment_"+std::to_string(segment_run_counter)+"_dfunc.m";
	}

	dfunc->computeModel(dvec);

	GeneralizedGaussianDistribution * ggdnfunc	= new GeneralizedGaussianDistribution(true,true);
	ggdnfunc->nr_refineiters					= 4;
	DistanceWeightFunction2PPR3 * nfuncTMP		= new DistanceWeightFunction2PPR3(ggdnfunc);
	nfunc = nfuncTMP;
	nfuncTMP->startreg				= 0.00;
	nfuncTMP->debugg_print			= false;
	nfuncTMP->bidir					= false;
	nfuncTMP->zeromean				= true;
	nfuncTMP->maxp					= 0.9999;
	nfuncTMP->maxd					= 1.0;
	nfuncTMP->histogram_size		= 1000;
	nfuncTMP->fixed_histogram_size	= true;
	nfuncTMP->startmaxd				= nfuncTMP->maxd;
	nfuncTMP->starthistogram_size	= nfuncTMP->histogram_size;
	nfuncTMP->blurval				= 0.5;
	nfuncTMP->stdval2				= 1;
	nfuncTMP->maxnoise				= 1;
	nfuncTMP->ggd					= true;
	nfuncTMP->reset();
	nfunc->computeModel(nvec);

	if(savePath.size() != 0){
		nfuncTMP->savePath = std::string(savePath)+"/segment_"+std::to_string(segment_run_counter)+"_nfunc.m";
	}



	printf("training time:     %5.5fs\n",getTime()-startTime);

	long frameConnections = 0;
	std::vector< std::vector< std::vector<float> > > pixel_weights;

	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud  (new pcl::PointCloud<pcl::PointXYZRGBNormal>);

	double total_priortime = 0;
	double total_connectiontime = 0;
	double total_alloctime = 0;
	double total_dealloctime = 0;
	double total_Dynw = 0;

	double maxprob_same = 0.999999999999999999;

	for(unsigned int i = 0; i < frames.size(); i++){
		if(debugg != 0){printf("currently workin on frame %i\n",i);}
		int offset = offsets[i];
		RGBDFrame * frame = frames[i];
		float		   * normalsdata	= (float			*)(frame->normals.data);
		startTime = getTime();
		std::vector< std::vector<float> > probs = frame->getImageProbs();

		total_connectiontime += getTime()-startTime;
		pixel_weights.push_back(probs);
		unsigned short * depthdata	= (unsigned short	*)(frame->depth.data);

		Camera * camera				= frame->camera;
		const unsigned int width	= camera->width;
		const unsigned int height	= camera->height;
		const float idepth			= camera->idepth_scale;
		const float cx				= camera->cx;
		const float cy				= camera->cy;
		const float ifx				= 1.0/camera->fx;
		const float ify				= 1.0/camera->fy;

		startTime = getTime();
		unsigned int nr_pixels = width * height;
		double * current_overlaps	= new double[nr_pixels];
		double * current_occlusions		= new double[nr_pixels];
		double * current_notocclusions		= new double[nr_pixels];
		for(unsigned int j = 0; j < nr_pixels; j++){
			current_overlaps[j] = 0;
			current_occlusions[j] = 0.0;
			current_notocclusions[j] = 0.0;
		}

		double * bg_overlaps			= new double[nr_pixels];
		double * bg_occlusions			= new double[nr_pixels];
		double * bg_notocclusions		= new double[nr_pixels];
		for(unsigned int j = 0; j < nr_pixels; j++){
			bg_overlaps[j]	= 0;
			bg_occlusions[j] = 0.0;
			bg_notocclusions[j] = 0.0;
		}
		total_alloctime += getTime()-startTime;

		startTime = getTime();
		for(unsigned int j = 0; j < frames.size(); j++){
			if(i == j){continue;}
			Eigen::Matrix4d p = poses[i].inverse() * poses[j];
			getDynamicWeights(false,dvec,nvec,dfunc,nfunc,p.inverse(),frames[i],framesp_test[i],framesp[i], current_overlaps, current_occlusions, current_notocclusions, frames[j],framesp[j],offsets[i],offsets[j],interframe_connectionId,interframe_connectionStrength,false);
		}

		for(unsigned int j = 0; j < bgcf.size(); j++){
			Eigen::Matrix4d p = poses[i].inverse() * bgcp[j];
			getDynamicWeights(false,dvec,nvec,dfunc,nfunc,p.inverse(),frames[i],framesp_test[i],framesp[i], bg_overlaps, bg_occlusions, bg_notocclusions, bgcf[j],bgsp[j],-1,-1,interframe_connectionId,interframe_connectionStrength,false);
		}

		for(unsigned int j = 0; j < nr_pixels; j++){
			bg_occlusions[j]		= std::min(0.99999,bg_occlusions[j]/std::max(1.0,bg_notocclusions[j]));
			current_occlusions[j]	= std::min(0.99999,current_occlusions[j]/std::max(1.0,current_notocclusions[j]));
		}

		total_Dynw += getTime()-startTime;



		startTime = getTime();
		unsigned char * detdata = (unsigned char*)(frame->det_dilate.data);
		for(unsigned int h = 0; h < height;h++){
			for(unsigned int w = 0; w < width;w++){
				int ind = h*width+w;

				valids[offset+ind] = detdata[ind] == 0 && normalsdata[3*ind] != 2;
				setupPriors(3,
						current_occlusions[ind],current_overlaps[ind],bg_occlusions[ind],bg_overlaps[ind],valids[offset+ind],
						priors[3*(offset+ind)+0], priors[3*(offset+ind)+1],priors[3*(offset+ind)+2],
						prior_weights[2*(offset+ind)+0], prior_weights[2*(offset+ind)+1]);

				if(probs[0][ind] > 0.00000001){frameConnections++;}
				if(probs[1][ind] > 0.00000001){frameConnections++;}

				current_point++;
			}
		}

		double start_inf = getTime();
		gc::Graph<double,double,double> * g = new gc::Graph<double,double,double>(nr_pixels,2*nr_pixels);
		for(unsigned long ind = 0; ind < nr_pixels;ind++){
			g -> add_node();
			double weightFG = prior_weights[2*(offset+ind)+0];
			double weightBG = prior_weights[2*(offset+ind)+1];
			g -> add_tweights( ind, weightFG, weightBG );
		}

		for(unsigned int w = 0; w < width;w++){
			for(unsigned int h = 0; h < height;h++){
				int ind = h*width+w;
				if(w > 0 && probs[0][ind] > 0.00000001){
					int other = ind-1;
					double p_same = std::min(double(probs[0][ind]),maxprob_same);
					double not_p_same = 1-p_same;
					double weight = -log(not_p_same);
					g -> add_edge( ind, other, weight, weight );
				}

				if(h > 0 && probs[1][ind] > 0.00000001){
					int other = ind-width;
					double p_same = std::min(double(probs[1][ind]),maxprob_same);
					double not_p_same = 1-p_same;
					double weight = -log(not_p_same);
					g -> add_edge( ind, other, weight, weight );
				}
			}
		}

		g -> maxflow();
		for(unsigned long ind = 0; ind < nr_pixels;ind++){labels[offset+ind] = g->what_segment(ind);}

		if(debugg != 0){printf("local inference time: %10.10fs\n\n",getTime()-start_inf);}




		//		cv::Mat detrgb;
		//		detrgb.create(height,width,CV_8UC3);

		//		for(unsigned long ind = 0; ind < nr_pixels;ind++){
		//			int dd = detdata[ind] == 0;
		//			detrgb.data[3*ind+0] = dd * frame->rgb.data[3*ind+0];
		//			detrgb.data[3*ind+1] = dd * frame->rgb.data[3*ind+1];
		//			detrgb.data[3*ind+2] = dd * frame->rgb.data[3*ind+2];
		//		}

		//		cv::namedWindow( "edgeimg"		, cv::WINDOW_AUTOSIZE );		cv::imshow( "edgeimg",		edgeimg );
		//		cv::namedWindow( "det_dilate"	, cv::WINDOW_AUTOSIZE );		cv::imshow( "det_dilate",	frame->det_dilate);
		//		cv::namedWindow( "det_dilate2"	, cv::WINDOW_AUTOSIZE );		cv::imshow( "det_dilate2",	detrgb);
		//		cv::namedWindow( "rgb"			, cv::WINDOW_AUTOSIZE );		cv::imshow( "rgb",			frame->rgb );
		//		cv::namedWindow( "depth"		, cv::WINDOW_AUTOSIZE );		cv::imshow( "depth",		frame->depth );
		//		cv::namedWindow( "priors"		, cv::WINDOW_AUTOSIZE );		cv::imshow( "priors",		priorsimg );
		//		cv::namedWindow( "labelimg"		, cv::WINDOW_AUTOSIZE );		cv::imshow( "labelimg",		labelimg );
		//		cv::waitKey(0);


		if(savePath.size() != 0){
			cv::Mat edgeimg;
			edgeimg.create(height,width,CV_8UC3);
			unsigned char * edgedata = (unsigned char*)edgeimg.data;

			for(int j = 0; j < width*height; j++){
				edgedata[3*j+0] = 0;
				edgedata[3*j+1] = 255.0*(1-probs[0][j]);
				edgedata[3*j+2] = 255.0*(1-probs[1][j]);
			}

			cv::Mat labelimg;
			labelimg.create(height,width,CV_8UC3);
			unsigned char * labelimgdata = (unsigned char*)labelimg.data;
			for(unsigned long ind = 0; ind < nr_pixels;ind++){
				double label = g->what_segment(ind);
				labelimgdata[3*ind+0] = 255*label;
				labelimgdata[3*ind+1] = 255*label;
				labelimgdata[3*ind+2] = 255*label;
			}

			cv::Mat priorsimg;
			priorsimg.create(height,width,CV_8UC3);
			unsigned char * priorsdata = (unsigned char*)priorsimg.data;
			for(unsigned long ind = 0; ind < nr_pixels;ind++){
				priorsdata[3*ind+0]			= 255.0*priors[3*(offset+ind)+2];
				priorsdata[3*ind+1]			= 255.0*priors[3*(offset+ind)+1];
				priorsdata[3*ind+2]			= 255.0*priors[3*(offset+ind)+0];
			}

			cv::Mat current_overlapsimg;
			current_overlapsimg.create(height,width,CV_8UC3);
			unsigned char * current_overlapsdata = (unsigned char*)current_overlapsimg.data;
			for(unsigned long ind = 0; ind < nr_pixels;ind++){
				current_overlapsdata[3*ind+0]			= 255.0*current_overlaps[ind];
				current_overlapsdata[3*ind+1]			= 255.0*current_overlaps[ind];
				current_overlapsdata[3*ind+2]			= 255.0*current_overlaps[ind];
			}

			cv::Mat bg_overlapsimg;
			bg_overlapsimg.create(height,width,CV_8UC3);
			unsigned char * bg_overlapsdata = (unsigned char*)bg_overlapsimg.data;
			for(unsigned long ind = 0; ind < nr_pixels;ind++){
				bg_overlapsdata[3*ind+0]			= 255.0*bg_overlaps[ind];
				bg_overlapsdata[3*ind+1]			= 255.0*bg_overlaps[ind];
				bg_overlapsdata[3*ind+2]			= 255.0*bg_overlaps[ind];
			}

			cv::Mat current_occlusionsimg;
			current_occlusionsimg.create(height,width,CV_8UC3);
			unsigned char * current_occlusionsdata = (unsigned char*)current_occlusionsimg.data;
			for(unsigned long ind = 0; ind < nr_pixels;ind++){
				current_occlusionsdata[3*ind+0]			= 255.0*current_occlusions[ind];
				current_occlusionsdata[3*ind+1]			= 255.0*current_occlusions[ind];
				current_occlusionsdata[3*ind+2]			= 255.0*current_occlusions[ind];
			}

			cv::Mat bg_occlusionsimg;
			bg_occlusionsimg.create(height,width,CV_8UC3);
			unsigned char * bg_occlusionsdata = (unsigned char*)bg_occlusionsimg.data;
			for(unsigned long ind = 0; ind < nr_pixels;ind++){
				bg_occlusionsdata[3*ind+0]			= 255.0*bg_occlusions[ind];
				bg_occlusionsdata[3*ind+1]			= 255.0*bg_occlusions[ind];
				bg_occlusionsdata[3*ind+2]			= 255.0*bg_occlusions[ind];
			}

			cv::imwrite( savePath+"/segment_"+std::to_string(segment_run_counter)+"_frame_"+std::to_string(i)+"_current_overlapsimg.png", current_overlapsimg );
			cv::imwrite( savePath+"/segment_"+std::to_string(segment_run_counter)+"_frame_"+std::to_string(i)+"_bg_overlapsimg.png", bg_overlapsimg );
			cv::imwrite( savePath+"/segment_"+std::to_string(segment_run_counter)+"_frame_"+std::to_string(i)+"_current_occlusionsimg.png", current_occlusionsimg );
			cv::imwrite( savePath+"/segment_"+std::to_string(segment_run_counter)+"_frame_"+std::to_string(i)+"_bg_occlusionsimg.png", bg_occlusionsimg );
			cv::imwrite( savePath+"/segment_"+std::to_string(segment_run_counter)+"_frame_"+std::to_string(i)+"_edgeimg.png", edgeimg );
			cv::imwrite( savePath+"/segment_"+std::to_string(segment_run_counter)+"_frame_"+std::to_string(i)+"_rgb.png", frame->rgb  );
			cv::imwrite( savePath+"/segment_"+std::to_string(segment_run_counter)+"_frame_"+std::to_string(i)+"_priors.png", priorsimg );
			cv::imwrite( savePath+"/segment_"+std::to_string(segment_run_counter)+"_frame_"+std::to_string(i)+"_labelimg.png", labelimg );
		}
		delete g;

		Eigen::Matrix4d p = poses[i];
		float m00 = p(0,0); float m01 = p(0,1); float m02 = p(0,2); float m03 = p(0,3);
		float m10 = p(1,0); float m11 = p(1,1); float m12 = p(1,2); float m13 = p(1,3);
		float m20 = p(2,0); float m21 = p(2,1); float m22 = p(2,2); float m23 = p(2,3);
		for(unsigned int h = 0; h < height;h++){
			for(unsigned int w = 0; w < width;w++){
				int ind = h*width+w;
				float z = idepth*float(depthdata[ind]);
				float x = (float(w) - cx) * z * ifx;
				float y = (float(h) - cy) * z * ify;
				pcl::PointXYZRGBNormal point;
				point.x = m00*x + m01*y + m02*z + m03;
				point.y = m10*x + m11*y + m12*z + m13;
				point.z = m20*x + m21*y + m22*z + m23;
				point.r = frame->rgb.data[3*ind+2];//priors[3*(offset+ind)+0]*255.0;
				point.g = frame->rgb.data[3*ind+1];//priors[3*(offset+ind)+1]*255.0;
				point.b = frame->rgb.data[3*ind+0];//priors[3*(offset+ind)+2]*255.0;
				cloud->points.push_back(point);
			}
		}
		total_priortime += getTime()-startTime;

		startTime = getTime();
		delete[] current_occlusions;
		delete[] current_notocclusions;
		delete[] current_overlaps;
		delete[] bg_occlusions;
		delete[] bg_notocclusions;
		delete[] bg_overlaps;
		total_dealloctime += getTime()-startTime;
	}

	delete dfuncTMP;
	delete nfuncTMP;

	printf("total_priortime        = %5.5fs\n",		total_priortime);
	printf("total_connectiontime   = %5.5fs\n",		total_connectiontime);
	printf("total_alloctime        = %5.5fs\n",		total_alloctime);
	printf("total_dealloctime      = %5.5fs\n",		total_dealloctime);
	printf("total_Dynw             = %5.5fs\n",		total_Dynw);

	long interframeConnections = 0;
	for(unsigned int i = 0; i < interframe_connectionId.size();i++){interframeConnections += interframe_connectionId[i].size();}

	double start_inf = getTime();
	gc::Graph<double,double,double> * g = new gc::Graph<double,double,double>(current_point,frameConnections+interframeConnections);
	for(unsigned long i = 0; i < current_point;i++){
		g -> add_node();
		double weightFG = prior_weights[2*i+0];
		double weightBG = prior_weights[2*i+1];

		g -> add_tweights( i, weightFG, weightBG );
	}

	//double maxprob_same = 0.99999999999;
	for(unsigned int i = 0; i < frames.size(); i++){
		int offset = offsets[i];
		Camera * camera				= frames[i]->camera;
		const unsigned int width	= camera->width;
		const unsigned int height	= camera->height;
		std::vector< std::vector<float> > & probs = pixel_weights[i];
		for(unsigned int w = 0; w < width;w++){
			for(unsigned int h = 0; h < height;h++){
				int ind = h*width+w;
				if(w > 0 && probs[0][ind] > 0.00000001 && w < width-1){
					int other = ind-1;
					double p_same = std::min(double(probs[0][ind]),maxprob_same);
					double not_p_same = 1-p_same;
					double weight = -log(not_p_same);
					g -> add_edge( ind+offset, other+offset, weight, weight );
				}

				if(h > 0 && probs[1][ind] > 0.00000001 && h < height-1) {
					int other = ind-width;
					double p_same = std::min(double(probs[1][ind]),maxprob_same);
					double not_p_same = 1-p_same;
					double weight = -log(not_p_same);
					g -> add_edge( ind+offset, other+offset, weight, weight );
				}
			}
		}
	}

	std::vector<std::vector<unsigned long> > interframe_connectionId_added;
	interframe_connectionId_added.resize(interframe_connectionId.size());

	std::vector<std::vector<double> > interframe_connectionStrength_added;
	interframe_connectionStrength_added.resize(interframe_connectionStrength.size());

	double initAdded = 0;
	for(unsigned int i = 0; i < interframe_connectionId.size();i++){
		for(unsigned int j = 0; j < interframe_connectionId[i].size();j++){
			double weight = interframe_connectionStrength[i][j];
			unsigned long other = interframe_connectionId[i][j];
			if(weight > 0.01 && labels[i] != labels[other]){
				g -> add_edge( i, other, weight, weight );

				interframe_connectionId_added[i].push_back(other);
				interframe_connectionStrength_added[i].push_back(weight);

				interframe_connectionStrength[i][j] = interframe_connectionStrength[i].back();
				interframe_connectionStrength[i].pop_back();

				interframe_connectionId[i][j] = interframe_connectionId[i].back();
				interframe_connectionId[i].pop_back();
				j--;

				initAdded++;
			}
		}
	}

	g -> maxflow();
	for(unsigned long ind = 0; ind < current_point;ind++){labels[ind] = g->what_segment(ind);}

	double tot_inf = 0;
	for(unsigned int it = 0; it < 40; it++){
		double start_inf1 = getTime();

		double diffs = 0;
		for(unsigned int i = 0; i < interframe_connectionId.size();i++){
			for(unsigned int j = 0; j < interframe_connectionId[i].size();j++){
				double weight = interframe_connectionStrength[i][j];
				unsigned long other = interframe_connectionId[i][j];
				if(weight > 0.01 && labels[i] != labels[other]){diffs++;}
			}
		}


		double adds = 100000;
		double prob = std::min(adds / diffs,1.0);
		printf("diffs: %f adds: %f prob: %f ",diffs,adds,prob);
		printf("ratio of total diffs: %f\n",diffs/double(frameConnections+interframeConnections));

		if(diffs == 0){break;}

		double trueadds = 0;
		for(unsigned int i = 0; i < interframe_connectionId.size();i++){
			for(unsigned int j = 0; j < interframe_connectionId[i].size();j++){
				double weight = interframe_connectionStrength[i][j];
				unsigned long other = interframe_connectionId[i][j];
				if(weight > 0.01 && labels[i] != labels[other]){
					if(rand() <= prob*RAND_MAX){
						trueadds++;
						g -> add_edge( i, other, weight, weight );

						interframe_connectionId_added[i].push_back(other);
						interframe_connectionStrength_added[i].push_back(weight);

						interframe_connectionStrength[i][j] = interframe_connectionStrength[i].back();
						interframe_connectionStrength[i].pop_back();

						interframe_connectionId[i][j] = interframe_connectionId[i].back();
						interframe_connectionId[i].pop_back();
						j--;
					}
				}
			}
		}

		g -> maxflow();
		for(unsigned long ind = 0; ind < current_point;ind++){labels[ind] = g->what_segment(ind);}

		tot_inf += getTime()-start_inf1;
		if(debugg != 0){printf("static inference1 time: %10.10fs total: %10.10f\n\n",getTime()-start_inf1,tot_inf);}

		if(tot_inf > 90){break;}
	}

	double interfrace_constraints_added = 0;
	for(unsigned int i = 0; i < interframe_connectionId.size();i++){
		interfrace_constraints_added += interframe_connectionId_added[i].size();
		for(unsigned int j = 0; j < interframe_connectionId_added[i].size();j++){
			interframe_connectionStrength[i].push_back(interframe_connectionStrength_added[i][j]);
			interframe_connectionId[i].push_back(interframe_connectionId_added[i][j]);
		}
	}

	delete g;
	double end_inf = getTime();
	printf("static inference time: %10.10fs interfrace_constraints added ratio: %f\n",end_inf-start_inf,interfrace_constraints_added/double(interframeConnections));

	const unsigned int nr_frames		= frames.size();
	const unsigned int width			= frames[0]->camera->width;
	const unsigned int height			= frames[0]->camera->height;
	const unsigned int pixels_per_image	= width*height;
	const unsigned int nr_pixels		= nr_frames*pixels_per_image;

	double probthresh = 0.5;
	double str_probthresh = -log(probthresh);
	unsigned int number_of_dynamics = 0;
	unsigned int nr_obj_dyn = 0;
	unsigned int nr_obj_mov = 0;
	std::vector<unsigned int> objectlabel;
	std::vector<int> labelID;
	labelID.push_back(0);
	objectlabel.resize(nr_pixels);

	for(unsigned long i = 0; i < nr_pixels; i++){objectlabel[i] = 0;}
	for(unsigned long ind = 0; ind < nr_pixels; ind++){
		if(valids[ind] && objectlabel[ind] == 0 && labels[ind] != 0){
			unsigned int current_label = labels[ind];
			number_of_dynamics++;
			objectlabel[ind] = number_of_dynamics;
			unsigned long todocounter = 0;
			std::vector< unsigned long > todo;
			todo.push_back(ind);

			double score0 = 0;
			double score1 = 0;

			double pscore0 = 0;
			double pscore1 = 0;

			double nscore0 = 0;
			double nscore1 = 0;

			double totsum = 0;
			while(todocounter < todo.size()){
				unsigned long cind = todo[todocounter++];
				unsigned long frameind = cind / pixels_per_image;


				unsigned long iind = cind % pixels_per_image;
				unsigned long w = iind % width;
				unsigned long h = iind / width;

				double p0 = priors[3*cind+0];
				double p1 = priors[3*cind+1];
				double p2 = priors[3*cind+2];

				if(valids[cind]){
					double s0 = 0;
					if(p1 > p2){s0 += p0 - p1;}
					else{       s0 += p0 - p2;}
					score0 += s0;

					if(s0 > 0){	pscore0 += s0;}
					else{		nscore0 += s0;}

					double s1 = 0;
					if(p0 > p2){s1 += p1 - p0;}
					else{       s1 += p1 - p2;}
					score1 += s1;

					if(s1 > 0){	pscore1 += s1;}
					else{		nscore1 += s1;}
					totsum++;
				}

				float * dedata = (float*)(frames[frameind]->de.data);
				unsigned short * depthdata = (unsigned short *)(frames[frameind]->depth.data);
				if(depthdata[iind] == 0){printf("big giant WTF... file %i line %i\n",__FILE__,__LINE__);continue;}

				int dir;
				dir = -1;
				if( w > 0 && objectlabel[cind+dir] == 0 && labels[cind+dir] == current_label && depthdata[iind+dir] != 0 && (dedata[3*(iind+dir)+1]+dedata[3*(iind+dir)+2]) < probthresh){
					objectlabel[cind+dir] = number_of_dynamics;
					todo.push_back(cind+dir);
				}

				dir = 1;
				if( w < (width-1) && objectlabel[cind+dir] == 0 && labels[cind+dir] == current_label && depthdata[iind+dir] != 0 && (dedata[3*(iind+dir)+1]+dedata[3*(iind+dir)+2]) < probthresh){
					objectlabel[cind+dir] = number_of_dynamics;
					todo.push_back(cind+dir);
				}

				dir = -width;
				if( h > 0 && objectlabel[cind+dir] == 0 && labels[cind+dir] == current_label && depthdata[iind+dir] != 0 && (dedata[3*(iind+dir)+1]+dedata[3*(iind+dir)+2]) < probthresh){
					objectlabel[cind+dir] = number_of_dynamics;
					todo.push_back(cind+dir);
				}

				dir = width;
				if( h < (height-1) && objectlabel[cind+dir] == 0 && labels[cind+dir] == current_label && depthdata[iind+dir] != 0 && (dedata[3*(iind+dir)+1]+dedata[3*(iind+dir)+2]) < probthresh){
					objectlabel[cind+dir] = number_of_dynamics;
					todo.push_back(cind+dir);
				}

				for(unsigned long j = 0; j < interframe_connectionId[cind].size();j++){
					unsigned long other = interframe_connectionId[cind][j];
					if(interframe_connectionStrength[cind][j] > str_probthresh && objectlabel[other] == 0 && labels[other] == current_label){
						objectlabel[other] = number_of_dynamics;
						todo.push_back(other);
					}
				}
			}

			score0 = 100.0*pscore0+nscore0;
			score1 = pscore1+nscore1;


			labelID.push_back(0);
			if(debugg != 0){
				if(totsum > 100){
					printf("---------------------------\n");
					printf("score0: %10.10f score1: %10.10f ",score0,score1);
					printf("totsum: %10.10f\n",totsum);

					printf("pscore0: %10.10f nscore0: %10.10f ",pscore0,nscore0);
					printf("pscore1: %10.10f nscore1: %10.10f\n",pscore1,nscore1);
				}
			}

			if(std::max(score0,score1) < 100){continue;}

			if(score1 > score0){
				labelID.back() = ++nr_obj_dyn;
				if(debugg != 0){printf("Dynamic: %f -> %f\n",score1,totsum);}
				sr.component_dynamic.push_back(todo);
				sr.scores_dynamic.push_back(score1);
				sr.total_dynamic.push_back(totsum);
			}else{
				if(score0 > 10000){
					labelID.back() = --nr_obj_mov;
					if(debugg != 0){printf("Moving: %f -> %f\n",score0,totsum);}
					sr.component_moving.push_back(todo);
					sr.scores_moving.push_back(score0);
					sr.total_moving.push_back(totsum);
				}
			}
		}
	}

	for(unsigned long ind = 0; ind < nr_pixels; ind++){
		unsigned int ol = objectlabel[ind];
		if(ol != 0 && labelID[ol] == 0){
			objectlabel[ind] = 0;
			labels[ind] = 0;
		}
	}

	for(unsigned int i = 0; i < interframe_connectionId.size();i++){interframe_connectionId[i].clear();}
	interframe_connectionId.clear();

	for(unsigned int i = 0; i < interframe_connectionStrength.size();i++){interframe_connectionStrength[i].clear();}
	interframe_connectionStrength.clear();

	printf("connectedComponent: %5.5fs\n",getTime()-start_inf);

	int current = 0;
	for(unsigned long i = 0; i < frames.size(); i++){
		Camera * camera				= frames[i]->camera;
		const unsigned int width	= camera->width;
		const unsigned int height	= camera->height;

		cv::Mat m;
		m.create(height,width,CV_8UC1);
		unsigned char * mdata = (unsigned char*)m.data;

		cv::Mat d;
		d.create(height,width,CV_8UC1);
		unsigned char * ddata = (unsigned char*)d.data;

		cv::Mat d2;
		d2.create(height,width,CV_8UC1);
		unsigned char * ddata2 = (unsigned char*)d2.data;


		for(int j = 0; j < width*height; j++){
			mdata[j] = 0;
			ddata[j] = 0;
			ddata2[j] = labels[current];
			unsigned int label = objectlabel[current];
			int lid = labelID[label];
			if(lid >  0){
				ddata[j] = lid;
			}else if(lid < 0){
				mdata[j] = -lid;
			}
			current++;
		}
		movemask.push_back(m);
		dynmask.push_back(d);
	}

	if(savePath.size() != 0){
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_sample (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
		cloud_sample->points.resize(current_point);
		cloud_sample->width = current_point;
		cloud_sample->height = 1;
		for(unsigned int i = 0; i < current_point; i++){
			cloud_sample->points[i]	= cloud->points[i];
			cloud_sample->points[i].r = priors[3*i+0]*255.0;
			cloud_sample->points[i].g = priors[3*i+1]*255.0;
			cloud_sample->points[i].b = priors[3*i+2]*255.0;
		}
		pcl::io::savePCDFileBinaryCompressed (savePath+"/segment_"+std::to_string(segment_run_counter)+"_priors.pcd", *cloud_sample);

		for(unsigned int i = 0; i < current_point; i++){
			cloud_sample->points[i].r = 0;
			cloud_sample->points[i].g = 0;
			cloud_sample->points[i].b = 255;
		}

		for(unsigned int c = 0; c < sr.component_dynamic.size(); c++){
			for(unsigned int i = 0; i < sr.component_dynamic[c].size(); i++){
				cloud_sample->points[sr.component_dynamic[c][i]].r = 0;
				cloud_sample->points[sr.component_dynamic[c][i]].g = 255;
				cloud_sample->points[sr.component_dynamic[c][i]].b = 0;
			}
		}

		for(unsigned int c = 0; c < sr.component_moving.size(); c++){
			for(unsigned int i = 0; i < sr.component_moving[c].size(); i++){
				cloud_sample->points[sr.component_moving[c][i]].r = 255;
				cloud_sample->points[sr.component_moving[c][i]].g = 0;
				cloud_sample->points[sr.component_moving[c][i]].b = 0;
			}
		}
		pcl::io::savePCDFileBinaryCompressed (savePath+"/segment_"+std::to_string(segment_run_counter)+"_classes.pcd", *cloud_sample);

		for(unsigned int i = 0; i < current_point; i++){
			cloud_sample->points[i].r = 0;
			cloud_sample->points[i].g = 0;
			cloud_sample->points[i].b = 255;
		}

		for(unsigned int c = 0; c < sr.component_dynamic.size(); c++){
			int randr = rand()%256;
			int randg = rand()%256;
			int randb = rand()%256;

			for(unsigned int i = 0; i < sr.component_dynamic[c].size(); i++){
				cloud_sample->points[sr.component_dynamic[c][i]].r = randr;
				cloud_sample->points[sr.component_dynamic[c][i]].g = randg;
				cloud_sample->points[sr.component_dynamic[c][i]].b = randb;
			}
		}

		for(unsigned int c = 0; c < sr.component_moving.size(); c++){
			int randr = rand()%256;
			int randg = rand()%256;
			int randb = rand()%256;

			for(unsigned int i = 0; i < sr.component_moving[c].size(); i++){
				cloud_sample->points[sr.component_moving[c][i]].r = randr;
				cloud_sample->points[sr.component_moving[c][i]].g = randg;
				cloud_sample->points[sr.component_moving[c][i]].b = randb;
			}
		}
		pcl::io::savePCDFileBinaryCompressed (savePath+"/segment_"+std::to_string(segment_run_counter)+"_clusters.pcd", *cloud_sample);

		cloud->width = cloud->points.size();
		cloud->height = 1;
		pcl::io::savePCDFileBinaryCompressed (savePath+"/segment_"+std::to_string(segment_run_counter)+"_full.pcd", *cloud);

		cloud_sample->points.resize(0);
		for(unsigned int c = 0; c < sr.component_dynamic.size(); c++){
			for(unsigned int i = 0; i < sr.component_dynamic[c].size(); i++){
				cloud_sample->points.push_back(cloud->points[sr.component_dynamic[c][i]]);
			}
		}
		cloud_sample->width = cloud_sample->points.size();
		cloud_sample->height = 1;
		pcl::io::savePCDFileBinaryCompressed (savePath+"/segment_"+std::to_string(segment_run_counter)+"_dynamicobjects.pcd", *cloud_sample);
	}


	printf("computeMovingDynamicStatic total time: %5.5fs\n",getTime()-computeMovingDynamicStatic_startTime);
	if(debugg){
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_sample (new pcl::PointCloud<pcl::PointXYZRGBNormal>);

		for(unsigned int i = 0; i < current_point; i++){
			cloud->points[i].r = priors[3*i+0]*255.0;
			cloud->points[i].g = priors[3*i+1]*255.0;
			cloud->points[i].b = priors[3*i+2]*255.0;
		}

		cloud_sample->points.clear();
		for(unsigned int i = 0; i < current_point; i++){
			if(rand() % 4 == 0){
				cloud_sample->points.push_back(cloud->points[i]);
			}
		}
		viewer->removeAllPointClouds();
		viewer->addPointCloud<pcl::PointXYZRGBNormal> (cloud_sample, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(cloud_sample), "cloud");
		viewer->spin();

		cloud_sample->points.clear();
		for(unsigned int i = 0; i < current_point; i++){
			if(rand() % 4 == 0){

				double p_fg = exp(-prior_weights[2*i+0]);

				cloud_sample->points.push_back(cloud->points[i]);
				cloud_sample->points.back().r = p_fg*255.0;//(priors[3*i+0]+priors[3*i+2])*255.0;
				cloud_sample->points.back().g = p_fg*255.0;
				cloud_sample->points.back().b = p_fg*255.0;//;
			}
		}
		viewer->removeAllPointClouds();
		viewer->addPointCloud<pcl::PointXYZRGBNormal> (cloud_sample, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(cloud_sample), "cloud");
		viewer->spin();


		cloud_sample->points.clear();
		for(unsigned int i = 0; i < current_point; i++){
			if(rand() % 4 == 0 && labels[i] == 0){
				cloud_sample->points.push_back(cloud->points[i]);
				cloud_sample->points.back().r = 0;
				cloud_sample->points.back().g = 0;
				cloud_sample->points.back().b = 255;
			}
		}

		for(unsigned int c = 0; c < sr.component_dynamic.size(); c++){
			int randr = rand()%256;
			int randg = rand()%256;
			int randb = rand()%256;
			for(unsigned int i = 0; i < sr.component_dynamic[c].size(); i++){
				cloud_sample->points.push_back(cloud->points[sr.component_dynamic[c][i]]);
				cloud_sample->points.back().r = 0;//randr;
				cloud_sample->points.back().g = 255;//randg;
				cloud_sample->points.back().b = 0;//randb;
			}
		}

		for(unsigned int c = 0; c < sr.component_moving.size(); c++){
			int randr = rand()%256;
			int randg = rand()%256;
			int randb = rand()%256;
			for(unsigned int i = 0; i < sr.component_moving[c].size(); i++){
				cloud_sample->points.push_back(cloud->points[sr.component_moving[c][i]]);
				cloud_sample->points.back().r = 255;//randr;
				cloud_sample->points.back().g = 0;//randg;
				cloud_sample->points.back().b = 0;//randb;
			}
		}
		viewer->removeAllPointClouds();
		viewer->addPointCloud<pcl::PointXYZRGBNormal> (cloud_sample, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(cloud_sample), "cloud");
		viewer->spin();

		cloud_sample->points.clear();
		for(unsigned int i = 0; i < current_point; i++){
			if(rand() % 1 == 0){
				cloud_sample->points.push_back(cloud->points[i]);
			}
		}
		viewer->removeAllPointClouds();
		viewer->addPointCloud<pcl::PointXYZRGBNormal> (cloud_sample, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(cloud_sample), "cloud");
		viewer->spin();
	}

	delete[] valids;
	delete[] priors;
	delete[] prior_weights;
	//printf("computeMovingDynamicStatic total time: %5.5fs\n",getTime()-computeMovingDynamicStatic_startTime);
}

/*
void ModelUpdater::computeMovingDynamicStatic(std::vector<cv::Mat> & movemask, std::vector<cv::Mat> & dynmask, vector<Matrix4d> bgcp, vector<RGBDFrame*> bgcf, vector<Matrix4d> poses, vector<RGBDFrame*> frames, bool debugg){
	double computeMovingDynamicStatic_startTime = getTime();

	SegmentationResults sr;

	int tot_nr_pixels = 0;
	std::vector<int> offsets;

	for(unsigned int i = 0; i < frames.size(); i++){
		offsets.push_back(tot_nr_pixels);
		unsigned int nr_pixels = frames[i]->camera->width * frames[i]->camera->height;
		tot_nr_pixels += nr_pixels;
	}

	//Graph...
	std::vector< std::vector<int> > interframe_connectionId;
	std::vector< std::vector<float> > interframe_connectionStrength;
	interframe_connectionId.resize(tot_nr_pixels);
	interframe_connectionStrength.resize(tot_nr_pixels);


	int current_point           = 0;
	float * priors             = new float[3*tot_nr_pixels];
	bool * valids             = new bool[tot_nr_pixels];

	std::vector<double> dvec;
	std::vector<double> nvec;
	DistanceWeightFunction2 * dfunc;
	DistanceWeightFunction2 * nfunc;

	double startTime = getTime();

	std::vector< std::vector<superpoint> > framesp_test;
	std::vector< std::vector<superpoint> > framesp;
	for(unsigned int i = 0; i < frames.size(); i++){
		framesp_test.push_back(frames[i]->getSuperPoints(Eigen::Matrix4d::Identity(),10,false));
		framesp.push_back(frames[i]->getSuperPoints());
	}
	printf("frames init time: %5.5fs\n",getTime()-startTime);

	startTime = getTime();
	std::vector< std::vector<superpoint> > bgsp;
	for(unsigned int i = 0; i < bgcf.size(); i++){
		bgsp.push_back(bgcf[i]->getSuperPoints());
	}
	printf("bg init time:     %5.5fs\n",getTime()-startTime);

	startTime = getTime();
	//	printf("computing all residuals\n");
	for(unsigned int i = 0; i < frames.size(); i++){
		std::vector<superpoint> & framesp1_test = framesp_test[i];
		std::vector<superpoint> & framesp1		= framesp[i];
		for(unsigned int j = 0; j < frames.size(); j++){
			if(i == j){continue;}
			Eigen::Matrix4d p = poses[i].inverse() * poses[j];
			std::vector<superpoint> & framesp2 = framesp[j];
			getDynamicWeights(true,dvec,nvec,dfunc,nfunc,p.inverse(),frames[i],framesp1_test,framesp1, 0, 0, frames[j],framesp2,offsets[i],offsets[j],interframe_connectionId,interframe_connectionStrength,false);
		}
	}


	//	printf("training ppr\n");
	double dstdval = 0;
	for(unsigned int i = 0; i < dvec.size(); i++){dstdval += dvec[i]*dvec[i];}
	dstdval = sqrt(dstdval/double(dvec.size()-1));

	DistanceWeightFunction2PPR2 * dfuncTMP = new DistanceWeightFunction2PPR2();
	dfunc = dfuncTMP;
	dfuncTMP->refine_mean			= true;
	dfuncTMP->zeromean				= false;
	dfuncTMP->startreg				= 0.0;
	//dfuncTMP->startreg				= 0.001;
	//dfuncTMP->startreg				= 0.0000001;
	dfuncTMP->debugg_print			= false;
	dfuncTMP->bidir					= true;
	//dfuncTMP->bidir					= false;
	//dfuncTMP->maxp					= 0.9;
	dfuncTMP->maxp					= 0.9;
	dfuncTMP->maxd					= 0.5;
	dfuncTMP->histogram_size		= 1000;
	dfuncTMP->fixed_histogram_size	= false;//true;
	dfuncTMP->max_under_mean		= false;
	dfuncTMP->startmaxd				= dfuncTMP->maxd;
	dfuncTMP->starthistogram_size	= dfuncTMP->histogram_size;
	dfuncTMP->blurval				= 0.5;
	dfuncTMP->stdval2				= dstdval;
	dfuncTMP->maxnoise				= dstdval;
	dfuncTMP->compute_infront		= true;
	dfuncTMP->reset();
	dfunc->computeModel(dvec);
	double dthreshold = dfunc->getNoise();

	GeneralizedGaussianDistribution * ggdnfunc	= new GeneralizedGaussianDistribution(true,true);
	ggdnfunc->nr_refineiters					= 4;
	DistanceWeightFunction2PPR3 * nfuncTMP		= new DistanceWeightFunction2PPR3(ggdnfunc);
	nfunc = nfuncTMP;
	nfuncTMP->startreg				= 0.00;
	nfuncTMP->debugg_print			= false;
	nfuncTMP->bidir					= false;
	nfuncTMP->zeromean				= true;
	nfuncTMP->maxp					= 0.999;
	nfuncTMP->maxd					= 1.0;
	nfuncTMP->histogram_size		= 1000;
	nfuncTMP->fixed_histogram_size	= true;
	nfuncTMP->startmaxd				= nfuncTMP->maxd;
	nfuncTMP->starthistogram_size	= nfuncTMP->histogram_size;
	nfuncTMP->blurval				= 0.5;
	nfuncTMP->stdval2				= 1;
	nfuncTMP->maxnoise				= 1;
	nfuncTMP->ggd					= true;
	nfuncTMP->reset();
	nfunc->computeModel(nvec);

	printf("training time:     %5.5fs\n",getTime()-startTime);

	long frameConnections = 0;
	std::vector< std::vector< std::vector<float> > > pixel_weights;

	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud  (new pcl::PointCloud<pcl::PointXYZRGBNormal>);

	double total_priortime = 0;
	double total_connectiontime = 0;
	double total_alloctime = 0;
	double total_dealloctime = 0;
	double total_Dynw = 0;

	for(unsigned int i = 0; i < frames.size(); i++){
		printf("currently workin on frame %i\n",i);
		int offset = offsets[i];
		RGBDFrame * frame = frames[i];
		startTime = getTime();
		std::vector< std::vector<float> > probs = getImageProbs(frame,9);
		total_connectiontime += getTime()-startTime;
		pixel_weights.push_back(probs);
		unsigned short * depthdata	= (unsigned short	*)(frame->depth.data);

		Camera * camera				= frame->camera;
		const unsigned int width	= camera->width;
		const unsigned int height	= camera->height;
		const float idepth			= camera->idepth_scale;
		const float cx				= camera->cx;
		const float cy				= camera->cy;
		const float ifx				= 1.0/camera->fx;
		const float ify				= 1.0/camera->fy;

		startTime = getTime();
		unsigned int nr_pixels = width * height;
		double * current_overlaps	= new double[nr_pixels];
		double * current_occlusions		= new double[nr_pixels];
		for(unsigned int j = 0; j < nr_pixels; j++){
			current_overlaps[j] = 0;
			current_occlusions[j] = 0;
		}

		double * bg_overlaps	= new double[nr_pixels];
		double * bg_occlusions		= new double[nr_pixels];
		for(unsigned int j = 0; j < nr_pixels; j++){
			bg_overlaps[j] = 0;
			bg_occlusions[j] = 0;
		}
		total_alloctime += getTime()-startTime;

		startTime = getTime();
		for(unsigned int j = 0; j < frames.size(); j++){
			if(i == j){continue;}
			Eigen::Matrix4d p = poses[i].inverse() * poses[j];
			getDynamicWeights(false,dvec,nvec,dfunc,nfunc,p.inverse(),frames[i],framesp_test[i],framesp[i], current_overlaps, current_occlusions, frames[j],framesp[j],offsets[i],offsets[j],interframe_connectionId,interframe_connectionStrength,false);
		}

		for(unsigned int j = 0; j < bgcf.size(); j++){
			Eigen::Matrix4d p = poses[i].inverse() * bgcp[j];
			getDynamicWeights(false,dvec,nvec,dfunc,nfunc,p.inverse(),frames[i],framesp_test[i],framesp[i], bg_overlaps, bg_occlusions, bgcf[j],bgsp[j],-1,-1,interframe_connectionId,interframe_connectionStrength,false);
		}
		total_Dynw += getTime()-startTime;


		startTime = getTime();
		unsigned char * detdata = (unsigned char*)(frame->det_dilate.data);
		float minprob = 0.01;
		for(unsigned int h = 0; h < height;h++){
			for(unsigned int w = 0; w < width;w++){
				int ind = h*width+w;

				valids[offset+ind] = detdata[ind] == 0;

				float p_moving  = current_occlusions[ind];
				float p_dynamic  =     std::max((bg_occlusions[ind]-p_moving)*current_overlaps[ind],0.0);
				p_dynamic       += 0.5*std::max((bg_occlusions[ind]-p_moving)*(1-current_overlaps[ind]),0.0);
				p_moving        += 0.5*std::max((bg_occlusions[ind]-p_moving)*(1-current_overlaps[ind]),0.0);

				float p_static  = std::max(bg_overlaps[ind]-p_moving-p_dynamic,0.0);

				float leftover = 1-p_moving-p_dynamic-p_static;
				float notMoving = current_overlaps[ind];

				float bias = 0.0001;

				float p_moving_leftover   = -bias+(1-notMoving)*leftover/4.0;
				float p_dynamic_leftover  = -bias+0.5*leftover*notMoving + (1-notMoving)*leftover/4.0;
				float p_static_leftover   = 2.0*bias+0.5*leftover*notMoving + (1-notMoving)*leftover/2.0;

				float p_moving_tot	= (1.0-4.0*minprob)*(p_moving+p_moving_leftover)	+ minprob;
				float p_dynamic_tot = (1.0-4.0*minprob)*(p_dynamic+p_dynamic_leftover)	+ minprob;
				float p_static_tot	= (1.0-4.0*minprob)*(p_static+p_static_leftover)	+ 2.0*minprob;

				priors[3*(offset+ind)+0]       = p_moving_tot;
				priors[3*(offset+ind)+1]       = p_dynamic_tot;
				priors[3*(offset+ind)+2]       = p_static_tot;

				if(probs[0][ind] > 0.00000001){frameConnections++;}
				if(probs[1][ind] > 0.00000001){frameConnections++;}

				current_point++;
			}
		}

		Eigen::Matrix4d p = poses[i];
		float m00 = p(0,0); float m01 = p(0,1); float m02 = p(0,2); float m03 = p(0,3);
		float m10 = p(1,0); float m11 = p(1,1); float m12 = p(1,2); float m13 = p(1,3);
		float m20 = p(2,0); float m21 = p(2,1); float m22 = p(2,2); float m23 = p(2,3);
		for(unsigned int h = 0; h < height;h++){
			for(unsigned int w = 0; w < width;w++){
				int ind = h*width+w;
				float z = idepth*float(depthdata[ind]);
				float x = (float(w) - cx) * z * ifx;
				float y = (float(h) - cy) * z * ify;
				pcl::PointXYZRGBNormal point;
				point.x = m00*x + m01*y + m02*z + m03;
				point.y = m10*x + m11*y + m12*z + m13;
				point.z = m20*x + m21*y + m22*z + m23;
				point.r = priors[3*(offset+ind)+0]*255.0;
				point.g = priors[3*(offset+ind)+1]*255.0;
				point.b = priors[3*(offset+ind)+2]*255.0;
				cloud->points.push_back(point);
			}
		}
		total_priortime += getTime()-startTime;

		startTime = getTime();
		delete[] current_occlusions;
		delete[] current_overlaps;
		delete[] bg_occlusions;
		delete[] bg_overlaps;
		total_dealloctime += getTime()-startTime;
	}

	delete dfuncTMP;
	delete nfuncTMP;

	printf("total_priortime        = %5.5fs\n",		total_priortime);
	printf("total_connectiontime   = %5.5fs\n",		total_connectiontime);
	printf("total_alloctime        = %5.5fs\n",		total_alloctime);
	printf("total_dealloctime      = %5.5fs\n",		total_dealloctime);
	printf("total_Dynw             = %5.5fs\n",		total_Dynw);

	long interframeConnections = 0;
	for(unsigned int i = 0; i < interframe_connectionId.size();i++){
		for(unsigned int j = 0; j < interframe_connectionId[i].size();j++){
			interframeConnections++;
		}
	}

	double start_inf = getTime();
	gc::Graph<double,double,double> * g = new gc::Graph<double,double,double>(current_point,frameConnections+interframeConnections);
	for(unsigned long i = 0; i < current_point;i++){
		g -> add_node();
		double p_fg = priors[3*i+0]+priors[3*i+1];
		double p_bg = priors[3*i+2];
		double norm = p_fg + p_bg;
		p_fg /= norm;
		p_bg /= norm;
		if(priors[3*i+0]+priors[3*i+1]+priors[3*i+2] <= 0){
			g -> add_tweights( i, 0, 0 );
			continue;
		}

		double weightFG = -log(p_fg);
		double weightBG = -log(p_bg);
		g -> add_tweights( i, weightFG, weightBG );
	}

	double maxprob_same = 0.99999999999;
	for(unsigned int i = 0; i < frames.size(); i++){
		int offset = offsets[i];
		Camera * camera				= frames[i]->camera;
		const unsigned int width	= camera->width;
		const unsigned int height	= camera->height;
		std::vector< std::vector<float> > & probs = pixel_weights[i];
		for(unsigned int w = 0; w < width;w++){
			for(unsigned int h = 0; h < height;h++){
				int ind = h*width+w;
				if(w > 0 && probs[0][ind] > 0.00000001){
					int other = ind-1;
					double p_same = std::min(double(probs[0][ind]),maxprob_same);
					double not_p_same = 1-p_same;
					double weight = -log(not_p_same);
					g -> add_edge( ind+offset, other+offset, weight, weight );
				}

				if(h > 0 && probs[1][ind] > 0.00000001){
					int other = ind-width;
					double p_same = std::min(double(probs[1][ind]),maxprob_same);
					double not_p_same = 1-p_same;
					double weight = -log(not_p_same);
					g -> add_edge( ind+offset, other+offset, weight, weight );
				}
			}
		}
	}
	printf("run inference\n");

	std::vector<std::vector<unsigned long> > interframe_connectionId_added;
	interframe_connectionId_added.resize(interframe_connectionId.size());

	std::vector<std::vector<double> > interframe_connectionStrength_added;
	interframe_connectionStrength_added.resize(interframe_connectionStrength.size());

	double tot_inf = 0;
	for(unsigned int it = 0; it < 10; it++){
		double start_inf1 = getTime();

		g -> maxflow();

		double diffs = 0;
		for(unsigned int i = 0; i < interframe_connectionId.size();i++){
			for(unsigned int j = 0; j < interframe_connectionId[i].size();j++){
				double weight = interframe_connectionStrength[i][j];
				unsigned long other = interframe_connectionId[i][j];
				if(weight > 1 && g->what_segment(i) != g->what_segment(other)){
					diffs++;
				}
			}
		}

		if(diffs < 1000){break;}

		double adds = 100000;
		double prob = std::min(adds / diffs,1.0);
		printf("diffs: %f adds: %f prob: %f\n",diffs,adds,prob);

		double trueadds = 0;

		for(unsigned int i = 0; i < interframe_connectionId.size();i++){
			for(unsigned int j = 0; j < interframe_connectionId[i].size();j++){
				double weight = interframe_connectionStrength[i][j];
				unsigned long other = interframe_connectionId[i][j];
				if(weight > 0.1 && g->what_segment(i) != g->what_segment(other)){
					if(rand() <= prob*RAND_MAX){
						trueadds++;
						g -> add_edge( i, other, weight, weight );

						interframe_connectionId_added[i].push_back(other);
						interframe_connectionStrength_added[i].push_back(weight);

						interframe_connectionStrength[i][j] = interframe_connectionStrength[i].back();
						interframe_connectionStrength[i].pop_back();

						interframe_connectionId[i][j] = interframe_connectionId[i].back();
						interframe_connectionId[i].pop_back();
						j--;
					}
				}
			}
		}
		printf("trueadds: %f\n",trueadds);

		tot_inf += getTime()-start_inf1;
		printf("static inference1 time: %10.10fs total: %10.10f\n\n",getTime()-start_inf1,tot_inf);

		if(tot_inf > 90){break;}
	}

	for(unsigned int i = 0; i < interframe_connectionId.size();i++){
		for(unsigned int j = 0; j < interframe_connectionId_added[i].size();j++){
			interframe_connectionStrength[i].push_back(interframe_connectionStrength_added[i][j]);
			interframe_connectionId[i].push_back(interframe_connectionId_added[i][j]);
		}
	}

	int dynamic_label = 1;
	std::vector<unsigned char> labels;
	std::vector<int> dyn_ind;
	std::vector<int> stat_ind;
	labels.resize(current_point);
	stat_ind.resize(current_point);
	long nr_dynamic = 0;
	for(unsigned long i = 0; i < current_point;i++){
		labels[i] = g->what_segment(i);
		nr_dynamic += labels[i]==dynamic_label;
		if(labels[i]==dynamic_label){
			stat_ind[i] = dyn_ind.size();
			dyn_ind.push_back(i);
		}
	}
	delete g;

	double end_inf = getTime();
	printf("static inference time: %10.10fs\n",end_inf-start_inf);

	long dynamic_frameConnections = 0;

	for(unsigned long i = 0; i < frames.size(); i++){
		int offset = offsets[i];
		Camera * camera				= frames[i]->camera;
		const unsigned int width	= camera->width;
		const unsigned int height	= camera->height;
		std::vector< std::vector<float> > & probs = pixel_weights[i];
		for(unsigned int w = 0; w < width;w++){
			for(unsigned int h = 0; h < height;h++){
				int ind = h*width+w;
				if(labels[ind+offset] == dynamic_label){
					if(w > 0 && probs[0][ind] > 0.00000001){
						int other = ind-1;
						if(labels[ind+offset] == labels[other+offset]){dynamic_frameConnections++;}
					}

					if(h > 0 && probs[1][ind] > 0.00000001){
						int other = ind-width;
						if(labels[ind+offset] == labels[other+offset]){dynamic_frameConnections++;}
					}
				}
			}
		}
	}

	long dynamic_interframeConnections = 0;
	for(unsigned long i = 0; i < current_point;i++){
		if(labels[i] == dynamic_label){
			for(unsigned int j = 0; j < interframe_connectionId[i].size();j++){
				if(labels[interframe_connectionId[i][j]] == labels[i]){dynamic_interframeConnections++;}
			}
		}
	}

	start_inf = getTime();
	gc::Graph<double,double,double> * dynamic_g = new gc::Graph<double,double,double>(nr_dynamic,dynamic_frameConnections+dynamic_interframeConnections);
	for(unsigned long i = 0; i < dyn_ind.size();i++){
		dynamic_g -> add_node();
		double p_fg = priors[3*dyn_ind[i]+0];
		double p_bg = priors[3*dyn_ind[i]+1];
		double norm = p_fg+p_bg;
		p_fg /= norm;
		p_bg /= norm;
		if(norm <= 0){
			dynamic_g -> add_tweights( i, 0, 0 );
			continue;
		}

		double weightFG = -log(p_fg);
		double weightBG = -log(p_bg);
		dynamic_g -> add_tweights( i, weightFG, weightBG );
	}

	for(unsigned long i = 0; i < frames.size(); i++){
		int offset = offsets[i];
		Camera * camera				= frames[i]->camera;
		const unsigned int width	= camera->width;
		const unsigned int height	= camera->height;
		std::vector< std::vector<float> > & probs = pixel_weights[i];
		for(unsigned int w = 0; w < width;w++){
			for(unsigned int h = 0; h < height;h++){
				int ind = h*width+w;
				if(labels[ind+offset] == dynamic_label){
					if(w > 0 && probs[0][ind] > 0.00000001){
						int other = ind-1;
						if(labels[ind+offset] == labels[other+offset]){
							//dynamic_frameConnections++;
							double p_same = std::min(double(probs[0][ind]),maxprob_same);
							double not_p_same = 1-p_same;
							double weight = -log(not_p_same);
							dynamic_g -> add_edge( stat_ind[ind+offset], stat_ind[other+offset], weight, weight );
						}
					}

					if(h > 0 && probs[1][ind] > 0.00000001){
						int other = ind-width;
						if(labels[ind+offset] == labels[other+offset]){
							//dynamic_frameConnections++;
							double p_same = std::min(double(probs[0][ind]),maxprob_same);
							double not_p_same = 1-p_same;
							double weight = -log(not_p_same);
							dynamic_g -> add_edge( stat_ind[ind+offset], stat_ind[other+offset], weight, weight );
						}
					}
				}
			}
		}
	}

	for(unsigned long i = 0; i < current_point;i++){
		if(labels[i] == dynamic_label){
			for(unsigned int j = 0; j < interframe_connectionId[i].size();j++){
				if(labels[interframe_connectionId[i][j]] == labels[i]){
					double weight = interframe_connectionStrength[i][j];
					dynamic_g -> add_edge( stat_ind[i], stat_ind[ interframe_connectionId[i][j] ], weight, weight );
				}
			}
		}
	}
	dynamic_g -> maxflow();


	tot_inf = 0;
	for(unsigned int it = 0; false && it < 100; it++){
		double start_inf1 = getTime();
		dynamic_g -> maxflow();

		double diffs = 0;
		for(unsigned long i = 0; i < current_point;i++){
			if(labels[i] == dynamic_label){
				for(unsigned int j = 0; j < interframe_connectionId[i].size();j++){
					if(labels[interframe_connectionId[i][j]] == labels[i]){
						double weight = interframe_connectionStrength[i][j];

						unsigned long i2 = stat_ind[i];
						unsigned long j2 = stat_ind[ interframe_connectionId[i][j] ];
						if(weight > 1 && g->what_segment(i2) != g->what_segment(j2)){
							diffs++;
						}
	dfuncTMP->compute_infront		= true;
					}
				}
			}
		}


		double adds = 100000;
		double prob = std::min(adds / diffs,1.0);
		printf("diffs: %f adds: %f prob: %f\n",diffs,adds,prob);

		if(diffs < 1000){break;}


		double trueadds = 0;

		for(unsigned int i = 0; i < interframe_connectionId.size();i++){
			if(labels[i] == dynamic_label){
				for(unsigned int j = 0; j < interframe_connectionId[i].size();j++){
					if(labels[interframe_connectionId[i][j]] == labels[i]){

						double weight = interframe_connectionStrength[i][j];
						unsigned long other = interframe_connectionId[i][j];

						unsigned long i2 = stat_ind[i];
						unsigned long j2 = stat_ind[ interframe_connectionId[i][j] ];
						if(weight > 1 && g->what_segment(i2) != g->what_segment(j2)){
							if(rand() <= prob*RAND_MAX){
								trueadds++;
								g -> add_edge( i2, j2, weight, weight );

								interframe_connectionId_added[i].push_back(other);
								interframe_connectionStrength_added[i].push_back(weight);

								interframe_connectionStrength[i][j] = interframe_connectionStrength[i].back();
								interframe_connectionStrength[i].pop_back();

								interframe_connectionId[i][j] = interframe_connectionId[i].back();
								interframe_connectionId[i].pop_back();
								j--;
							}
						}
					}
				}
			}
		}
		printf("trueadds: %f\n",trueadds);

		tot_inf += getTime()-start_inf1;
		printf("static inference1 time: %10.10fs total: %10.10f\n\n",getTime()-start_inf1,tot_inf);
		if(tot_inf > 90){break;}
	}

	end_inf = getTime();
	printf("dynamic inference time: %10.10fs\n",end_inf-start_inf);

	long nr_moving = 0;
	for(unsigned long i = 0; i < dyn_ind.size();i++){
		int res = dynamic_g->what_segment(i);
		if(res == 1){
			labels[dyn_ind[i]] = 2;
			nr_moving++;
		}
	}
	delete dynamic_g;

	for(unsigned int i = 0; i < interframe_connectionId.size();i++){
		for(unsigned int j = 0; j < interframe_connectionId_added[i].size();j++){
			interframe_connectionStrength[i].push_back(interframe_connectionStrength_added[i][j]);
			interframe_connectionId[i].push_back(interframe_connectionId_added[i][j]);
		}
	}

	const unsigned int nr_frames		= frames.size();
	const unsigned int width			= frames[0]->camera->width;
	const unsigned int height			= frames[0]->camera->height;
	const unsigned int pixels_per_image	= width*height;
	const unsigned int nr_pixels		= nr_frames*pixels_per_image;

	double probthresh = 0.5;
	double str_probthresh = -log(probthresh);
	unsigned int number_of_dynamics = 0;
	unsigned int nr_obj_dyn = 0;
	unsigned int nr_obj_mov = 0;
	std::vector<unsigned int> objectlabel;
	std::vector<int> labelID;
	labelID.push_back(0);
	objectlabel.resize(nr_pixels);
	for(unsigned long i = 0; i < nr_pixels; i++){objectlabel[i] = 0;}
	for(unsigned long ind = 0; ind < nr_pixels; ind++){
		if(valids[ind] && objectlabel[ind] == 0 && labels[ind] != 0){
			unsigned int current_label = labels[ind];
			number_of_dynamics++;
			objectlabel[ind] = number_of_dynamics;
			unsigned long todocounter = 0;
			std::vector< unsigned long > todo;
			todo.push_back(ind);
			double score = 0;
			double totsum = 0;
			while(todocounter < todo.size()){
				unsigned long cind = todo[todocounter++];
				unsigned long frameind = cind / pixels_per_image;


				unsigned long iind = cind % pixels_per_image;
				unsigned long w = iind % width;
				unsigned long h = iind / width;

				double p0 = priors[3*cind+0];
				double p1 = priors[3*cind+1];
				double p2 = priors[3*cind+2];

				if(valids[cind]){
					double s = 0;
					if(p0 > p2){s += p1 - p0;}
					else{       s += p1 - p2;}
					score += s;
					totsum++;
				}

				float * dedata = (float*)(frames[frameind]->de.data);

				int dir;
				dir = -1;
				if( w > 0 && objectlabel[cind+dir] == 0 && labels[cind+dir] == current_label && (dedata[3*(iind+dir)+1]+dedata[3*(iind+dir)+2]) < probthresh){
					objectlabel[cind+dir] = number_of_dynamics;
					todo.push_back(cind+dir);
				}

				dir = 1;
				if( w < (width-1) && objectlabel[cind+dir] == 0 && labels[cind+dir] == current_label && (dedata[3*(iind+dir)+1]+dedata[3*(iind+dir)+2]) < probthresh){
					objectlabel[cind+dir] = number_of_dynamics;
					todo.push_back(cind+dir);
				}

				dir = -width;
				if( h > 0 && objectlabel[cind+dir] == 0 && labels[cind+dir] == current_label && (dedata[3*(iind+dir)+1]+dedata[3*(iind+dir)+2]) < probthresh){
					objectlabel[cind+dir] = number_of_dynamics;
					todo.push_back(cind+dir);
				}

				dir = width;
				if( h < (height-1) && objectlabel[cind+dir] == 0 && labels[cind+dir] == current_label && (dedata[3*(iind+dir)+1]+dedata[3*(iind+dir)+2]) < probthresh){
					objectlabel[cind+dir] = number_of_dynamics;
					todo.push_back(cind+dir);
				}

				for(unsigned long j = 0; j < interframe_connectionId[cind].size();j++){
					unsigned long other = interframe_connectionId[cind][j];
					if(interframe_connectionStrength[cind][j] > str_probthresh && objectlabel[other] == 0 && labels[other] == current_label){
						objectlabel[other] = number_of_dynamics;
						todo.push_back(other);

						double d = sqrt(pow(cloud->points[cind].x-cloud->points[other].x,2)+pow(cloud->points[cind].y-cloud->points[other].y,2)+pow(cloud->points[cind].z-cloud->points[other].z,2));
						if(d > 0.15){
							printf("d: %f -> %f\n",d,exp(-interframe_connectionStrength[cind][j]));
						}
					}
				}
			}

			printf("score: %f\n",score);

			if(debugg && todo.size() > 10000){


				printf("3-class\n");
				for(unsigned int i = 0; i < current_point; i++){
					cloud->points[i].r = priors[3*i+0]*255.0;
					cloud->points[i].g = priors[3*i+1]*255.0;
					cloud->points[i].b = priors[3*i+2]*255.0;
				}

				pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_sample (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
				cloud_sample->points.clear();
				for(unsigned int i = 0; i < current_point; i++){
					//if(rand() % 4 == 0){
						cloud_sample->points.push_back(cloud->points[i]);
					//}
				}
				viewer->removeAllPointClouds();
				viewer->addPointCloud<pcl::PointXYZRGBNormal> (cloud_sample, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(cloud_sample), "cloud");
				viewer->spin();

				for(unsigned int i = 0; i < current_point; i++){
					cloud_sample->points[i].r = 255;
					cloud_sample->points[i].g = 0;
					cloud_sample->points[i].b = 0;
				}

				for(unsigned int j = 0; j < todo.size(); j++){
					unsigned int i = todo[j];
					cloud_sample->points[i].r = 0;
					cloud_sample->points[i].g = 255;
					cloud_sample->points[i].b = 0;
				}
				viewer->removeAllPointClouds();
				viewer->addPointCloud<pcl::PointXYZRGBNormal> (cloud_sample, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(cloud_sample), "cloud");
				viewer->spin();
			}

			labelID.push_back(0);
			//if(score < 200){continue;}

			if(score < 0){continue;}
			if(current_label == 1){
				labelID.back() = ++nr_obj_dyn;
				printf("Dynamic: %f -> %f\n",score,totsum);
				sr.component_dynamic.push_back(todo);
				sr.scores_dynamic.push_back(score);
				sr.total_dynamic.push_back(totsum);
			}
			if(current_label == 2){
				labelID.back() = --nr_obj_mov;
				printf("Moving: %f -> %f\n",score,totsum);
				sr.component_moving.push_back(todo);
				sr.scores_moving.push_back(score);
				sr.total_moving.push_back(totsum);
			}
		}
	}
	interframe_connectionId.clear();
	interframe_connectionStrength.clear();

	printf("connectedComponent: %5.5fs\n",getTime()-start_inf);

	int current = 0;
	for(unsigned long i = 0; i < frames.size(); i++){
		Camera * camera				= frames[i]->camera;
		const unsigned int width	= camera->width;
		const unsigned int height	= camera->height;

		cv::Mat m;
		m.create(height,width,CV_8UC1);
		unsigned char * mdata = (unsigned char*)m.data;

		cv::Mat d;
		d.create(height,width,CV_8UC1);
		unsigned char * ddata = (unsigned char*)d.data;

		cv::Mat d2;
		d2.create(height,width,CV_8UC1);
		unsigned char * ddata2 = (unsigned char*)d2.data;

		for(int j = 0; j < width*height; j++){
			mdata[j] = 0;
			ddata[j] = 0;
			ddata2[j] = labels[current];
			unsigned int label = objectlabel[current];
			int lid = labelID[label];
			if(lid >  0){
				ddata[j] = lid;
			}else if(lid < 0){
				mdata[j] = -lid;
			}
			current++;
		}
		movemask.push_back(m);
		dynmask.push_back(d);
	}

	if(debugg){
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_sample (new pcl::PointCloud<pcl::PointXYZRGBNormal>);

		printf("3-class\n");
		for(unsigned int i = 0; i < current_point; i++){
			cloud->points[i].r = priors[3*i+0]*255.0;
			cloud->points[i].g = priors[3*i+1]*255.0;
			cloud->points[i].b = priors[3*i+2]*255.0;
		}

		cloud_sample->points.clear();
		for(unsigned int i = 0; i < current_point; i++){
			if(rand() % 4 == 0){
				cloud_sample->points.push_back(cloud->points[i]);
			}
		}
		viewer->removeAllPointClouds();
		viewer->addPointCloud<pcl::PointXYZRGBNormal> (cloud_sample, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(cloud_sample), "cloud");
		viewer->spin();


		cloud_sample->points.clear();
		for(unsigned int i = 0; i < current_point; i++){
			if(rand() % 4 == 0 && labels[i] == 0){
				cloud_sample->points.push_back(cloud->points[i]);
				cloud_sample->points.back().r = 0;
				cloud_sample->points.back().g = 0;
				cloud_sample->points.back().b = 255;
			}
		}

		for(unsigned int c = 0; c < sr.component_dynamic.size(); c++){
			int randr = rand()%256;
			int randg = rand()%256;
			int randb = rand()%256;
			for(unsigned int i = 0; i < sr.component_dynamic[c].size(); i++){
				cloud_sample->points.push_back(cloud->points[sr.component_dynamic[c][i]]);
				cloud_sample->points.back().r = randr;
				cloud_sample->points.back().g = randg;
				cloud_sample->points.back().b = randb;
			}
		}

		for(unsigned int c = 0; c < sr.component_moving.size(); c++){
			int randr = rand()%256;
			int randg = rand()%256;
			int randb = rand()%256;
			for(unsigned int i = 0; i < sr.component_moving[c].size(); i++){
				cloud_sample->points.push_back(cloud->points[sr.component_moving[c][i]]);
				cloud_sample->points.back().r = randr;
				cloud_sample->points.back().g = randg;
				cloud_sample->points.back().b = randb;
			}
		}
		viewer->removeAllPointClouds();
		viewer->addPointCloud<pcl::PointXYZRGBNormal> (cloud_sample, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(cloud_sample), "cloud");
		viewer->spin();
	}

	delete[] priors;
	printf("computeMovingDynamicStatic total time: %5.5fs\n",getTime()-computeMovingDynamicStatic_startTime);
}
*/

vector<Mat> ModelUpdater::computeDynamicObject(vector<Matrix4d> bgcp, vector<RGBDFrame*> bgcf, vector<Mat> bgmm, vector<Matrix4d> cp, vector<RGBDFrame*> cf, vector<Mat> mm, vector<Matrix4d> poses, vector<RGBDFrame*> frames, vector<Mat> masks, bool debugg){
	//
	std::vector<cv::Mat> newmasks;
	//for(double test = 0.99; test > 0.00000001; test *= 0.1)
	{
		//	double maxprob_same = 1-test;//0.99;

		double maxprob_same = 0.999999;

		double maxprob = 0.7;

		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud  (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);

		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr bgcloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);

		int tot_nr_pixels = 0;
		std::vector<int> offsets;

		for(unsigned int i = 0; i < frames.size(); i++){
			offsets.push_back(tot_nr_pixels);
			unsigned int nr_pixels = frames[i]->camera->width * frames[i]->camera->height;
			tot_nr_pixels += nr_pixels;
		}

		//Graph...
		std::vector<double> prior;
		std::vector< std::vector<int> > connectionId;
		std::vector< std::vector<double> > connectionStrength;
		std::vector< std::vector<int> > interframe_connectionId;
		std::vector< std::vector<double> > interframe_connectionStrength;

		prior.resize(tot_nr_pixels);
		connectionId.resize(tot_nr_pixels);
		connectionStrength.resize(tot_nr_pixels);
		interframe_connectionId.resize(tot_nr_pixels);
		interframe_connectionStrength.resize(tot_nr_pixels);



		for(unsigned int i = 0; i < frames.size(); i++){
			printf("frame: %i\n",i);
			int offset = offsets[i];
			RGBDFrame * frame = frames[i];
			unsigned short * depthdata	= (unsigned short	*)(frame->depth.data);
			unsigned char * rgbdata		= (unsigned char	*)(frame->rgb.data);
			unsigned char * detdata		= (unsigned char	*)(frame->det_dilate.data);
			Camera * camera				= frame->camera;
			const unsigned int width	= camera->width;
			const unsigned int height	= camera->height;


			unsigned int nr_pixels = frame->camera->width * frame->camera->height;
			double * overlaps	= new double[nr_pixels];
			double * occlusions		= new double[nr_pixels];
			for(unsigned int j = 0; j < nr_pixels; j++){
				overlaps[j] = 0;
				occlusions[j] = 0;
			}

			std::vector< std::vector<int> > interframe_connectionId_tmp;
			std::vector< std::vector<double> > interframe_connectionStrength_tmp;

			for(unsigned int j = 0; j < bgcp.size(); j++){
				if(frame == bgcf[j]){continue;}
				Eigen::Matrix4d p = poses[i].inverse() * bgcp[j];
				getDynamicWeights(false,p.inverse(), frame, overlaps, occlusions, bgcf[j],bgmm[j],0,0,interframe_connectionId_tmp,interframe_connectionStrength_tmp,false);
			}

			for(unsigned int j = 0; j < nr_pixels; j++){
				occlusions[j] = 0;
			}
			printf("CHECK AGAINST BACKGROUND\n");
			for(unsigned int j = 0; j < cp.size(); j++){
				if(frame == cf[j]){continue;}
				Eigen::Matrix4d p = poses[i].inverse() * cp[j];
				getDynamicWeights(false,p.inverse(), frame, overlaps, occlusions, cf[j],mm[j],0,0,interframe_connectionId_tmp,interframe_connectionStrength_tmp,false);
			}
			std::vector< std::vector<float> > probs = getImageProbs(frames[i],9);

			std::vector<double> frame_prior;
			std::vector< std::vector<int> > frame_connectionId;
			std::vector< std::vector<double> > frame_connectionStrength;

			frame_prior.resize(nr_pixels);
			frame_connectionId.resize(nr_pixels);
			frame_connectionStrength.resize(nr_pixels);


			printf("starting partition\n");
			double start_part = getTime();
			unsigned int nr_data = width*height;
			for(unsigned int j = 0; j < nr_data;j++){
				if(depthdata[j] == 0){
					frame_prior[j] = -1;
					prior[offset+j] = -1;
					continue;
				}

				double p_fg = 0.499999999;

				if(occlusions[j] >= 1){	p_fg = maxprob;}
				else if(overlaps[j] >= 1){	p_fg = 0.4;}

				p_fg = std::max(1-maxprob,std::min(maxprob,p_fg));

				frame_prior[j] = p_fg;
				prior[offset+j] = p_fg;
			}


			for(unsigned int w = 0; w < width;w++){
				for(unsigned int h = 0; h < height;h++){
					int ind = h*width+w;
					if(w > 0 && w < width-1){
						int other = ind-1;
						double p_same = std::min(double(probs[0][ind]),maxprob_same);
						double not_p_same = 1-p_same;
						double weight = -log(not_p_same);
						if(!std::isnan(weight) && weight > 0){
							frame_connectionId[ind].push_back(other);
							frame_connectionStrength[ind].push_back(weight);

							connectionId[ind+offset].push_back(other+offset);
							connectionStrength[ind+offset].push_back(weight);
						}
					}

					if(h > 0 && h < height-1){
						int other = ind-width;
						double p_same = std::min(double(probs[1][ind]),maxprob_same);
						double not_p_same = 1-p_same;
						double weight = -log(not_p_same);

						if(!std::isnan(weight) && weight > 0){
							frame_connectionId[ind].push_back(other);
							frame_connectionStrength[ind].push_back(weight);

							connectionId[ind+offset].push_back(other+offset);
							connectionStrength[ind+offset].push_back(weight);
						}
					}

					/*
				if(w > 0 && w < width-1){
					double ax = 0.5;
					double bx = 0.5;
					for(unsigned int p = 0; p < probs.size(); p+=2){
						double pr = probs[p][ind];
						ax *= pr;
						bx *= 1.0-pr;
					}
					double px = ax/(ax+bx);

					int other = ind-1;
					//double p_same = std::max(probs[probs.size()-2][ind],std::min(px,maxprob_same));
					//double p_same = std::max(probs[probs.size()-2][ind],std::min(px,maxprob_same));
					//double p_same = std::min(probs[probs.size()-2][ind],maxprob_same);
					//double p_same = std::max(1-maxprob_same,std::min(std::min(px,probs[probs.size()-2][ind]),maxprob_same));
					//double p_same = std::min(std::min(px,probs[probs.size()-2][ind]),maxprob_same);
					//double p_same = std::min(px,maxprob_same);

//					double p_same = std::min(probs[probs.size()-2][ind],maxprob_same);
//					if(detdata[ind] == 0){
//						p_same = std::min(px,maxprob_same);
//					}

					float p_same = std::min(px,maxprob_same);

					double not_p_same = 1-p_same;
					double weight = -log(not_p_same);
					if(!std::isnan(weight) && weight > 0){
						frame_connectionId[ind].push_back(other);
						frame_connectionStrength[ind].push_back(weight);

						connectionId[ind+offset].push_back(other+offset);
						connectionStrength[ind+offset].push_back(weight);
					}
				}

				if(h > 0 && h < height-1){

					double ay = 0.5;
					double by = 0.5;
					for(unsigned int p = 1; p < probs.size(); p+=2){
						double pr = probs[p][ind];
						ay *= pr;
						by *= 1.0-pr;
					}
					double py = ay/(ay+by);

					int other = ind-width;
					//double p_same = std::max(probs[probs.size()-1][ind],std::min(py,maxprob_same));

					//double p_same = std::min(py,maxprob_same);
					//double p_same = std::max(1-maxprob_same,std::min(std::min(py,probs[probs.size()-1][ind]),maxprob_same));
					//double p_same = std::min(probs[probs.size()-1][ind],maxprob_same);

					//double p_same = std::min(std::min(py,probs[probs.size()-1][ind]),maxprob_same);
					double p_same = std::min(py,maxprob_same);

					//double p_same = std::min(probs[probs.size()-1][ind],maxprob_same);
					//if(detdata[ind] == 0){
					//	p_same = std::min(py,maxprob_same);
					//}

					double not_p_same = 1-p_same;
					double weight = -log(not_p_same);

					if(!std::isnan(weight) && weight > 0){
						frame_connectionId[ind].push_back(other);
						frame_connectionStrength[ind].push_back(weight);

						connectionId[ind+offset].push_back(other+offset);
						connectionStrength[ind+offset].push_back(weight);
					}
				}
				*/
				}
			}



			double end_part = getTime();
			printf("part time: %10.10fs\n",end_part-start_part);

			//std::vector<int> labels = doInference(frame_prior, frame_connectionId, frame_connectionStrength);

			Eigen::Matrix4d p = poses[i];
			float m00 = p(0,0); float m01 = p(0,1); float m02 = p(0,2); float m03 = p(0,3);
			float m10 = p(1,0); float m11 = p(1,1); float m12 = p(1,2); float m13 = p(1,3);
			float m20 = p(2,0); float m21 = p(2,1); float m22 = p(2,2); float m23 = p(2,3);


			const float idepth			= camera->idepth_scale;
			const float cx				= camera->cx;
			const float cy				= camera->cy;
			const float ifx				= 1.0/camera->fx;
			const float ify				= 1.0/camera->fy;

			for(unsigned int w = 0; w < width;w++){
				for(unsigned int h = 0; h < height;h++){
					int ind = h*width+w;
					float z = idepth*float(depthdata[ind]);
					if(z > 0){
						float x = (float(w) - cx) * z * ifx;
						float y = (float(h) - cy) * z * ify;

						pcl::PointXYZRGBNormal point;
						point.x = m00*x + m01*y + m02*z + m03;
						point.y = m10*x + m11*y + m12*z + m13;
						point.z = m20*x + m21*y + m22*z + m23;
						point.b = rgbdata[3*ind+0];
						point.g = rgbdata[3*ind+1];
						point.r = rgbdata[3*ind+2];

						cloud->points.push_back(point);
					}
				}
			}

			bool debugg2 = debugg;
			if(debugg2){
				for(unsigned int w = 0; w < width;w++){
					for(unsigned int h = 0; h < height;h++){
						int ind = h*width+w;
						float z = idepth*float(depthdata[ind]);
						if(z > 0){
							float x = (float(w) - cx) * z * ifx;
							float y = (float(h) - cy) * z * ify;

							float tx	= m00*x + m01*y + m02*z + m03;
							float ty	= m10*x + m11*y + m12*z + m13;
							float tz	= m20*x + m21*y + m22*z + m23;

							pcl::PointXYZRGBNormal point;
							point.x = tx;
							point.y = ty;
							point.z = tz;

							point.r = 0;
							point.g = 255;
							point.b = 255;

							if(overlaps[ind] >= 1){
								point.r = 0;
								point.g = 255;
								point.b = 0;
							}
							if(occlusions[ind] >= 1){
								point.r = 255;
								point.g = 0;
								point.b = 0;
							}
							//cloud->points[ind] = point;
							cloud2->points.push_back(point);
						}
					}
				}
			}

			for(unsigned int j = 0; j < nr_pixels; j++){
				occlusions[j] = 0;
			}

			for(unsigned int j = 0; j < frames.size(); j++){
				if(frame == frames[j]){continue;}
				Eigen::Matrix4d p = poses[i].inverse() * poses[j];
				getDynamicWeights(false,p.inverse(), frame, overlaps, occlusions, frames[j],masks[j],offsets[i],offsets[j],interframe_connectionId,interframe_connectionStrength,false);
			}

			//Time to compute external masks...
			delete[] overlaps;
			delete[] occlusions;
		}

		//std::vector<int> global_labels = doInference(prior,connectionId,connectionStrength);

		for(unsigned int i = 0; i < interframe_connectionId.size(); i++){
			for(unsigned int j = 0; j < interframe_connectionId[i].size(); j++){
				connectionId[i].push_back(interframe_connectionId[i][j]);
				connectionStrength[i].push_back(interframe_connectionStrength[i][j]);
			}
		}

		std::vector<int> improvedglobal_labels = doInference(prior,connectionId,connectionStrength);

		for(unsigned int f = 0; f < frames.size(); f++){
			int offset = offsets[f];
			RGBDFrame * frame = frames[f];
			unsigned short * depthdata	= (unsigned short	*)(frame->depth.data);
			//		unsigned char * rgbdata		= (unsigned char	*)(frame->rgb.data);

			Camera * camera				= frame->camera;
			const unsigned int width	= camera->width;
			const unsigned int height	= camera->height;

			unsigned int nr_pixels = frame->camera->width * frame->camera->height;

			//		cv::Mat originalmask;
			//		originalmask.create(height,width,CV_8UC1);
			//		unsigned char * originaldata = (unsigned char *)(originalmask.data);

			cv::Mat improvedmask;
			improvedmask.create(height,width,CV_8UC1);
			unsigned char * improveddata = (unsigned char *)(improvedmask.data);

			//		cv::Mat diffmask;
			//		diffmask.create(height,width,CV_8UC3);
			//		unsigned char * diffdata = (unsigned char *)(diffmask.data);

			for(unsigned int i = 0; i < nr_pixels;i++){
				//internaldata[i] = 255.0*((depthdata[i] == 0) || (g->what_segment(i) == gc::Graph<double,double,double>::SOURCE));
				//originaldata[i] = 255.0*((depthdata[i] == 0) || (global_labels[i+offset] == gc::Graph<double,double,double>::SOURCE));
				improveddata[i] = 255.0*((depthdata[i] == 0) || (improvedglobal_labels[i+offset] == gc::Graph<double,double,double>::SOURCE));
				//			if(improveddata[i] < originaldata[i]){
				//				diffdata[3*i+0] = 0;
				//				diffdata[3*i+1] = 255;
				//				diffdata[3*i+2] = 0;
				//			}else if(improveddata[i] > originaldata[i]){
				//				diffdata[3*i+0] = 0;
				//				diffdata[3*i+1] = 0;
				//				diffdata[3*i+2] = 255;
				//			}else{
				//				diffdata[3*i+0] = 0;
				//				diffdata[3*i+1] = 0;
				//				diffdata[3*i+2] = 0;
				//			}


			}

			//				frame->show(false);
			//				//cv::namedWindow( "Original", cv::WINDOW_AUTOSIZE );
			//				//cv::imshow( "Original", originalmask );
			//				cv::namedWindow( "Improved", cv::WINDOW_AUTOSIZE );
			//				cv::imshow( "Improved", improvedmask );
			//				//cv::namedWindow( "Difference", cv::WINDOW_AUTOSIZE );
			//				//cv::imshow( "Difference", diffmask );
			//				cv::waitKey(0);
			newmasks.push_back(improvedmask);
		}


		if(debugg){


			for(unsigned int i = 0; i < bgcf.size(); i++){
				RGBDFrame * frame = bgcf[i];
				unsigned short * depthdata	= (unsigned short	*)(frame->depth.data);
				unsigned char * rgbdata		= (unsigned char	*)(frame->rgb.data);

				Camera * camera				= frame->camera;
				const unsigned int width	= camera->width;
				const unsigned int height	= camera->height;

				Eigen::Matrix4d p = bgcp[i];
				float m00 = p(0,0); float m01 = p(0,1); float m02 = p(0,2); float m03 = p(0,3);
				float m10 = p(1,0); float m11 = p(1,1); float m12 = p(1,2); float m13 = p(1,3);
				float m20 = p(2,0); float m21 = p(2,1); float m22 = p(2,2); float m23 = p(2,3);


				const float idepth			= camera->idepth_scale;
				const float cx				= camera->cx;
				const float cy				= camera->cy;
				const float ifx				= 1.0/camera->fx;
				const float ify				= 1.0/camera->fy;

				for(unsigned int w = 0; w < width;w++){
					for(unsigned int h = 0; h < height;h++){
						int ind = h*width+w;
						float z = idepth*float(depthdata[ind]);
						if(z > 0){
							float x = (float(w) - cx) * z * ifx;
							float y = (float(h) - cy) * z * ify;

							float tx	= m00*x + m01*y + m02*z + m03;
							float ty	= m10*x + m11*y + m12*z + m13;
							float tz	= m20*x + m21*y + m22*z + m23;

							pcl::PointXYZRGBNormal point;
							point.x = tx;
							point.y = ty;
							point.z = tz;

							point.b = rgbdata[3*ind+0];
							point.g = rgbdata[3*ind+1];
							point.r = rgbdata[3*ind+2];

							bgcloud->points.push_back(point);
						}
					}
				}
			}
			/*
		viewer->removeAllPointClouds();
		viewer->addPointCloud<pcl::PointXYZRGBNormal> (cloud1, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(cloud1), "scloud");
		viewer->spin();

		viewer->removeAllPointClouds();
		viewer->addPointCloud<pcl::PointXYZRGBNormal> (cloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(cloud), "scloud");
		viewer->spin();
		*/
			viewer->removeAllPointClouds();
			viewer->addPointCloud<pcl::PointXYZRGBNormal> (cloud2, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(cloud2), "scloud");
			//viewer->addPointCloud<pcl::PointXYZRGBNormal> (bgcloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(bgcloud), "bgcloud");
			viewer->spin();

			viewer->removeAllPointClouds();

		}
	}
	return newmasks;
}

std::vector<cv::Mat> ModelUpdater::computeDynamicObject(reglib::Model * bg,  Eigen::Matrix4d bgpose, vector<Matrix4d> cp, vector<RGBDFrame*> cf, vector<cv::Mat> oldmasks){
	std::vector<cv::Mat> newmasks;
	for(unsigned int i = 0; i < cf.size(); i++){
		unsigned short * depthdata		= (unsigned short	*)(cf[i]->depth.data);

		Camera * camera				= cf[i]->camera;
		const unsigned int width	= camera->width;
		const unsigned int height	= camera->height;


		unsigned int nr_pixels = cf[i]->camera->width * cf[i]->camera->height;
		double * overlaps	= new double[nr_pixels];
		double * occlusions		= new double[nr_pixels];
		for(unsigned int j = 0; j < nr_pixels; j++){
			overlaps[j] = 0;
			occlusions[j] = 0;
		}


		std::vector< std::vector<int> > interframe_connectionId_tmp;
		std::vector< std::vector<double> > interframe_connectionStrength_tmp;

		//        for(unsigned int j = 0; j < bgcp.size(); j++){
		//			if(frame == bgcf[j]){continue;}
		//            Eigen::Matrix4d p = poses[i].inverse() * bgcp[j];
		//			getDynamicWeights(false,p.inverse(), frame, overlaps, occlusions, bgcf[j],bgmm[j],0,0,interframe_connectionId_tmp,interframe_connectionStrength_tmp,false);
		//        }

		for(unsigned int j = 0; j < cf.size(); j++){
			if(i == j){continue;}
			Eigen::Matrix4d p = cp[i].inverse() * cp[j];
			getDynamicWeights(false,p.inverse(), cf[i], overlaps, occlusions, cf[j],oldmasks[j],0,0,interframe_connectionId_tmp,interframe_connectionStrength_tmp);
		}
		std::vector< std::vector<float> > probs = getImageProbs(cf[i],5);

		printf("starting partition\n");
		double start_part = getTime();
		unsigned int nr_data = width*height;
		gc::Graph<double,double,double> *g = new gc::Graph<double,double,double>( nr_data, width*(height-1) + (width-1)*height);

		double maxprob = 0.7;
		for(unsigned int j = 0; j < nr_data;j++){
			g -> add_node();
			if(depthdata[j] == 0){
				g -> add_tweights(j,0,0);
				continue;
			}

			double p_fg = 0.499;

			if(occlusions[j] >= 1){	p_fg = maxprob;}
			else if(overlaps[j] >= 1){	p_fg = 0.4;}

			p_fg = std::max(1-maxprob,std::min(maxprob,p_fg));

			double p_bg = 1-p_fg;
			double weightFG = -log(p_fg);
			double weightBG = -log(p_bg);
			g -> add_tweights( j, weightFG, weightBG );
		}

		float maxprob_same = 0.999999999;
		for(unsigned int w = 0; w < width;w++){
			for(unsigned int h = 0; h < height;h++){
				int ind = h*width+w;

				double ax = 0.5;
				double bx = 0.5;
				for(unsigned int p = 0; p < probs.size(); p+=2){
					double pr = probs[p][ind];
					ax *= pr;
					bx *= 1.0-pr;
				}
				float px = ax/(ax+bx);

				double ay = 0.5;
				double by = 0.5;
				for(unsigned int p = 1; p < probs.size(); p+=2){
					double pr = probs[p][ind];
					ay *= pr;
					by *= 1.0-pr;
				}
				float py = ay/(ay+by);

				if(w > 0){
					int other = ind-1;
					double p_same = std::max(probs[probs.size()-2][ind],std::min(px,maxprob_same));
					double not_p_same = 1-p_same;
					double weight = -log(not_p_same);
					g -> add_edge( ind, other, weight, weight );
				}

				if(h > 0){
					int other = ind-width;
					double p_same = std::max(probs[probs.size()-1][ind],std::min(py,maxprob_same));
					double not_p_same = 1-p_same;
					double weight = -log(not_p_same);
					g -> add_edge( ind, other, weight, weight );
				}
			}
		}

		int flow = g -> maxflow();
		printf("flow: %i\n",flow);


		double end_part = getTime();
		printf("part time: %10.10fs\n",end_part-start_part);

		cv::Mat internalmask;

		internalmask.create(height,width,CV_8UC1);
		unsigned char * internaldata = (unsigned char *)(internalmask.data);
		for(unsigned int i = 0; i < width*height;i++){internaldata[i] = 255.0*(g->what_segment(i) == gc::Graph<double,double,double>::SOURCE);}

		//		cv::imshow( "rgb", cf[i]->rgb );
		//		cv::imshow( "internalmask", internalmask );
		//		cv::waitKey(30);

		newmasks.push_back(internalmask);
		//Time to compute external masks...
		delete[] overlaps;
		delete[] occlusions;
	}

	return newmasks;
}

OcclusionScore ModelUpdater::computeOcclusionScore(RGBDFrame * src, ModelMask * src_modelmask, RGBDFrame * dst, ModelMask * dst_modelmask, Eigen::Matrix4d p, int step, bool debugg){
	OcclusionScore oc;

	unsigned char  * src_maskdata		= (unsigned char	*)(src_modelmask->mask.data);
	unsigned char  * src_rgbdata		= (unsigned char	*)(src->rgb.data);
	unsigned short * src_depthdata		= (unsigned short	*)(src->depth.data);
	float		   * src_normalsdata	= (float			*)(src->normals.data);

	unsigned char  * dst_maskdata		= (unsigned char	*)(dst_modelmask->mask.data);
	unsigned char  * dst_rgbdata		= (unsigned char	*)(dst->rgb.data);
	unsigned short * dst_depthdata		= (unsigned short	*)(dst->depth.data);
	float		   * dst_normalsdata	= (float			*)(dst->normals.data);

	float m00 = p(0,0); float m01 = p(0,1); float m02 = p(0,2); float m03 = p(0,3);
	float m10 = p(1,0); float m11 = p(1,1); float m12 = p(1,2); float m13 = p(1,3);
	float m20 = p(2,0); float m21 = p(2,1); float m22 = p(2,2); float m23 = p(2,3);

	Camera * src_camera				= src->camera;
	const unsigned int src_width	= src_camera->width;
	const unsigned int src_height	= src_camera->height;
	const float src_idepth			= src_camera->idepth_scale;
	const float src_cx				= src_camera->cx;
	const float src_cy				= src_camera->cy;
	const float src_ifx				= 1.0/src_camera->fx;
	const float src_ify				= 1.0/src_camera->fy;

	Camera * dst_camera				= dst->camera;
	const unsigned int dst_width	= dst_camera->width;
	const unsigned int dst_height	= dst_camera->height;
	const float dst_idepth			= dst_camera->idepth_scale;
	const float dst_cx				= dst_camera->cx;
	const float dst_cy				= dst_camera->cy;
	const float dst_fx				= dst_camera->fx;
	const float dst_fy				= dst_camera->fy;
	const float dst_ifx				= 1.0/dst_camera->fx;
	const float dst_ify				= 1.0/dst_camera->fy;
	const unsigned int dst_width2	= dst_camera->width  - 2;
	const unsigned int dst_height2	= dst_camera->height - 2;

	//bool debugg = false;
	cv::Mat debugg_img;
	unsigned char  * debugg_img_data;
	if(debugg){
		debugg_img		= src->rgb.clone();
		debugg_img_data	= (unsigned char	*)(debugg_img.data);
	}

	std::vector<float> residuals;
	std::vector<float> angleweights;
	std::vector<float> weights;
	std::vector<int> ws;
	std::vector<int> hs;

	std::vector<int> dst_ws;
	std::vector<int> dst_hs;

	residuals.reserve(src_width*src_height);
	angleweights.reserve(src_width*src_height);
	weights.reserve(src_width*src_height);
	if(debugg){
		ws.reserve(src_width*src_height);
		hs.reserve(src_width*src_height);
		dst_ws.reserve(dst_width*dst_height);
		dst_hs.reserve(dst_width*dst_height);
	}

	double sum = 0;
	double count = 0;

	std::vector<int> & testw = src_modelmask->testw;
	std::vector<int> & testh = src_modelmask->testh;

	unsigned int test_nrdata = testw.size();
	unsigned int indstep = step;
	//	int indstep = std::max(1.0,double(test_nrdata)/double(step));
	//	printf("indstep: %i\n",indstep);
	//	for(unsigned int ind = 0; ind < test_nrdata;ind+=indstep){
	//	//for(unsigned int src_w = 0; src_w < src_width-1; src_w++){
	//	//    for(unsigned int src_h = 0; src_h < src_height;src_h++){
	//		unsigned int src_w = testw[ind];
	//		unsigned int src_h = testh[ind];

	//		int src_ind0 = src_h*src_width+src_w-1;
	//		int src_ind1 = src_h*src_width+src_w;
	//		int src_ind2 = src_h*src_width+src_w+1;
	//		if(src_maskdata[src_ind0] == 255 && src_maskdata[src_ind1] == 255 && src_maskdata[src_ind2] == 255){// && p.z > 0 && !isnan(p.normal_x)){
	//			float z0 = src_idepth*float(src_depthdata[src_ind0]);
	//			float z1 = src_idepth*float(src_depthdata[src_ind1]);
	//			float z2 = src_idepth*float(src_depthdata[src_ind2]);

	//			double diff0 = (z0-z1)/(z0*z0+z1*z1);
	//			double diff1 = (z2-z1)/(z2*z2+z1*z1);
	//			if( fabs(diff0) < fabs(diff1)){
	//				if(diff0 != 0){
	//					sum += diff0*diff0;
	//					count++;
	//				}
	//			}else{
	//				if(diff1 != 0){
	//					sum += diff1*diff1;
	//					count++;
	//				}
	//			}
	//		}
	//	}
	//    double me = sqrt(sum/(count+1));
	//    double pred = 1.3*sqrt(2)*me;
	/*
	for(unsigned int ind = 0; ind < test_nrdata;ind++){
	//for(unsigned int src_w = 0; src_w < src_width-1; src_w++){
	//    for(unsigned int src_h = 0; src_h < src_height;src_h++){
		unsigned int src_w = testw[ind];
		unsigned int src_h = testh[ind];

		int src_ind = src_h*src_width+src_w;
		if(src_maskdata[src_ind] == 255){// && p.z > 0 && !isnan(p.normal_x)){
			float z = src_idepth*float(src_depthdata[src_ind]);
			float nx = src_normalsdata[3*src_ind+0];
			if(z > 0 && nx != 2){
			//if(z > 0){
				float x = (float(src_w) - src_cx) * z * src_ifx;
				float y = (float(src_h) - src_cy) * z * src_ify;

				float tx	= m00*x + m01*y + m02*z + m03;
				float ty	= m10*x + m11*y + m12*z + m13;
				float tz	= m20*x + m21*y + m22*z + m23;

				//float tnx	= m00*nx + m01*ny + m02*nz;
				//float tny	= m10*nx + m11*ny + m12*nz;
				float tnz	= m20*nx + m21*ny + m22*nz;

				float itz	= 1.0/tz;
				float dst_w	= dst_fx*tx*itz + dst_cx;
				float dst_h	= dst_fy*ty*itz + dst_cy;

				if((dst_w > 0) && (dst_h > 0) && (dst_w < dst_width2) && (dst_h < dst_height2)){
					unsigned int dst_ind = unsigned(dst_h+0.5) * dst_width + unsigned(dst_w+0.5);
					float dst_z = dst_idepth*float(dst_depthdata[dst_ind]);
					if(dst_z > 0){
						float diff_z = (dst_z-tz)/(z*z+dst_z*dst_z);//if tz < dst_z then tz infront and diff_z > 0
						residuals.push_back(diff_z);
						if(debugg){
							ws.push_back(src_w);
							hs.push_back(src_h);
						}
					}
				}
			}
		}
	}
*/
	for(unsigned int ind = 0; ind < test_nrdata;ind+=indstep){
		unsigned int src_w = testw[ind];
		unsigned int src_h = testh[ind];

		int src_ind = src_h*src_width+src_w;
		if(src_maskdata[src_ind] == 255){
			float src_z = src_idepth*float(src_depthdata[src_ind]);
			float src_nx = src_normalsdata[3*src_ind+0];
			if(src_z > 0 && src_nx != 2){
				float src_ny = src_normalsdata[3*src_ind+1];
				float src_nz = src_normalsdata[3*src_ind+2];
				//if(src_z > 0){
				float src_x = (float(src_w) - src_cx) * src_z * src_ifx;
				float src_y = (float(src_h) - src_cy) * src_z * src_ify;

				float tx	= m00*src_x + m01*src_y + m02*src_z + m03;
				float ty	= m10*src_x + m11*src_y + m12*src_z + m13;
				float tz	= m20*src_x + m21*src_y + m22*src_z + m23;
				float itz	= 1.0/tz;
				float dst_w	= dst_fx*tx*itz + dst_cx;
				float dst_h	= dst_fy*ty*itz + dst_cy;

				if((dst_w > 0) && (dst_h > 0) && (dst_w < dst_width2) && (dst_h < dst_height2)){
					unsigned int dst_ind = unsigned(dst_h+0.5) * dst_width + unsigned(dst_w+0.5);

					float dst_z = dst_idepth*float(dst_depthdata[dst_ind]);
					float dst_nx = dst_normalsdata[3*dst_ind+0];
					if(dst_z > 0 && dst_nx != 2){
						float dst_ny = dst_normalsdata[3*dst_ind+1];
						float dst_nz = dst_normalsdata[3*dst_ind+2];

						float dst_x = (float(dst_w) - dst_cx) * dst_z * dst_ifx;
						float dst_y = (float(dst_h) - dst_cy) * dst_z * dst_ify;

						float tnx	= m00*src_nx + m01*src_ny + m02*src_nz;
						float tny	= m10*src_nx + m11*src_ny + m12*src_nz;
						float tnz	= m20*src_nx + m21*src_ny + m22*src_nz;

						double info = (src_z*src_z+dst_z*dst_z);

						//double d = mysign(dst_z-tz)*fabs(dst_nx*(dst_x-tx) + dst_ny*(dst_y-ty) + dst_nz*(dst_z-tz));
						double d = mysign(dst_z-tz)*fabs(tnx*(dst_x-tx) + tny*(dst_y-ty) + tnz*(dst_z-tz));
						residuals.push_back(d/info);

						//float diff_z = (dst_z-tz)/(src_z*src_z+dst_z*dst_z);//if tz < dst_z then tz infront and diff_z > 0
						//residuals.push_back(diff_z);





						//						float dst_x = (float(dst_w) - dst_cx) * dst_z * dst_ifx;
						//						float dst_y = (float(dst_h) - dst_cy) * dst_z * dst_ify;
						//float angle = (tnx*dst_x+tny*dst_y+tnz*dst_z)/sqrt(dst_x*dst_x + dst_y*dst_y + dst_z*dst_z);
						//weights.push_back(1-angle);
						//weights.push_back(ny*dst_ny);


						double dist_dst = sqrt(dst_x*dst_x+dst_y*dst_y+dst_z*dst_z);
						double angle_dst = fabs((dst_x*dst_nx+dst_y*dst_ny+dst_z*dst_nz)/dist_dst);

						double dist_src = sqrt(src_x*src_x+src_y*src_y+src_z*src_z);
						double angle_src = fabs((src_x*src_nx+src_y*src_ny+src_z*src_nz)/dist_src);
						weights.push_back(angle_src * angle_dst);

						if(debugg){
							ws.push_back(src_w);
							hs.push_back(src_h);

							dst_ws.push_back(dst_w);
							dst_hs.push_back(dst_h);
						}
					}
				}
			}
		}
	}




	//	DistanceWeightFunction2PPR2 * func = new DistanceWeightFunction2PPR2();
	//	func->maxp			= 1.0;
	//	func->update_size	= true;
	//	func->zeromean      = true;
	//	func->startreg		= 0.0001;
	//	func->debugg_print	= debugg;
	//	func->bidir			= true;
	//	func->maxnoise      = pred;
	//	func->reset();

	DistanceWeightFunction2 * func = new DistanceWeightFunction2();
	func->f = THRESHOLD;
	func->p = 0.02;

	Eigen::MatrixXd X = Eigen::MatrixXd::Zero(1,residuals.size());
	for(unsigned int i = 0; i < residuals.size(); i++){X(0,i) = residuals[i];}
	func->computeModel(X);

	Eigen::VectorXd  W = func->getProbs(X);

	delete func;

	for(unsigned int i = 0; i < residuals.size(); i++){
		float r = residuals[i];
		float weight = W(i);
		float ocl = 0;
		if(r > 0){ocl += 1-weight;}
		oc.score		+= weight*weights.at(i);
		oc.occlusions	+= ocl*weights.at(i);
	}

	//debugg = true;
	if(debugg){
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr scloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr dcloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
		scloud->points.resize(src_width*src_height);
		dcloud->points.resize(dst_width*dst_height);
		/*
		for(unsigned int src_w = 0; src_w < src_width-1; src_w++){
			for(unsigned int src_h = 0; src_h < src_height;src_h++){
				int src_ind = src_h*src_width+src_w;
				if(src_maskdata[src_ind] != 255){continue;}
				float z = src_idepth*float(src_depthdata[src_ind]);
				if(z > 0){
					float x = (float(src_w) - src_cx) * z * src_ifx;
					float y = (float(src_h) - src_cy) * z * src_ify;
					float tx	= m00*x + m01*y + m02*z + m03;
					float ty	= m10*x + m11*y + m12*z + m13;
					float tz	= m20*x + m21*y + m22*z + m23;
					scloud->points[src_ind].x = tx;
					scloud->points[src_ind].y = ty;
					scloud->points[src_ind].z = tz+2;
					scloud->points[src_ind].r = 0;
					scloud->points[src_ind].g = 0;
					scloud->points[src_ind].b = 255;
				}
			}
		}
		for(unsigned int dst_w = 0; dst_w < dst_width; dst_w++){
			for(unsigned int dst_h = 0; dst_h < dst_height;dst_h++){
				unsigned int dst_ind = dst_h*dst_width+dst_w;
				if(true || dst_maskdata[dst_ind] == 255){// && p.z > 0 && !isnan(p.normal_x)){
					float z = dst_idepth*float(dst_depthdata[dst_ind]);
					if(z > 0){// && (dst_w%3 == 0) && (dst_h%3 == 0)){
						float x = (float(dst_w) - dst_cx) * z * dst_ifx;
						float y = (float(dst_h) - dst_cy) * z * dst_ify;
						dcloud->points[dst_ind].x = x;
						dcloud->points[dst_ind].y = y;
						dcloud->points[dst_ind].z = z+2;
						dcloud->points[dst_ind].r = dst_rgbdata[3*dst_ind+2];
						dcloud->points[dst_ind].g = dst_rgbdata[3*dst_ind+1];
						dcloud->points[dst_ind].b = dst_rgbdata[3*dst_ind+0];
						if(dst_maskdata[dst_ind] == 255){
							dcloud->points[dst_ind].r = 255;
							dcloud->points[dst_ind].g = 000;
							dcloud->points[dst_ind].b = 255;
						}
					}
				}
			}
		}
		viewer->removeAllPointClouds();
		//printf("%i showing results\n",__LINE__);
		viewer->addPointCloud<pcl::PointXYZRGBNormal> (scloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(scloud), "scloud");
		viewer->addPointCloud<pcl::PointXYZRGBNormal> (dcloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(dcloud), "dcloud");
		viewer->spin();
		viewer->removeAllPointClouds();
		*/
		/*
		for(unsigned int dst_w = 0; dst_w < dst_width; dst_w++){
			for(unsigned int dst_h = 0; dst_h < dst_height;dst_h++){
				int dst_ind = dst_h*dst_width+dst_w;
				dcloud->points[dst_ind].x = 0;
				dcloud->points[dst_ind].y = 0;
				dcloud->points[dst_ind].z = 0;
			}
		}


		for(unsigned int i = 0; i < residuals.size(); i++){
			float r = residuals[i];
			float weight = W(i);
			float ocl = 0;
			if(r > 0){ocl += 1-weight;}
			if(debugg){
				int w = ws[i];
				int h = hs[i];
				unsigned int src_ind = h * src_width + w;
				if(ocl > 0.01 || weight > 0.01){
					scloud->points[src_ind].r = 255.0*ocl;
					scloud->points[src_ind].g = 0;
					scloud->points[src_ind].b = 0;
				}else{
					scloud->points[src_ind].x = 0;
					scloud->points[src_ind].y = 0;
					scloud->points[src_ind].z = 0;
				}
			}
		}

		viewer->removeAllPointClouds();
		//printf("%i showing results\n",__LINE__);
		viewer->addPointCloud<pcl::PointXYZRGBNormal> (scloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(scloud), "scloud");
		viewer->addPointCloud<pcl::PointXYZRGBNormal> (dcloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(dcloud), "dcloud");
		viewer->spin();
		viewer->removeAllPointClouds();

		for(unsigned int src_w = 0; src_w < src_width-1; src_w++){
			for(unsigned int src_h = 0; src_h < src_height;src_h++){
				int src_ind = src_h*src_width+src_w;
				if(src_maskdata[src_ind] != 255){continue;}
				float z = src_idepth*float(src_depthdata[src_ind]);
				if(z > 0){
					float x = (float(src_w) - src_cx) * z * src_ifx;
					float y = (float(src_h) - src_cy) * z * src_ify;
					float tx	= m00*x + m01*y + m02*z + m03;
					float ty	= m10*x + m11*y + m12*z + m13;
					float tz	= m20*x + m21*y + m22*z + m23;
					scloud->points[src_ind].x = tx;
					scloud->points[src_ind].y = ty;
					scloud->points[src_ind].z = tz+2;
					scloud->points[src_ind].r = 0;
					scloud->points[src_ind].g = 0;
					scloud->points[src_ind].b = 255;
				}
			}
		}
		for(unsigned int dst_w = 0; dst_w < dst_width; dst_w++){
			for(unsigned int dst_h = 0; dst_h < dst_height;dst_h++){
				int dst_ind = dst_h*dst_width+dst_w;
				if(true || dst_maskdata[dst_ind] == 255){// && p.z > 0 && !isnan(p.normal_x)){
					float z = dst_idepth*float(dst_depthdata[dst_ind]);
					if(z > 0){// && (dst_w%3 == 0) && (dst_h%3 == 0)){
						float x = (float(dst_w) - dst_cx) * z * dst_ifx;
						float y = (float(dst_h) - dst_cy) * z * dst_ify;
						dcloud->points[dst_ind].x = x;
						dcloud->points[dst_ind].y = y;
						dcloud->points[dst_ind].z = z+2;
						dcloud->points[dst_ind].r = dst_rgbdata[3*dst_ind+2];
						dcloud->points[dst_ind].g = dst_rgbdata[3*dst_ind+1];
						dcloud->points[dst_ind].b = dst_rgbdata[3*dst_ind+0];
						if(dst_maskdata[dst_ind] == 255){
							dcloud->points[dst_ind].r = 255;
							dcloud->points[dst_ind].g = 000;
							dcloud->points[dst_ind].b = 255;
						}
					}
				}
			}
		}
		for(unsigned int i = 0; i < residuals.size(); i++){
			float r = residuals[i];
			float weight = W(i);
			float ocl = 0;
			if(r > 0){ocl += 1-weight;}
			if(debugg){
				int w = ws[i];
				int h = hs[i];
				unsigned int src_ind = h * src_width + w;
				if(ocl > 0.01 || weight > 0.01){
					scloud->points[src_ind].r = 255.0*ocl;//*weights.at(i);
					scloud->points[src_ind].g = 255.0*weight;//*weights.at(i);
					scloud->points[src_ind].b = 0;
				}else{
					scloud->points[src_ind].x = 0;
					scloud->points[src_ind].y = 0;
					scloud->points[src_ind].z = 0;
				}
			}
		}

		viewer->removeAllPointClouds();

		//printf("%i showing results\n",__LINE__);
		viewer->addPointCloud<pcl::PointXYZRGBNormal> (scloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(scloud), "scloud");
		viewer->addPointCloud<pcl::PointXYZRGBNormal> (dcloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(dcloud), "dcloud");
		viewer->spin();
		viewer->removeAllPointClouds();
*/
		/*
		for(unsigned int src_w = 0; src_w < src_width-1; src_w++){
			for(unsigned int src_h = 0; src_h < src_height;src_h++){
				int src_ind = src_h*src_width+src_w;
				if(src_maskdata[src_ind] != 255){continue;}
				float z = src_idepth*float(src_depthdata[src_ind]);
				if(z > 0){
					float x = (float(src_w) - src_cx) * z * src_ifx;
					float y = (float(src_h) - src_cy) * z * src_ify;
					scloud->points[src_ind].x = x;
					scloud->points[src_ind].y = y;
					scloud->points[src_ind].z = z;
					scloud->points[src_ind].r = 0;
					scloud->points[src_ind].g = 0;
					scloud->points[src_ind].b = 0;
				}
			}
		}
*/

		for(unsigned int dst_w = 0; dst_w < dst_width; dst_w++){
			for(unsigned int dst_h = 0; dst_h < dst_height;dst_h++){
				unsigned int dst_ind = dst_h*dst_width+dst_w;
				float z = dst_idepth*float(dst_depthdata[dst_ind]);
				if(z > 0){
					float x = (float(dst_w) - dst_cx) * z * dst_ifx;
					float y = (float(dst_h) - dst_cy) * z * dst_ify;
					dcloud->points[dst_ind].x = x;
					dcloud->points[dst_ind].y = y;
					dcloud->points[dst_ind].z = z+2;
					dcloud->points[dst_ind].r = dst_rgbdata[3*dst_ind+2];
					dcloud->points[dst_ind].g = dst_rgbdata[3*dst_ind+1];
					dcloud->points[dst_ind].b = dst_rgbdata[3*dst_ind+0];
					if(false && dst_maskdata[dst_ind] == 255){
						dcloud->points[dst_ind].r = 255;
						dcloud->points[dst_ind].g = 000;
						dcloud->points[dst_ind].b = 255;
					}
				}
			}
		}
		//		viewer->removeAllPointClouds();
		//		//printf("%i showing results\n",__LINE__);
		//		viewer->addPointCloud<pcl::PointXYZRGBNormal> (scloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(scloud), "scloud");
		//		viewer->addPointCloud<pcl::PointXYZRGBNormal> (dcloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(dcloud), "dcloud");

		viewer->removeAllShapes();
		for(unsigned int i = 0; i < residuals.size(); i++){
			float r = residuals[i];
			float weight = W(i);
			float ocl = 0;
			if(r > 0){ocl += 1-weight;}
			if(debugg){
				int w = ws[i];
				int h = hs[i];
				unsigned int src_ind = h * src_width + w;

				int dst_w = dst_ws[i];
				int dst_h = dst_hs[i];
				unsigned int dst_ind = dst_h * src_width + dst_w;
				if(ocl > 0.01 || weight > 0.01){
					float z = src_idepth*float(src_depthdata[src_ind]);
					float x = (float(w) - src_cx) * z * src_ifx;
					float y = (float(h) - src_cy) * z * src_ify;
					float tx	= m00*x + m01*y + m02*z + m03;
					float ty	= m10*x + m11*y + m12*z + m13;
					float tz	= m20*x + m21*y + m22*z + m23;
					scloud->points[src_ind].x = tx;
					scloud->points[src_ind].y = ty;
					scloud->points[src_ind].z = tz+2;

					scloud->points[src_ind].r = 255.0*ocl*weights.at(i);
					scloud->points[src_ind].g = 255.0*weight*weights.at(i);
					scloud->points[src_ind].b = 0;

					char buf [1024];
					sprintf(buf,"line%i",i);
					viewer->addLine<pcl::PointXYZRGBNormal> (scloud->points[src_ind], dcloud->points[dst_ind],buf);
				}else{
					scloud->points[src_ind].x = 0;
					scloud->points[src_ind].y = 0;
					scloud->points[src_ind].z = 0;
				}
			}
		}

		viewer->removeAllPointClouds();
		viewer->addPointCloud<pcl::PointXYZRGBNormal> (scloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(scloud), "scloud");
		viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "scloud");

		viewer->addPointCloud<pcl::PointXYZRGBNormal> (dcloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(dcloud), "dcloud");
		viewer->spin();
		viewer->removeAllPointClouds();

		/*
		for(unsigned int i = 0; i < residuals.size(); i++){
			float r = residuals[i];
			float weight = W(i);
			float ocl = 0;
			if(r > 0){ocl += 1-weight;}
			if(debugg){
				int w = ws[i];
				int h = hs[i];
				unsigned int src_ind = h * src_width + w;
				if(ocl > 0.01 || weight > 0.01){
					scloud->points[src_ind].r = 255.0*(1-weights.at(i));
					scloud->points[src_ind].g = 255.0*weights.at(i);
					scloud->points[src_ind].b = 0;
				}else{
					scloud->points[src_ind].x = 0;
					scloud->points[src_ind].y = 0;
					scloud->points[src_ind].z = 0;
				}
			}
		}

		viewer->removeAllPointClouds();
		viewer->addPointCloud<pcl::PointXYZRGBNormal> (scloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(scloud), "scloud");
		viewer->spin();
		viewer->removeAllPointClouds();
*/
	}
	/*
	if(debugg){
		viewer->removeAllPointClouds();
		viewer->addPointCloud<pcl::PointXYZRGBNormal> (scloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(scloud), "scloud");
		viewer->addPointCloud<pcl::PointXYZRGBNormal> (dcloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(dcloud), "dcloud");
		cv::namedWindow("debuggimage",		cv::WINDOW_AUTOSIZE );
		cv::imshow(		"debuggimage",		debugg_img );
		//cv::waitKey(30);
		viewer->spin();
		viewer->removeAllPointClouds();
	}
	*/
	return oc;
}

using namespace std;
using namespace Eigen;

vector<vector< OcclusionScore > > ModelUpdater::computeAllOcclusionScores(RGBDFrame * src, cv::Mat src_mask, RGBDFrame * dst, cv::Mat dst_mask,Eigen::Matrix4d p, bool debugg){
	unsigned char  * src_maskdata		= (unsigned char	*)(src_mask.data);
	unsigned char  * src_rgbdata		= (unsigned char	*)(src->rgb.data);
	unsigned short * src_depthdata		= (unsigned short	*)(src->depth.data);
	float		   * src_normalsdata	= (float			*)(src->normals.data);

	unsigned char  * dst_maskdata		= (unsigned char	*)(dst_mask.data);
	unsigned char  * dst_rgbdata		= (unsigned char	*)(dst->rgb.data);
	unsigned short * dst_depthdata		= (unsigned short	*)(dst->depth.data);
	float		   * dst_normalsdata	= (float			*)(dst->normals.data);

	float m00 = p(0,0); float m01 = p(0,1); float m02 = p(0,2); float m03 = p(0,3);
	float m10 = p(1,0); float m11 = p(1,1); float m12 = p(1,2); float m13 = p(1,3);
	float m20 = p(2,0); float m21 = p(2,1); float m22 = p(2,2); float m23 = p(2,3);

	Camera * src_camera				= src->camera;
	const unsigned int src_width	= src_camera->width;
	const unsigned int src_height	= src_camera->height;
	const float src_idepth			= src_camera->idepth_scale;
	const float src_cx				= src_camera->cx;
	const float src_cy				= src_camera->cy;
	const float src_ifx				= 1.0/src_camera->fx;
	const float src_ify				= 1.0/src_camera->fy;
	const int * src_labels			= src->labels;
	const int src_nr_labels			= src->nr_labels;

	Camera * dst_camera				= dst->camera;
	const unsigned int dst_width	= dst_camera->width;
	const unsigned int dst_height	= dst_camera->height;
	const float dst_idepth			= dst_camera->idepth_scale;
	const float dst_cx				= dst_camera->cx;
	const float dst_cy				= dst_camera->cy;
	const float dst_fx				= dst_camera->fx;
	const float dst_fy				= dst_camera->fy;
	const float dst_ifx				= 1.0/dst_camera->fx;
	const float dst_ify				= 1.0/dst_camera->fy;
	const unsigned int dst_width2	= dst_camera->width  - 2;
	const unsigned int dst_height2	= dst_camera->height - 2;
	const int * dst_labels			= dst->labels;
	const int dst_nr_labels			= src->nr_labels;

	vector< vector< OcclusionScore > > all_scores;
	all_scores.resize(src_nr_labels);
	for(int i = 0; i < src_nr_labels; i++){
		all_scores[i].resize(dst_nr_labels);
	}

	std::vector< std::vector< std::vector<float> > > all_residuals;
	all_residuals.resize(src_nr_labels);
	for(int i = 0; i < src_nr_labels; i++){
		all_residuals[i].resize(dst_nr_labels);
	}

	for(unsigned int src_w = 0; src_w < src_width; src_w++){
		for(unsigned int src_h = 0; src_h < src_height;src_h++){
			int src_ind = src_h*src_width+src_w;
			if(src_maskdata[src_ind] == 255){// && p.z > 0 && !isnan(p.normal_x)){
				float z = src_idepth*float(src_depthdata[src_ind]);
				float nx = src_normalsdata[3*src_ind+0];

				if(z > 0 && nx != 2){
					float ny = src_normalsdata[3*src_ind+1];
					float nz = src_normalsdata[3*src_ind+2];

					float x = (float(src_w) - src_cx) * z * src_ifx;
					float y = (float(src_h) - src_cy) * z * src_ify;

					float tx	= m00*x + m01*y + m02*z + m03;
					float ty	= m10*x + m11*y + m12*z + m13;
					float tz	= m20*x + m21*y + m22*z + m23;

					float tnx	= m00*nx + m01*ny + m02*nz;
					float tny	= m10*nx + m11*ny + m12*nz;
					float tnz	= m20*nx + m21*ny + m22*nz;

					float itz	= 1.0/tz;
					float dst_w	= dst_fx*tx*itz + dst_cx;
					float dst_h	= dst_fy*ty*itz + dst_cy;

					if((dst_w > 0) && (dst_h > 0) && (dst_w < dst_width2) && (dst_h < dst_height2)){
						unsigned int dst_ind = unsigned(dst_h+0.5) * dst_width + unsigned(dst_w+0.5);

						float dst_z = dst_idepth*float(dst_depthdata[dst_ind]);
						if(dst_z > 0){
							float dst_x = (float(dst_w) - dst_cx) * dst_z * dst_ifx;
							float dst_y = (float(dst_h) - dst_cy) * dst_z * dst_ify;
							float diff_z2 = tnx*(dst_x-tx)+tny*(dst_y-ty)+tnz*(dst_z-tz);

							float diff_z = (dst_z-tz)/(z*z+dst_z*dst_z);//if tz < dst_z then tz infront and diff_z > 0
							if(diff_z < 0 && diff_z2 > 0){diff_z2 *= -1;}
							if(diff_z > 0 && diff_z2 < 0){diff_z2 *= -1;}
							diff_z2 /= (z*z+dst_z*dst_z);//if tz < dst_z then tz infront and diff_z > 0

							int src_label = src_labels[src_ind];
							int dst_label = dst_labels[dst_ind];
							all_residuals[src_label][dst_label].push_back(diff_z2);
						}
					}
				}
			}
		}
	}

	DistanceWeightFunction2PPR * func = new DistanceWeightFunction2PPR();
	func->update_size = true;
	func->startreg = 0.00001;
	func->debugg_print = true;
	func->reset();


	delete func;

	for(int i = 0; i < src_nr_labels; i++){
		for(int j = 0; j < dst_nr_labels; i++){
			std::vector<float> & resi = all_residuals[i][j];
			OcclusionScore score;
			for(unsigned int k = 0; k < resi.size(); k++){
				float r = resi[k];
				float weight = 1;
				if(fabs(r) > 0.0005){weight = 0;}//Replace with PPR

				float ocl = 0;
				if(r > 0){ocl += 1-weight;}

				score.score			+= weight;
				score.occlusions	+= ocl;
			}
			all_scores[i][j] = score;
		}
	}
	return all_scores;
}

vector<vector < OcclusionScore > > ModelUpdater::getOcclusionScores(std::vector<Eigen::Matrix4d> current_poses, std::vector<RGBDFrame*> current_frames, std::vector<ModelMask*> current_modelmasks, bool debugg_scores, double speedup){
	//printf("getOcclusionScores\n");

	long total_points = 0;
	for(unsigned int i = 0; i < current_frames.size(); i++){total_points+=current_modelmasks[i]->testw.size();}
	int step = std::max(long(1),long(speedup*total_points*long(current_frames.size()))/long(50000000));

	//	printf("total_points: %i\n",total_points);
	//	printf("current_frames.size(): %i\n",current_frames.size());
	//	printf("ratio: %f\n",double(total_points*long(current_frames.size()))/double(50000000));
	printf("step: %i\n",step);

	vector<vector < OcclusionScore > > occlusionScores;
	occlusionScores.resize(current_frames.size());
	for(unsigned int i = 0; i < current_frames.size(); i++){occlusionScores[i].resize(current_frames.size());}

	int max_points = step;//100000.0/double(current_frames.size()*(current_frames.size()-1));
	//float occlusion_penalty = 10.0f;
	std::vector<std::vector < float > > scores;
	scores.resize(occlusionScores.size());
	for(unsigned int i = 0; i < occlusionScores.size(); i++){scores[i].resize(occlusionScores.size());}

	bool lock = false;
	for(unsigned int i = 0; i < current_frames.size(); i++){
		scores[i][i] = 0;
		for(unsigned int j = i+1; j < current_frames.size(); j++){
			if(lock && current_modelmasks[j]->sweepid == current_modelmasks[i]->sweepid && current_modelmasks[j]->sweepid != -1){
				occlusionScores[i][j].score = 999999;
				occlusionScores[i][j].occlusions = 0;
				occlusionScores[j][i].score = 999999;
				occlusionScores[j][i].occlusions = 0;
			}else{
				Eigen::Matrix4d relative_pose = current_poses[i].inverse() * current_poses[j];
				occlusionScores[j][i]		= computeOcclusionScore(current_frames[j], current_modelmasks[j],current_frames[i], current_modelmasks[i], relative_pose,max_points,debugg_scores);
				occlusionScores[i][j]		= computeOcclusionScore(current_frames[i], current_modelmasks[i],current_frames[j], current_modelmasks[j], relative_pose.inverse(),max_points,debugg_scores);
				//printf("scores: %i %i -> occlusion_penalty: %f -> (%f %f) and (%f %f) -> %f \n",i,j,occlusion_penalty,occlusionScores[i][j].score,occlusionScores[i][j].occlusions,occlusionScores[j][i].score,occlusionScores[j][i].occlusions,occlusionScores[i][j].score+occlusionScores[j][i].score - occlusion_penalty*(occlusionScores[i][j].occlusions+occlusionScores[j][i].occlusions));
			}
			scores[i][j] = occlusionScores[i][j].score+occlusionScores[j][i].score - occlusion_penalty*(occlusionScores[i][j].occlusions+occlusionScores[j][i].occlusions);
			scores[j][i] = scores[i][j];
		}
	}
	return occlusionScores;
}

CloudData * ModelUpdater::getCD(std::vector<Eigen::Matrix4d> current_poses, std::vector<RGBDFrame*> current_frames,std::vector<cv::Mat> current_masks, int step){return 0;}

void ModelUpdater::computeMassRegistration(std::vector<Eigen::Matrix4d> current_poses, std::vector<RGBDFrame*> current_frames,std::vector<cv::Mat> current_masks){}

std::vector<std::vector < float > > ModelUpdater::getScores(std::vector<std::vector < OcclusionScore > > occlusionScores){//, float occlusion_penalty){
	std::vector<std::vector < float > > scores;
	scores.resize(occlusionScores.size());
	for(unsigned int i = 0; i < occlusionScores.size(); i++){scores[i].resize(occlusionScores.size());}
	for(unsigned int i = 0; i < scores.size(); i++){
		scores[i][i] = 0;
		for(unsigned int j = i+1; j < scores.size(); j++){
			scores[i][j] = occlusionScores[i][j].score+occlusionScores[j][i].score - occlusion_penalty*(occlusionScores[i][j].occlusions+occlusionScores[j][i].occlusions);
			scores[j][i] = scores[i][j];
		}
	}
	return scores;
}

}
