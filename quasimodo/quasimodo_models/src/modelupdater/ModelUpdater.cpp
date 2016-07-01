#include "modelupdater/ModelUpdater.h"

#include <boost/graph/incremental_components.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/iteration_macros.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/one_bit_color_map.hpp>
#include <boost/graph/stoer_wagner_min_cut.hpp>
#include <boost/range/adaptor/reversed.hpp>
#include <boost/graph/copy.hpp>
#include <unordered_map>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

typedef boost::property<boost::edge_weight_t, float> edge_weight_property;
typedef boost::property<boost::vertex_name_t, size_t> vertex_name_property;
using Graph = boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, vertex_name_property, edge_weight_property>;
using Vertex = boost::graph_traits<Graph>::vertex_descriptor;
using VertexIndex = boost::graph_traits<Graph>::vertices_size_type;
using Edge = boost::graph_traits<Graph>::edge_descriptor;
using Components = boost::component_index<VertexIndex>;

using namespace std;

namespace reglib
{

double mysign(double v){
	if(v < 0){return -1;}
	return 1;
}

double getTime(){
	struct timeval start1;
	gettimeofday(&start1, NULL);
	return double(start1.tv_sec+(start1.tv_usec/1000000.0));
}

float graph_cut(vector<Graph*>& graphs_out,vector<vector<int>>& second_graphinds, Graph& graph_in, std::vector<int> graph_inds){
	using adjacency_iterator = boost::graph_traits<Graph>::adjacency_iterator;
	typename boost::property_map<Graph, boost::vertex_index_t>::type vertex_id		= boost::get(boost::vertex_index, graph_in);
	typename boost::property_map<Graph, boost::edge_weight_t>::type  edge_id		= boost::get(boost::edge_weight, graph_in);
	typename boost::property_map<Graph, boost::vertex_name_t>::type  vertex_name	= boost::get(boost::vertex_name, graph_in);

	BOOST_AUTO(parities, boost::make_one_bit_color_map(boost::num_vertices(graph_in), boost::get(boost::vertex_index, graph_in)));

	float w = boost::stoer_wagner_min_cut(graph_in, boost::get(boost::edge_weight, graph_in), boost::parity_map(parities));

	unordered_map<VertexIndex, VertexIndex> mappings;
	VertexIndex counters[2] = {0, 0};

	graphs_out.push_back(new Graph(1));
	graphs_out.push_back(new Graph(1));
	second_graphinds.push_back(vector<int>());
	second_graphinds.push_back(vector<int>());
	//std::cout << "One set of vertices consists of:" << std::endl;
	bool flag;
	Edge edge;
	for (size_t i = 0; i < boost::num_vertices(graph_in); ++i) {
		int first = boost::get(parities, i);
		second_graphinds[first].push_back(graph_inds[i]);
		// iterate adjacent edges
		adjacency_iterator ai, ai_end;
		for (tie(ai, ai_end) = boost::adjacent_vertices(i, graph_in);  ai != ai_end; ++ai) {
			VertexIndex neighbor_index = boost::get(vertex_id, *ai);
			int second = boost::get(parities, neighbor_index);
			if (first == second && neighbor_index < i) {
				tie(edge, flag) = boost::edge(i, neighbor_index, graph_in);
				edge_weight_property weight = boost::get(edge_id, edge);
				if (mappings.count(i) == 0) {
					mappings[i] = counters[first]++;
				}
				if (mappings.count(neighbor_index) == 0) {
					mappings[neighbor_index] = counters[first]++;
				}
				tie(edge, flag) = boost::add_edge(mappings[neighbor_index], mappings[i], weight, *graphs_out[first]);

				typename boost::property_map<Graph, boost::vertex_name_t>::type vertex_name_first = boost::get(boost::vertex_name, *graphs_out[first]);
				boost::get(vertex_name_first, mappings[i]) = boost::get(vertex_name, i);
				boost::get(vertex_name_first, mappings[neighbor_index]) = boost::get(vertex_name, *ai);
			}
		}
	}
	return w;
}

float recursive_split(std::vector<Graph*> * graphs_out,std::vector<std::vector<int>> * graphinds_out, Graph * graph, std::vector<int> graph_inds){
	if(boost::num_vertices(*graph) == 1){
		graphs_out->push_back(graph);
		graphinds_out->push_back(graph_inds);
		return 0;
	}

	vector<Graph*> second_graphs;
	vector<vector<int>> second_graphinds;
	float w = graph_cut(second_graphs,second_graphinds,*graph,graph_inds);
	if(w <= 0){
		delete graph;
		return 2*w + recursive_split(graphs_out, graphinds_out,second_graphs.front(),second_graphinds.front()) + recursive_split(graphs_out, graphinds_out, second_graphs.back(),second_graphinds.back());
	}else{
		graphs_out->push_back(graph);
		graphinds_out->push_back(graph_inds);
		delete second_graphs.front();
		delete second_graphs.back();
		return 0;
	}
}

std::vector<int> partition_graph(std::vector< std::vector< float > > & scores){
	int nr_data = scores.size();
	Graph* graph = new Graph(nr_data);
	std::vector<int> graph_inds;
	graph_inds.resize(nr_data);

	typename boost::property_map<Graph, boost::vertex_name_t>::type vertex_name = boost::get(boost::vertex_name, *graph);

	float sum = 0;
	for(int i = 0; i < nr_data; i++){
		graph_inds[i] = i;
		for(int j = i+1; j < nr_data; j++){
			float weight = scores[i][j];
			if(weight != 0){
				sum += 2*weight;
				edge_weight_property e = weight;
				boost::add_edge(i, j, e, *graph);
			}
		}
	}

	std::vector<Graph*> * graphs_out = new std::vector<Graph*>();
	std::vector<std::vector<int>> * graphinds_out = new std::vector<std::vector<int>>();
	float best = sum-recursive_split(graphs_out,graphinds_out, graph,graph_inds );

	std::vector<int> part;
	part.resize(nr_data);
	//int * part = new int[nr_data];
	for(unsigned int i = 0; i < graphinds_out->size(); i++){
		for(unsigned int j = 0; j < graphinds_out->at(i).size(); j++){
			part[graphinds_out->at(i).at(j)] = i;
		}
	}
	return part;
}

std::vector<int> ModelUpdater::getPartition(std::vector< std::vector< float > > & scores, int dims, int nr_todo, double timelimit){
	return partition_graph(scores);
}

ModelUpdater::ModelUpdater(){
	occlusion_penalty = 10;
    massreg_timeout = 60;
    model = 0;
}

ModelUpdater::ModelUpdater(Model * model_){	model = model_;}
ModelUpdater::~ModelUpdater(){}

FusionResults ModelUpdater::registerModel(Model * model2, Eigen::Matrix4d guess, double uncertanity){return FusionResults();}

void ModelUpdater::fuse(Model * model2, Eigen::Matrix4d guess, double uncertanity){
	for(unsigned int i = 0; i < model2->frames.size();i++){
		model->frames.push_back(model2->frames[i]);
		//model->masks.push_back(model2->masks[i]);
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

	for(unsigned int i = 0; i < scores.size(); i++){
		for(unsigned int j = 0; j < scores.size(); j++){
			if(scores[i][j] > 0){printf(" ");}
			printf("%5.5f ",0.0001*scores[i][j]);
//			if(scores[i][j] < 0){
//				Eigen::Matrix4d relative_pose = model->relativeposes[i].inverse() * model->relativeposes[j];
//				computeOcclusionScore(model->frames[j], model->modelmasks[j],model->frames[i], model->modelmasks[i], relative_pose,1,true);
//				computeOcclusionScore(model->frames[i], model->modelmasks[i],model->frames[j], model->modelmasks[j], relative_pose.inverse(),1,true);
//			}
		}
		printf("\n");
	}

	printf("partition");
	for(unsigned int i = 0; i < partition.size(); i++){printf("%i ", partition[i]);}
	printf("\n");
	return failed;
}

OcclusionScore ModelUpdater::computeOcclusionScore(vector<superpoint> & spvec, Matrix4d cp, RGBDFrame* cf, ModelMask* cm, int step,  bool debugg){
//	printf("start:: %s::%i\n",__FILE__,__LINE__);
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
	func->p = 0.01;

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
		//cf->show(true);

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
					if(dst_w % 10 == 0 && dst_h % 10 == 0){
						//printf("%i %i -> %4.4f %4.4f %4.4f\n",dst_w,dst_h,dst_normalsdata[3*dst_ind+0],dst_h,dst_normalsdata[3*dst_ind+1],dst_h,dst_normalsdata[3*dst_ind+2]);
					}
					//if(false && dst_maskdata[dst_ind] == 255){
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

					if(i % 100 == 0){
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

    int minHessian = 400;

    cv::SurfFeatureDetector detector( minHessian );
    cv::SurfDescriptorExtractor extractor;


    //cv::ORB orb = cv::ORB(250);//,1.2f, 1, 3, 0,2, cv::ORB::HARRIS_SCORE, 31);
    cv::ORB orb = cv::ORB(10);//,1.2f, 1, 3, 0,2, cv::ORB::HARRIS_SCORE, 31);
//	for(unsigned int i = 0; i < model->frames.size(); i++){
//        std::vector<cv::KeyPoint> keypoints;
//        cv::Mat descriptors;
//        //orb(model->frames[i]->rgb, model->modelmasks[i]->getMask(), keypoints, descriptors);
//        orb(model->frames[i]->rgb, cv::Mat(), keypoints, descriptors);
//        model->all_keypoints.push_back(keypoints);
//        model->all_descriptors.push_back(descriptors);

//        printf("keypoints: %i\n",keypoints.size());


//        cv::Mat descriptors_surf;
//        std::vector<cv::KeyPoint> keypoints_surf;

//        detector.detect(model->frames[i]->rgb, keypoints_surf, model->modelmasks[i]->getMask() );
//        extractor.compute( model->frames[i]->rgb, keypoints_surf, descriptors_surf );
//        model->all_keypoints.push_back(keypoints_surf);
//        model->all_descriptors.push_back(descriptors_surf);

////        printf("keypoints: %i\n",keypoints.size());
////        cv::Mat out;
////        cv::drawKeypoints(model->frames[i]->rgb, keypoints_surf, out, cv::Scalar::all(255));
////        cv::imshow("Kpts", out);
////        cv::waitKey(0);
//	}

	MassRegistrationPPR2 * massreg = new MassRegistrationPPR2(0.05);
	massreg->timeout = 4*massreg_timeout;
	massreg->viewer = viewer;
	massreg->visualizationLvl = 1;//1;
	massreg->maskstep = std::max(1,int(0.5+0.2*double(model->frames.size())));
	massreg->nomaskstep = std::max(5,int(0.5+1.0*double(model->frames.size())));//std::max(1,int(0.5+1.0*double(model->frames.size())));
	massreg->nomask = true;
	massreg->stopval = 0.0001;

	//massreg->setData(model->frames,model->modelmasks);
	massreg->addModelData(model, false);
    std::vector<Eigen::Matrix4d> p = model->submodels_relativeposes;
    p.insert(p.end(), model->relativeposes.begin(), model->relativeposes.end());


    //MassFusionResults mfr = massreg->getTransforms(model->relativeposes);
    MassFusionResults mfr = massreg->getTransforms(p);


    model->relativeposes.clear();// = mfr.poses;
    model->relativeposes.insert( model->relativeposes.end(), mfr.poses.begin()+model->submodels.size(), mfr.poses.end());

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

	bool * isfused = new bool[width*height];
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

					float pb = rgbdata[3*ind+0];
					float pg = rgbdata[3*ind+1];
					float pr = rgbdata[3*ind+2];

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

	delete[] isfused;

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
			//printf("%3.3i -> %f\n",i,score);
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
			//printf("removing %i\n",worst_id);
		}else{break;}
	}
}

void ModelUpdater::refine(double reg,bool useFullMask, int visualization){
	//return ;
	printf("void ModelUpdater::refine()\n");

printf("%s::%i\n",__FILE__,__LINE__);
	vector<vector < OcclusionScore > > ocs = computeOcclusionScore(model->submodels,model->submodels_relativeposes,1,false);
printf("%s::%i\n",__FILE__,__LINE__);
	std::vector<std::vector < float > > scores = getScores(ocs);
printf("%s::%i\n",__FILE__,__LINE__);

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
	massreg->visualizationLvl = 1;
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
	//printf("void ModelUpdater::show()\n");
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
	func->p = 0.005;

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

	bool lock = true;
    for(unsigned int i = 0; i < current_frames.size(); i++){
		scores[i][i] = 0;
        for(unsigned int j = i+1; j < current_frames.size(); j++){
			//printf("scores %i %i\n",i,j);
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
