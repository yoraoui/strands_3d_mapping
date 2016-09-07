#include "core/Util.h"
namespace reglib{

	double mysign(double v){
		if(v < 0){return -1;}
		return 1;
	}

	double getTime(){
		struct timeval start1;
		gettimeofday(&start1, NULL);
		return double(start1.tv_sec+(start1.tv_usec/1000000.0));
	}

	float graph_cut(std::vector<Graph*>& graphs_out,std::vector<std::vector<int>>& second_graphinds, Graph& graph_in, std::vector<int> graph_inds){
		using adjacency_iterator = boost::graph_traits<Graph>::adjacency_iterator;
		typename boost::property_map<Graph, boost::vertex_index_t>::type vertex_id		= boost::get(boost::vertex_index, graph_in);
		typename boost::property_map<Graph, boost::edge_weight_t>::type  edge_id		= boost::get(boost::edge_weight, graph_in);
		typename boost::property_map<Graph, boost::vertex_name_t>::type  vertex_name	= boost::get(boost::vertex_name, graph_in);

		BOOST_AUTO(parities, boost::make_one_bit_color_map(boost::num_vertices(graph_in), boost::get(boost::vertex_index, graph_in)));

		float w = boost::stoer_wagner_min_cut(graph_in, boost::get(boost::edge_weight, graph_in), boost::parity_map(parities));

		std::unordered_map<VertexIndex, VertexIndex> mappings;
		VertexIndex counters[2] = {0, 0};

		graphs_out.push_back(new Graph(1));
		graphs_out.push_back(new Graph(1));
		second_graphinds.push_back(std::vector<int>());
		second_graphinds.push_back(std::vector<int>());
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

		std::vector<Graph*> second_graphs;
		std::vector<std::vector<int>> second_graphinds;
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

	Eigen::Matrix4d getMatTest(const double * const camera, int mode){
		Eigen::Matrix4d ret = Eigen::Matrix4d::Identity();
		double rr [9];
		ceres::AngleAxisToRotationMatrix(camera,rr);

		ret(0,0) = rr[0];
		ret(1,0) = rr[1];
		ret(2,0) = rr[2];

		ret(0,1) = rr[3];
		ret(1,1) = rr[4];
		ret(2,1) = rr[5];

		ret(0,2) = rr[6];
		ret(1,2) = rr[7];
		ret(2,2) = rr[8];

		ret(0,3) = camera[3];
		ret(1,3) = camera[4];
		ret(2,3) = camera[5];
		return ret;
	}


	double * getCamera(Eigen::Matrix4d mat, int mode){
		double * camera = new double[6];
		double rr [9];
		rr[0] = mat(0,0);
		rr[1] = mat(1,0);
		rr[2] = mat(2,0);

		rr[3] = mat(0,1);
		rr[4] = mat(1,1);
		rr[5] = mat(2,1);

		rr[6] = mat(0,2);
		rr[7] = mat(1,2);
		rr[8] = mat(2,2);
		ceres::RotationMatrixToAngleAxis(rr,camera);

		camera[3] = mat(0,3);
		camera[4] = mat(1,3);
		camera[5] = mat(2,3);
		return camera;
	}

	Eigen::Matrix4d constructTransformationMatrix (const double & alpha, const double & beta, const double & gamma, const double & tx,    const double & ty,   const double & tz){
		// Construct the transformation matrix from rotation and translation
		Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Zero ();
		transformation_matrix (0, 0) =  cos (gamma) * cos (beta);
		transformation_matrix (0, 1) = -sin (gamma) * cos (alpha) + cos (gamma) * sin (beta) * sin (alpha);
		transformation_matrix (0, 2) =  sin (gamma) * sin (alpha) + cos (gamma) * sin (beta) * cos (alpha);
		transformation_matrix (1, 0) =  sin (gamma) * cos (beta);
		transformation_matrix (1, 1) =  cos (gamma) * cos (alpha) + sin (gamma) * sin (beta) * sin (alpha);
		transformation_matrix (1, 2) = -cos (gamma) * sin (alpha) + sin (gamma) * sin (beta) * cos (alpha);
		transformation_matrix (2, 0) = -sin (beta);
		transformation_matrix (2, 1) =  cos (beta) * sin (alpha);
		transformation_matrix (2, 2) =  cos (beta) * cos (alpha);

		transformation_matrix (0, 3) = tx;
		transformation_matrix (1, 3) = ty;
		transformation_matrix (2, 3) = tz;
		transformation_matrix (3, 3) = 1;
		return transformation_matrix;
	}

	void point_to_plane2(		Eigen::Matrix<double, 3, Eigen::Dynamic> & X,
								Eigen::Matrix<double, 3, Eigen::Dynamic> & Xn,
								Eigen::Matrix<double, 3, Eigen::Dynamic> & Y,
								Eigen::Matrix<double, 3, Eigen::Dynamic> & Yn,
								Eigen::VectorXd & W){
		typedef Eigen::Matrix<double, 6, 1> Vector6d;
		typedef Eigen::Matrix<double, 6, 6> Matrix6d;

		Matrix6d ATA;
		Vector6d ATb;
		ATA.setZero ();
		ATb.setZero ();

		unsigned int xcols = X.cols();
		for(unsigned int i=0; i < xcols; i++) {
			const double & sx = X(0,i);
			const double & sy = X(1,i);
			const double & sz = X(2,i);
			const double & dx = Y(0,i);
			const double & dy = Y(1,i);
			const double & dz = Y(2,i);
			const double & nx = Xn(0,i);
			const double & ny = Xn(1,i);
			const double & nz = Xn(2,i);

			const double & weight = W(i);

			double a = nz*sy - ny*sz;
			double b = nx*sz - nz*sx;
			double c = ny*sx - nx*sy;

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

			double d = weight * (nx*dx + ny*dy + nz*dz - nx*sx - ny*sy - nz*sz);

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

		for(int k = 0; k < 6; k++){
			ATA(k,k) += 1;
		}
		// Solve A*x = b
		Vector6d x = static_cast<Vector6d> (ATA.inverse () * ATb);
		//Vector6d x = static_cast<Vector6d> (ATA.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(ATb));
		Eigen::Affine3d transformation = Eigen::Affine3d(constructTransformationMatrix(x(0,0),x(1,0),x(2,0),x(3,0),x(4,0),x(5,0)));

		X = transformation*X;
		transformation(0,3) = 0;
		transformation(1,3) = 0;
		transformation(2,3) = 0;
		Xn = transformation*Xn;
	}

    bool isconverged(std::vector<Eigen::Matrix4d> before, std::vector<Eigen::Matrix4d> after, double stopvalr, double stopvalt, bool verbose){
		double change_trans = 0;
		double change_rot = 0;
		unsigned int nr_frames = after.size();
		for(unsigned int i = 0; i < nr_frames; i++){
			for(unsigned int j = i+1; j < nr_frames; j++){
				Eigen::Matrix4d diff_before = after[i].inverse()*after[j];
				Eigen::Matrix4d diff_after	= before[i].inverse()*before[j];
				Eigen::Matrix4d diff = diff_before.inverse()*diff_after;
				double dt = 0;
				for(unsigned int k = 0; k < 3; k++){
					dt += diff(k,3)*diff(k,3);
					for(unsigned int l = 0; l < 3; l++){
						if(k == l){ change_rot += fabs(1-diff(k,l));}
						else{		change_rot += fabs(diff(k,l));}
					}
				}
				change_trans += sqrt(dt);
			}
		}

		change_trans /= double(nr_frames*(nr_frames-1));
		change_rot	 /= double(nr_frames*(nr_frames-1));

        if(verbose){
            printf("change_trans: %10.10f change_rot: %10.10f\n",change_trans,change_rot);
        }

		if(change_trans < stopvalt && change_rot < stopvalr){return true;}
		else{return false;}
	}

	//OcclusionScore ModelUpdater::computeOcclusionScore(vector<superpoint> & spvec, Matrix4d cp, RGBDFrame* cf){
	//	unsigned char  * dst_rgbdata		= (unsigned char	*)(cf->rgb.data);
	//	unsigned short * dst_depthdata		= (unsigned short	*)(cf->depth.data);
	//	float		   * dst_normalsdata	= (float			*)(cf->normals.data);

	//	float m00 = cp(0,0); float m01 = cp(0,1); float m02 = cp(0,2); float m03 = cp(0,3);
	//	float m10 = cp(1,0); float m11 = cp(1,1); float m12 = cp(1,2); float m13 = cp(1,3);
	//	float m20 = cp(2,0); float m21 = cp(2,1); float m22 = cp(2,2); float m23 = cp(2,3);

	//	Camera * dst_camera				= cf->camera;
	//	const unsigned int dst_width	= dst_camera->width;
	//	const unsigned int dst_height	= dst_camera->height;
	//	const float dst_idepth			= dst_camera->idepth_scale;
	//	const float dst_cx				= dst_camera->cx;
	//	const float dst_cy				= dst_camera->cy;
	//	const float dst_fx				= dst_camera->fx;
	//	const float dst_fy				= dst_camera->fy;
	//	const float dst_ifx				= 1.0/dst_camera->fx;
	//	const float dst_ify				= 1.0/dst_camera->fy;
	//	const unsigned int dst_width2	= dst_camera->width  - 2;
	//	const unsigned int dst_height2	= dst_camera->height - 2;


	//	std::vector<int>	src_inds;
	//	std::vector<int>	dst_inds;
	//	std::vector<double> residuals;
	//	std::vector<double>	cameraSurfaceAngles;
	//	std::vector<double>	angles;
	//	src_inds.reserve(nr_data);
	//	dst_inds.reserve(nr_data);
	//	residuals.reserve(nr_data);
	//	cameraSurfaceAngles.reserve(nr_data);
	//	angles.reserve(nr_data);

	//	unsigned long nr_data = spvec.size();
	//	for(unsigned long ind = 0; ind < nr_data;ind++){
	//		superpoint & sp = spvec[ind];

	//		float src_x = sp.point(0);
	//		float src_y = sp.point(1);
	//		float src_z = sp.point(2);

	//		float src_nx = sp.normal(0);
	//		float src_ny = sp.normal(1);
	//		float src_nz = sp.normal(2);

	//		float point_information = sp.point_information;

	//		float tx	= m00*src_x + m01*src_y + m02*src_z + m03;
	//		float ty	= m10*src_x + m11*src_y + m12*src_z + m13;
	//		float tz	= m20*src_x + m21*src_y + m22*src_z + m23;

	//		float itz	= 1.0/tz;
	//		float dst_w	= dst_fx*tx*itz + dst_cx;
	//		float dst_h	= dst_fy*ty*itz + dst_cy;

	//		if((dst_w > 0) && (dst_h > 0) && (dst_w < dst_width2) && (dst_h < dst_height2)){
	//			unsigned int dst_ind = unsigned(dst_h+0.5) * dst_width + unsigned(dst_w+0.5);

	//			float dst_z = dst_idepth*float(dst_depthdata[dst_ind]);
	//			float dst_nx = dst_normalsdata[3*dst_ind+0];
	//			if(dst_z > 0 && dst_nx != 2){
	//				if(dst_detdata[dst_ind] != 0){continue;}
	//				float dst_ny = dst_normalsdata[3*dst_ind+1];
	//				float dst_nz = dst_normalsdata[3*dst_ind+2];

	//				float dst_x = (float(dst_w) - dst_cx) * dst_z * dst_ifx;
	//				float dst_y = (float(dst_h) - dst_cy) * dst_z * dst_ify;

	//				float tnx	= m00*src_nx + m01*src_ny + m02*src_nz;
	//				float tny	= m10*src_nx + m11*src_ny + m12*src_nz;
	//				float tnz	= m20*src_nx + m21*src_ny + m22*src_nz;

	//				double d = mysign(dst_z-tz)*fabs(tnx*(dst_x-tx) + tny*(dst_y-ty) + tnz*(dst_z-tz));
	//				double dst_noise = dst_z * dst_z;
	//				double point_noise = 1.0/sqrt(point_information);

	//				double compare_mul = sqrt(dst_noise*dst_noise + point_noise*point_noise);
	//				d *= compare_mul;

	//				double dist_dst = sqrt(dst_x*dst_x+dst_y*dst_y+dst_z*dst_z);
	//				double angle_dst = fabs((dst_x*dst_nx+dst_y*dst_ny+dst_z*dst_nz)/dist_dst);

	//				residuals.push_back(d);
	//				weights.push_back(angle_dst*angle_dst*angle_dst);
	//				src_inds.push_back(ind);
	//				dst_inds.push_back(dst_ind);
	//			}
	//		}
	//	}
	//}
}
