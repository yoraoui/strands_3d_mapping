#include "registration/RegistrationRefinement.h"
#include <iostream>
#include <fstream>

//#include "registration/myhull.h"

//#include <pcl/surface/convex_hull.h>

namespace reglib
{

RegistrationRefinement::RegistrationRefinement(){
	//func = 0;
	nr_arraypoints = 0;
	arraypoints = 0;
	trees3d = 0;
	a3d = 0;
	only_initial_guess		= false;

	type					= PointToPlane;
	//type					= PointToPoint;
	use_PPR_weight			= true;
	use_features			= true;
	normalize_matchweights	= true;

	visualizationLvl = 1;

	target_points	= 250;
	dst_points		= 2500;
	allow_regularization = true;
	maxtime = 9999999;

    regularization = 0.1;
    convergence = 0.25;

	//	func = new DistanceWeightFunction2PPR2();
	//	func->startreg			= 0.1;
	//	func->debugg_print		= false;
}
RegistrationRefinement::~RegistrationRefinement(){
	//if(func != 0){delete func; func = 0;}
	if(arraypoints != 0){delete arraypoints; arraypoints = 0;}
	if(trees3d != 0){delete trees3d; trees3d = 0;}
	if(a3d != 0){delete a3d; a3d = 0;}
}

void RegistrationRefinement::setDst(std::vector<superpoint> & dst_){
	double startTime = getTime();

	dst = dst_;
	unsigned int d_nr_data = dst.size();//dst->data.cols();
	int stepy = std::max(1,int(d_nr_data)/dst_points);
	Y.resize(Eigen::NoChange,d_nr_data/stepy);
	N.resize(Eigen::NoChange,d_nr_data/stepy);
	ycols = Y.cols();


	int count = 0;
	for(unsigned int i = 0; i < d_nr_data/stepy; i++){
		Y(0,i)	= dst[i*stepy].x;//dst->data(0,i*stepy);
		Y(1,i)	= dst[i*stepy].y;//dst->data(1,i*stepy);
		Y(2,i)	= dst[i*stepy].z;//dst->data(2,i*stepy);
		N(0,i)	= dst[i*stepy].nx;//dst->normals(0,i*stepy);
		N(1,i)	= dst[i*stepy].ny;//dst->normals(1,i*stepy);
		N(2,i)	= dst[i*stepy].nz;//dst->normals(2,i*stepy);
		count++;
	}

	if(arraypoints != 0){delete arraypoints;}
	if(trees3d != 0){delete trees3d;}
	if(a3d != 0){delete a3d;}

	arraypoints = new double[3*count];
	nr_arraypoints = count;
	for(unsigned int c = 0; c < count; c++){
		arraypoints[3*c+0] = Y(0,c);
		arraypoints[3*c+1] = Y(1,c);
		arraypoints[3*c+2] = Y(2,c);
	}

	a3d = new ArrayData3D<double>;
	a3d->data	= arraypoints;
	a3d->rows	= count;
	trees3d	= new Tree3d(3, *a3d, nanoflann::KDTreeSingleIndexAdaptorParams(10));
	trees3d->buildIndex();

	DST_INORMATION = Eigen::VectorXd::Zero(ycols);
    for(unsigned int i = 0; i < d_nr_data/stepy; i++){DST_INORMATION(i) = sqrt(dst[i*stepy].point_information);}

	addTime("setDst", getTime()-startTime);
}

FusionResults RegistrationRefinement::getTransform(Eigen::MatrixXd guess){

double initStart = getTime();
	std::vector<double> total_dweight;
	total_dweight.resize(ycols);

	DistanceWeightFunction2PPR2 * func = new DistanceWeightFunction2PPR2();
	if(allow_regularization){
        func->startreg			= regularization;
	}else{
		func->startreg			= 0.0;
	}
    func->debugg_print		= visualizationLvl > 3;
    func->minNoise              = 0.0005;




	double stop		= 0.00001;

	func->reset();

	float m00 = guess(0,0); float m01 = guess(0,1); float m02 = guess(0,2); float m03 = guess(0,3);
	float m10 = guess(1,0); float m11 = guess(1,1); float m12 = guess(1,2); float m13 = guess(1,3);
	float m20 = guess(2,0); float m21 = guess(2,1); float m22 = guess(2,2); float m23 = guess(2,3);


    unsigned int s_nr_data = src.size();
    std::vector<int> inds;
    inds.resize(s_nr_data);
    for(unsigned int i = 0; i < s_nr_data; i++){inds[i] = i;}
    for(unsigned int i = 0; i < s_nr_data; i++){
        int rind = rand()%s_nr_data;
        int tmp = inds[i];
        inds[i] = inds[rind];
        inds[rind] = tmp;
    }

    unsigned int xcols = std::min(int(s_nr_data),int(target_points));

    Eigen::Matrix<double, 3, Eigen::Dynamic> X;
    Eigen::Matrix<double, 3, Eigen::Dynamic> Xn;
    X.resize( Eigen::NoChange,xcols);
    Xn.resize(Eigen::NoChange,xcols);

    /// Buffers
    Eigen::Matrix3Xd Qp		= Eigen::Matrix3Xd::Zero(3,	xcols);
    Eigen::Matrix3Xd Qn		= Eigen::Matrix3Xd::Zero(3,	xcols);
    Eigen::VectorXd  W		= Eigen::VectorXd::Zero(	xcols);
    Eigen::VectorXd  Wold	= Eigen::VectorXd::Zero(	xcols);
    Eigen::VectorXd  rangeW	= Eigen::VectorXd::Zero(	xcols);

    Eigen::VectorXd SRC_INORMATION = Eigen::VectorXd::Zero(xcols);
    for(unsigned int i = 0; i < xcols; i++){
        superpoint & p = src[inds[i]];

        SRC_INORMATION(i) = sqrt(p.point_information);//src->information(0,i*stepx);
        float x		= p.x;//src->data(0,i*stepx);
        float y		= p.y;//src->data(1,i*stepx);
        float z		= p.z;//src->data(2,i*stepx);
        float xn	= p.nx;//src->normals(0,i*stepx);
        float yn	= p.ny;//src->normals(1,i*stepx);
        float zn	= p.nz;//src->normals(2,i*stepx);
        X(0,i)	= m00*x + m01*y + m02*z + m03;
        X(1,i)	= m10*x + m11*y + m12*z + m13;
        X(2,i)	= m20*x + m21*y + m22*z + m23;
        Xn(0,i)	= m00*xn + m01*yn + m02*zn;
        Xn(1,i)	= m10*xn + m11*yn + m12*zn;
        Xn(2,i)	= m20*xn + m21*yn + m22*zn;
    }

/*
    unsigned int s_nr_data = src.size();//src->data.cols();
    int stepx = std::max(1,int(s_nr_data)/target_points);
	Eigen::Matrix<double, 3, Eigen::Dynamic> X;
	Eigen::Matrix<double, 3, Eigen::Dynamic> Xn;
	X.resize(Eigen::NoChange,s_nr_data/stepx);
	Xn.resize(Eigen::NoChange,s_nr_data/stepx);
	unsigned int xcols = X.cols();


	/// Buffers
	Eigen::Matrix3Xd Qp		= Eigen::Matrix3Xd::Zero(3,	xcols);
	Eigen::Matrix3Xd Qn		= Eigen::Matrix3Xd::Zero(3,	xcols);
	Eigen::VectorXd  W		= Eigen::VectorXd::Zero(	xcols);
	Eigen::VectorXd  Wold	= Eigen::VectorXd::Zero(	xcols);
	Eigen::VectorXd  rangeW	= Eigen::VectorXd::Zero(	xcols);

	Eigen::VectorXd SRC_INORMATION = Eigen::VectorXd::Zero(xcols);
	for(unsigned int i = 0; i < s_nr_data/stepx; i++){
		SRC_INORMATION(i) = src[i*stepx].point_information;//src->information(0,i*stepx);
		float x		= src[i*stepx].x;//src->data(0,i*stepx);
		float y		= src[i*stepx].y;//src->data(1,i*stepx);
		float z		= src[i*stepx].z;//src->data(2,i*stepx);
		float xn	= src[i*stepx].nx;//src->normals(0,i*stepx);
		float yn	= src[i*stepx].ny;//src->normals(1,i*stepx);
		float zn	= src[i*stepx].nz;//src->normals(2,i*stepx);
		X(0,i)	= m00*x + m01*y + m02*z + m03;
		X(1,i)	= m10*x + m11*y + m12*z + m13;
		X(2,i)	= m20*x + m21*y + m22*z + m23;
		Xn(0,i)	= m00*xn + m01*yn + m02*zn;
		Xn(1,i)	= m10*xn + m11*yn + m12*zn;
		Xn(2,i)	= m20*xn + m21*yn + m22*zn;
	}
*/
	Eigen::Matrix3Xd Xo1 = X;
	Eigen::Matrix3Xd Xo2 = X;
	Eigen::Matrix3Xd Xo3 = X;
	Eigen::Matrix3Xd Xo4 = X;
	Eigen::MatrixXd residuals;

	std::vector<int> matchid;
	matchid.resize(xcols);
	double score = 0;
	stop = 99999;

    if(visualizationLvl >= 3){printf("before noise: %f\n",func->getNoise());show(X,Y,true);}

	//	//printf("X: %i %i Y: %i %i\n",X.cols(), X.rows(),Y.cols(), Y.rows());
	//double startTest = getTime();
	//std::vector<size_t>				ret_indexes1(1);
	//std::vector<double>				out_dists_sqr1(1);
	//nanoflann::KNNResultSet<double>	resultSet1(1);
	//resultSet1.init(&ret_indexes1[0], &out_dists_sqr1[0] );
	//double qp1 [3];
	//nanoflann::SearchParams sp1 = nanoflann::SearchParams(10);

	//for(unsigned long it = 0; true; it++){
	//	qp1[0] = 0.001*(rand()%1000);
	//	qp1[1] = 0.001*(rand()%1000);
	//	qp1[2] = 0.001*(rand()%1000);
	//	trees3d->findNeighbors(resultSet1, qp1, sp1);
	//	int mid = ret_indexes1[0];
	//	if(it % 1000){
	//		printf("mps: %15.15f\n",double(it)/(getTime()-startTest));
	//	}
	//}

int wasted = 0;
int total = 0;

double last_stop1 = 1000000;
double last_stop2 = 1000000;
double last_stop3 = 1000000;

addTime("init", getTime()-initStart);

	double min_meaningfull_distance = 90000000000000000000000;

	double start = getTime();
	long matches = 0;
	bool timestopped = false;
	/// ICP
	for(int funcupdate=0; funcupdate < 100; ++funcupdate) {
		if( (getTime()-start) > maxtime ){timestopped = true; break;}


        double last_stop4 = 1000000;
        for(int rematching=0; rematching < 5; ++rematching) {
			if( (getTime()-start) > maxtime ){timestopped = true; break;}
            if(visualizationLvl >= 1){printf("func: %i rematch: %i\n",funcupdate,rematching);}

            double matchStart = getTime();
            if(funcupdate==0 || rematching > 0){
                //#pragma omp parallel for
                for(unsigned int i=0; i< xcols; ++i) {
                    std::vector<size_t>   ret_indexes(1);
                    std::vector<double> out_dists_sqr(1);
                    nanoflann::KNNResultSet<double> resultSet(1);

                    double * qp = X.col(i).data();
                    resultSet.init(&ret_indexes[0], &out_dists_sqr[0] );
                    trees3d->findNeighbors(resultSet, qp, nanoflann::SearchParams(10));
                    wasted += matchid[i] == ret_indexes[0];
                    total++;
                    matchid[i] = ret_indexes[0];
                }
                matches += xcols;

                /// Find closest point
                //#pragma omp parallel for
                for(unsigned int i=0; i< xcols; ++i) {
                    int id = matchid[i];
                    Qn.col(i) = N.col(id);
                    Qp.col(i) = Y.col(id);
                    rangeW(i) = 1.0/(1.0/SRC_INORMATION(i)+1.0/DST_INORMATION(id));
                }
            }
            addTime("matching", getTime()-matchStart);

            double last_stop3 = 1000000;
			for(int outer=0; outer< 1; ++outer) {
				if( (getTime()-start) > maxtime ){timestopped = true; break;}


				double computeModelStart = getTime();

				/// Compute weights
				switch(type) {
				case PointToPoint:	{residuals = X-Qp;} 						break;
				case PointToPlane:	{
					residuals		= Eigen::MatrixXd::Zero(1,	xcols);
					for(unsigned int i=0; i<xcols; ++i) {
						float dx = X(0,i)-Qp(0,i);
						float dy = X(1,i)-Qp(1,i);
						float dz = X(2,i)-Qp(2,i);
						float qx = Qn(0,i);
						float qy = Qn(1,i);
						float qz = Qn(2,i);
						float di = qx*dx + qy*dy + qz*dz;
                        residuals(0,i) = di*rangeW(i);
					}
				}break;
				default:			{printf("type not set\n");}					break;
				}
                //for(unsigned int i=0; i<xcols; ++i) {residuals.col(i) *= rangeW(i);}
				switch(type) {
				case PointToPoint:	{func->computeModel(residuals);} 	break;
				case PointToPlane:	{func->computeModel(residuals);}	break;
				default:  			{printf("type not set\n");} break;
				}

				addTime("computeModel", getTime()-computeModelStart);

                double last_stop2 = 1000000;
				for(int rematching2=0; rematching2 < 3; ++rematching2) {
					if( (getTime()-start) > maxtime ){timestopped = true; break;}

                    //printf("RegistrationRefinement::funcupdate: %i rematching: %i outer: %i\n",funcupdate,rematching,outer);

					matchStart = getTime();

					if(rematching2 != 0){
						for(unsigned int i=0; i< xcols; ++i) {
							std::vector<size_t>   ret_indexes(1);
							std::vector<double> out_dists_sqr(1);
							nanoflann::KNNResultSet<double> resultSet(1);

							double * qp = X.col(i).data();
							resultSet.init(&ret_indexes[0], &out_dists_sqr[0] );
							trees3d->findNeighbors(resultSet, qp, nanoflann::SearchParams(10));
							matchid[i] = ret_indexes[0];
						}
						matches += xcols;

						/// Find closest point
						for(unsigned int i=0; i< xcols; ++i) {
							int id = matchid[i];
							Qn.col(i) = N.col(id);
							Qp.col(i) = Y.col(id);
							rangeW(i) = 1.0/(1.0/SRC_INORMATION(i)+1.0/DST_INORMATION(id));
						}
					}

					addTime("matching", getTime()-computeModelStart);


                    double last_stop1 = 1000000;
                    for(int inner=0; inner< 10; ++inner) {
						if( (getTime()-start) > maxtime ){timestopped = true; break;}

						double optimizeStart = getTime();
						if(inner != 0){
							switch(type) {
							case PointToPoint:	{residuals = X-Qp;} 						break;
							case PointToPlane:	{
								residuals		= Eigen::MatrixXd::Zero(1,	xcols);
								for(unsigned int i=0; i< xcols; ++i) {
									float dx = X(0,i)-Qp(0,i);
									float dy = X(1,i)-Qp(1,i);
									float dz = X(2,i)-Qp(2,i);
									float qx = Qn(0,i);
									float qy = Qn(1,i);
									float qz = Qn(2,i);
									float di = qx*dx + qy*dy + qz*dz;
									residuals(0,i) = di;
								}
							}break;
							default:			{printf("type not set\n");}					break;
							}
                            for(unsigned int i=0; i<xcols; ++i) {residuals.col(i) *= rangeW(i);}
						}

						switch(type) {
						case PointToPoint:	{W = func->getProbs(residuals); } 					break;
						case PointToPlane:	{
							W = func->getProbs(residuals);
							for(unsigned int i=0; i<xcols; ++i) {
								W(i) = W(i)*float((Xn(0,i)*Qn(0,i) + Xn(1,i)*Qn(1,i) + Xn(2,i)*Qn(2,i)) > 0.0);
							}
						}	break;
						default:			{printf("type not set\n");} break;
						}
						Wold = W;

						//Normalizing weights has an effect simmilar to one to one matching
						//in that it reduces the effect of border points
						if(normalize_matchweights){
							for(unsigned int i=0; i < ycols; ++i)	{	total_dweight[i] = 99990.0000001;}//Reset to small number to avoid division by zero
							for(unsigned int i=0; i < xcols; ++i)	{	total_dweight[matchid[i]] = std::min(total_dweight[matchid[i]],residuals.col(i).norm());}
							for(unsigned int i=0; i < xcols; ++i)	{
								W(i) = (total_dweight[matchid[i]] == residuals.col(i).norm());//W(i) = W(i)*(W(i)/total_dweight[matchid[i]]);
							}
						}

						W = W.array()*rangeW.array()*rangeW.array();
                        double wsum = W.sum();

						if(visualizationLvl == 3){
							show(X,Y,false);
						}else if(visualizationLvl >= 4){

                            Eigen::VectorXd  Wold2 = func->getProbs(residuals);

							unsigned int s_nr_data = X.cols();
							unsigned int d_nr_data = Y.cols();

//                            for(unsigned int i = 0; i < s_nr_data; i++){
//                                if(Wold2(i) < 0.5){
//                                   printf("ID: %6.6i -> R:%5.5f -> W:%5.5f\n",i,residuals(i),Wold2(i));
//                                }
//                            }


							viewer->removeAllPointClouds();
							pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr scloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
							pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr dcloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
							scloud->points.clear();
							dcloud->points.clear();
                            for(unsigned int i = 0; i < s_nr_data; i++){pcl::PointXYZRGBNormal p;p.x = X(0,i);p.y = X(1,i);p.z = X(2,i);p.b = 255.0*Wold2(i);p.g = 255.0*Wold2(i);p.r = 255.0*Wold2(i);scloud->points.push_back(p);}
							for(unsigned int i = 0; i < d_nr_data; i++){pcl::PointXYZRGBNormal p;p.x = Y(0,i);p.y = Y(1,i);p.z = Y(2,i);p.b = 0;			p.g = 0;			p.r = 255;			dcloud->points.push_back(p);}
							viewer->addPointCloud<pcl::PointXYZRGBNormal> (scloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(scloud), "scloud");
							viewer->addPointCloud<pcl::PointXYZRGBNormal> (dcloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(dcloud), "dcloud");
							viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "scloud");
							viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "dcloud");
							if(visualizationLvl == 4){	viewer->spinOnce();}
							else{						viewer->spin();}
							viewer->removeAllPointClouds();
						}

						switch(type) {
						case PointToPoint:	{
							pcl::TransformationFromCorrespondences tfc1;
							for(unsigned int c = 0; c < X.cols(); c++){tfc1.add(Eigen::Vector3f(X(0,c), X(1,c),X(2,c)),Eigen::Vector3f(Qp(0,c),Qp(1,c),Qp(2,c)),W(c));}
							Eigen::Affine3d rot = tfc1.getTransformation().cast<double>();
							X = rot*X;
							Xn = rot.rotation()*Xn;
						}break;
						case PointToPlane:	{point_to_plane2(X, Xn, Qp, Qn, W);}break;
						default:  			{printf("type not set\n"); } break;
						}

                        stop = convergence*func->getNoise();
                        score = Wold.sum();///(func->getNoise()*float(xcols));

						addTime("optimize", getTime()-optimizeStart);



						double stop1 = (X-Xo1).colwise().norm().mean();
						Xo1 = X;

                        //if(visualizationLvl >= 1){printf("Inner: %i -> stop1: %15.15f stop: %15.15f\n",inner,stop1,stop);}
                        if(stop1 < stop) {break;}

                        if(fabs(last_stop1-stop1) <= stop1*1){break;} last_stop1 = stop1;
					}
					double stop2 = (X-Xo2).colwise().norm().mean();
					Xo2 = X;
					if(stop2 < stop) break;

//                    if(last_stop2 <= stop2*1.00001){break;}
//                    last_stop2 = stop2;
				}
				double stop3 = (X-Xo3).colwise().norm().mean();
                Xo3 = X;
				if(stop3 < stop) break;

//                if(last_stop3 <= stop3*1.00001){break;}
//                last_stop3 = stop3;
			}
			double stop4 = (X-Xo4).colwise().norm().mean();
            if(visualizationLvl >= 1){printf("stop4: %8.8f stop: %8.8f last_stop4: %8.8f\n",stop4,stop,last_stop4);}
            if(funcupdate==0 || rematching > 0){ Xo4 = X; }
            if(stop4 < stop) { break; } //if(visualizationLvl >= 1){printf("break from conv threshold\n"); }


            if(fabs(last_stop4-stop4) <= stop4*0.001){break;} last_stop4 = stop4;

//            if(last_stop4 <= stop4*1.000001){
//                if(visualizationLvl >= 1){printf("break from no change");}
//                break;
//            }
//            last_stop4 = stop4;
		}


        //for(unsigned int i=0; i<xcols; i+=10) {printf("%4.4f ",residuals(0,i));} printf("\n");

		double funcUpdateStart = getTime();
		double noise_before = func->getNoise();
        //func->debugg_print = true;
        func->update();
        //func->debugg_print = false;
		double noise_after = func->getNoise();
        //printf("before: %f after: %f\n",noise_before,noise_after);

		addTime("funcUpdate", getTime()-funcUpdateStart);
		if(fabs(1.0 - noise_after/noise_before) < 0.01){break;}
	}
	//printf("------------\n");

	double cleanupStart = getTime();

    if(visualizationLvl == 2 || visualizationLvl == 3){printf("after noise: %f\n",func->getNoise()); show(X,Y);}
	if(visualizationLvl >= 4){
		printf("visualizationLvl: %i\n",visualizationLvl);
		unsigned int s_nr_data = X.cols();
		unsigned int d_nr_data = Y.cols();
		viewer->removeAllPointClouds();
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr scloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr dcloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);

		scloud->points.clear();
		dcloud->points.clear();
		for(unsigned int i = 0; i < s_nr_data; i++){pcl::PointXYZRGBNormal p;p.x = X(0,i);p.y = X(1,i);p.z = X(2,i);p.b = 255.0*Wold(i);p.g = 255.0*Wold(i);p.r = 255.0*Wold(i);scloud->points.push_back(p);}
		for(unsigned int i = 0; i < d_nr_data; i++){pcl::PointXYZRGBNormal p;p.x = Y(0,i);p.y = Y(1,i);p.z = Y(2,i);p.b = 0;			p.g = 0;			p.r = 255;			dcloud->points.push_back(p);}
		viewer->addPointCloud<pcl::PointXYZRGBNormal> (scloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(scloud), "scloud");
		viewer->addPointCloud<pcl::PointXYZRGBNormal> (dcloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(dcloud), "dcloud");
		viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "scloud");
		viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "dcloud");
		viewer->spin();
		viewer->removeAllPointClouds();
	}


	pcl::TransformationFromCorrespondences tfc;
	tfc.reset();
	for(unsigned int i = 0; i < xcols; i++){
        //tfc.add(Eigen::Vector3f(src[i*stepx].x,	src[i*stepx].y,	src[i*stepx].z),Eigen::Vector3f (X(0,i),X(1,i),	X(2,i)));
        tfc.add(Eigen::Vector3f(src[inds[i]].x,	src[inds[i]].y,	src[inds[i]].z),Eigen::Vector3f (X(0,i),X(1,i),	X(2,i)));
	}
	guess = tfc.getTransformation().matrix().cast<double>();

	if(func != 0){delete func; func = 0;}

	FusionResults fr = FusionResults(guess,score);
	fr.timeout = timestopped;
	fr.stop = stop;

	addTime("cleanup", getTime()-cleanupStart);
	//printf("ratio: %5.5f\n",double(wasted)/double(total));

	return fr;

}

}
