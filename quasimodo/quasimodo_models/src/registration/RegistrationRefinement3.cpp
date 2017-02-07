#include "registration/RegistrationRefinement3.h"
#include <iostream>
#include <fstream>

namespace reglib
{

RegistrationRefinement3Instance::RegistrationRefinement3Instance( RegistrationRefinement3 * ref, Eigen::MatrixXd guess){
    refinement = ref;

    double timeStart;
    if(refinement->use_timer){ timeStart = getTime(); }
    result = guess;
    lastRematch = result;
    lastModel = result;

    useKeyPoints = refinement->useKeyPoints;
    useSurfacePoints = refinement->useSurfacePoints;

    surface_func = refinement->func->clone();
    kp_func = refinement->func->clone();
    //kp_func->debugg_print = true;

    surface_func->reset();
    kp_func->reset();


    capacity  = 1;
    ret_indexes = new size_t[capacity];
    out_dists_sqr = new double[capacity];

    meandist = 0;
    double meandist_wsum = 0;

    timestopped = false;

    stop		= 0.00001;
    stopD2      = 100000;
    lastModelNoise = 99999999999999999999999999;

    double m00 = result(0,0); double m01 = result(0,1); double m02 = result(0,2); double m03 = result(0,3);
    double m10 = result(1,0); double m11 = result(1,1); double m12 = result(1,2); double m13 = result(1,3);
    double m20 = result(2,0); double m21 = result(2,1); double m22 = result(2,2); double m23 = result(2,3);

//SURFACE

    if(useSurfacePoints){
        std::vector<superpoint> & surface_src = refinement->src;

        unsigned int surface_s_nr_data = surface_src.size();
        std::vector<int> surface_inds;
        surface_inds.resize(surface_s_nr_data);
        for(unsigned int i = 0; i < surface_s_nr_data; i++){surface_inds[i] = i;}
        for(unsigned int i = 0; i < surface_s_nr_data; i++){
            int rind = rand()%surface_s_nr_data;
            int tmp = surface_inds[i];
            surface_inds[i] = surface_inds[rind];
            surface_inds[rind] = tmp;
        }

        surface_s_nrp      = std::min(int(surface_s_nr_data),int(refinement->target_points));
        if(surface_s_nrp > 0){
            surface_sp         = new double        [3*surface_s_nrp];
            surface_sn         = new double        [3*surface_s_nrp];
            surface_tsp         = new double       [3*surface_s_nrp];
            surface_tsn         = new double       [3*surface_s_nrp];
            surface_si         = new double        [  surface_s_nrp];
            surface_matches    = new unsigned int  [  surface_s_nrp];
            surface_rangew     = new double        [  surface_s_nrp];
            surface_valid      = new double        [  surface_s_nrp];
            surface_Qp         = new double        [3*surface_s_nrp];
            surface_Qn         = new double        [3*surface_s_nrp];

            for(unsigned int i = 0; i < surface_s_nrp; i++){
                superpoint & p = surface_src[surface_inds[i]];
                double x = p.x;
                double y = p.y;
                double z = p.z;
                double xn = p.nx;
                double yn = p.ny;
                double zn = p.nz;
                surface_sp[3*i+0] = x;
                surface_sp[3*i+1] = y;
                surface_sp[3*i+2] = z;
                surface_sn[3*i+0] = xn;
                surface_sn[3*i+1] = yn;
                surface_sn[3*i+2] = zn;

                surface_tsp[3*i+0] = m00*x  + m01*y  + m02*z + m03;
                surface_tsp[3*i+1] = m10*x  + m11*y  + m12*z + m13;
                surface_tsp[3*i+2] = m20*x  + m21*y  + m22*z + m23;
                surface_tsn[3*i+0] = m00*xn + m01*yn + m02*zn;
                surface_tsn[3*i+1] = m10*xn + m11*yn + m12*zn;
                surface_tsn[3*i+2] = m20*xn + m21*yn + m22*zn;
                surface_si[i]     = p.point_information;
                surface_rangew[i] = 0;
                surface_valid[i]  = true;
                meandist_wsum += p.point_information;
                meandist += sqrt(x*x+y*y+z*z)*p.point_information;
            }

            surface_residuals   = 0;
            switch(refinement->type) {
                case PointToPoint:	{
                    surface_residuals   = new double[3*surface_s_nrp];
                } 						break;
                case PointToPlane:	{
                    surface_residuals   = new double[surface_s_nrp];
                }break;
                default:			{printf("type not set\n");}					break;
            }

            surface_d_nrp = refinement->d_nrp;
            surface_dp = refinement->dp;
            surface_dn = refinement->dn;
            surface_di = refinement->di;
            surface_dvalid = refinement->dvalid;
            surface_trees3d = refinement->trees3d;
        }
    }else{
        surface_s_nrp = 0;
    }


//KEYPOINTS
    if(useKeyPoints){
    std::vector<KeyPoint> & src_kp = refinement->src_kp;
        kp_s_nrp   = src_kp.size();
        if(kp_s_nrp > 0){
            kp_sp           = new double        [3*kp_s_nrp];
            kp_tsp          = new double        [3*kp_s_nrp];
            kp_si           = new double        [  kp_s_nrp];
            kp_matches      = new unsigned int  [  kp_s_nrp];
            kp_rangew       = new double        [  kp_s_nrp];
            kp_valid        = new double        [  kp_s_nrp];
            kp_Qp           = new double        [3*kp_s_nrp];
            kp_residuals    = new double        [3*kp_s_nrp];

            for(unsigned int i = 0; i < kp_s_nrp; i++){
                superpoint & p = src_kp[i].point;
                double x = p.x;
                double y = p.y;
                double z = p.z;
                kp_sp[3*i+0] = x;
                kp_sp[3*i+1] = y;
                kp_sp[3*i+2] = z;
                kp_tsp[3*i+0] = m00*x  + m01*y  + m02*z + m03;
                kp_tsp[3*i+1] = m10*x  + m11*y  + m12*z + m13;
                kp_tsp[3*i+2] = m20*x  + m21*y  + m22*z + m23;

                kp_si[i]     = p.point_information;
                kp_rangew[i] = 0;
                kp_valid[i]  = true;
                meandist_wsum += p.point_information;
                meandist += sqrt(x*x+y*y+z*z)*p.point_information;
            }

            kp_d_nrp    = refinement->kp_d_nrp;
            kp_dp       = refinement->kp_dp;
            kp_di       = refinement->kp_di;
        }
    }else{
        kp_s_nrp = 0;
    }

    meandist /= meandist_wsum;

    if(refinement->use_timer){ refinement->addTime("init", getTime()-timeStart); }
}
void RegistrationRefinement3Instance::transform(Eigen::Matrix4d p){
    const double & m00 = p(0,0); const double & m01 = p(0,1); const double & m02 = p(0,2); const double & m03 = p(0,3);
    const double & m10 = p(1,0); const double & m11 = p(1,1); const double & m12 = p(1,2); const double & m13 = p(1,3);
    const double & m20 = p(2,0); const double & m21 = p(2,1); const double & m22 = p(2,2); const double & m23 = p(2,3);
    for(unsigned int i = 0; i < surface_s_nrp; i++){
        double x  = surface_sp[3*i+0];
        double y  = surface_sp[3*i+1];
        double z  = surface_sp[3*i+2];
        double xn = surface_sn[3*i+0];
        double yn = surface_sn[3*i+1];
        double zn = surface_sn[3*i+2];
        surface_tsp[3*i+0] = m00*x + m01*y + m02*z + m03;
        surface_tsp[3*i+1] = m10*x + m11*y + m12*z + m13;
        surface_tsp[3*i+2] = m20*x + m21*y + m22*z + m23;
        surface_tsn[3*i+0] = m00*xn + m01*yn + m02*zn;
        surface_tsn[3*i+1] = m10*xn + m11*yn + m12*zn;
        surface_tsn[3*i+2] = m20*xn + m21*yn + m22*zn;
    }

    for(unsigned int i = 0; i < kp_s_nrp; i++){
        double x  = kp_sp[3*i+0];
        double y  = kp_sp[3*i+1];
        double z  = kp_sp[3*i+2];
        kp_tsp[3*i+0] = m00*x + m01*y + m02*z + m03;
        kp_tsp[3*i+1] = m10*x + m11*y + m12*z + m13;
        kp_tsp[3*i+2] = m20*x + m21*y + m22*z + m23;
    }
}

void RegistrationRefinement3Instance::computeResiduals(){
    double computeResidualsStart;
    if(refinement->use_timer){ computeResidualsStart = getTime(); }
    transform(result);
    switch(refinement->type) {
        case PointToPoint:	{
            for(unsigned int i=0; i<surface_s_nrp; ++i) {
                double rw = surface_rangew[i];
                surface_residuals[3*i+0] = rw*(surface_tsp[3*i+0]-surface_Qp[3*i+0]);
                surface_residuals[3*i+1] = rw*(surface_tsp[3*i+1]-surface_Qp[3*i+1]);
                surface_residuals[3*i+2] = rw*(surface_tsp[3*i+2]-surface_Qp[3*i+2]);
            }
        }break;
        case PointToPlane:	{
            for(unsigned int i=0; i<surface_s_nrp; ++i) {
                float dx = surface_tsp[3*i+0]-surface_Qp[3*i+0];
                float dy = surface_tsp[3*i+1]-surface_Qp[3*i+1];
                float dz = surface_tsp[3*i+2]-surface_Qp[3*i+2];
                float qx = surface_Qn[3*i+0];
                float qy = surface_Qn[3*i+1];
                float qz = surface_Qn[3*i+2];
                float di = qx*dx + qy*dy + qz*dz;
                surface_residuals[i] = di*surface_rangew[i];
            }
        }break;
        default:			{printf("type not set\n");}					break;
    }

    nr_kp_residuals = 0;
    for(unsigned int i=0; i<kp_s_nrp; ++i) {
        if(!kp_valid[i]){continue;}
        double rw = kp_rangew[i];
        double r0 = rw*(kp_tsp[3*i+0]-kp_Qp[3*i+0]);
        double r1 = rw*(kp_tsp[3*i+1]-kp_Qp[3*i+1]);
        double r2 = rw*(kp_tsp[3*i+2]-kp_Qp[3*i+2]);
        kp_residuals[3*nr_kp_residuals+0] = r0;
        kp_residuals[3*nr_kp_residuals+1] = r1;
        kp_residuals[3*nr_kp_residuals+2] = r2;
        nr_kp_residuals++;
    }
    if(refinement->use_timer){ refinement->addTime("computeResiduals", getTime()-computeResidualsStart); }
}

int RegistrationRefinement3Instance::rematch(bool force){
    if(!force && refinement->getChange(lastRematch,result,meandist) < stop){return -1;}

    double matchStart;
    if(refinement->use_timer){ matchStart = getTime(); }

    #pragma omp parallel for num_threads(11)
    for(unsigned int i=0; i< surface_s_nrp; ++i) {

        nanoflann::KNNResultSet<double> resultSet(1);
        size_t ret_indexes_test;
        double out_dists_sqr_test;
        //resultSet.init(ret_indexes, out_dists_sqr);
        resultSet.init(&ret_indexes_test, &out_dists_sqr_test);
        surface_trees3d->findNeighbors(resultSet, surface_tsp + 3*i, nanoflann::SearchParams(10));
        //int id = ret_indexes[0];
        int id = ret_indexes_test;
        surface_matches[i]      = id;
        //printf("%i -> %i\n",i,id);
        surface_Qp[3*i+0]       = surface_dp[3*id+0];
        surface_Qp[3*i+1]       = surface_dp[3*id+1];
        surface_Qp[3*i+2]       = surface_dp[3*id+2];
        surface_Qn[3*i+0]       = surface_dn[3*id+0];
        surface_Qn[3*i+1]       = surface_dn[3*id+1];
        surface_Qn[3*i+2]       = surface_dn[3*id+2];
        surface_rangew[i]       = sqrt( 1.0/(1.0/surface_si[i]+1.0/surface_di[id]) );
        surface_valid[i]        = surface_dvalid[id];
    }

    std::vector<int> & kpmatches = refinement->keypoint_matches;
    for(unsigned int i=0; i< kp_s_nrp; ++i) {
        int id    = kpmatches[i];
        kp_valid[i]        = id >= 0;
        kp_matches[i]      = id;
        if(id < 0){continue;}
        kp_Qp[3*i+0]       = kp_dp[3*id+0];
        kp_Qp[3*i+1]       = kp_dp[3*id+1];
        kp_Qp[3*i+2]       = kp_dp[3*id+2];
        kp_rangew[i]       = sqrt( 1.0/(1.0/kp_si[i]+1.0/kp_di[id]) );
    }
    lastRematch = result;

    if(refinement->use_timer){ refinement->addTime("matching", getTime()-matchStart); }
    return 0;
}

int RegistrationRefinement3Instance::model(bool force){
    //double noise_current = func->getNoise();//*func->dist->getNoise()/double(func->histogram_size);

    if(!force && (refinement->getChange(lastModel,result,meandist) < stop) ){return -1;}

//    std::cout << lastModel.inverse()*result << std::endl << std::endl;
//    std::cout << result << std::endl;
//    printf("change: %5.5f >= stop: %5.5f -> ratio %f -> %f\n",refinement->getChange(lastModel,result,meandist),stop,refinement->getChange(lastModel,result,meandist)/stop,meandist);


    computeResiduals();

    double computeModelStart;
    if(refinement->use_timer){ computeModelStart = getTime(); }

    lastModelNoise = 99999999;

    if(useSurfacePoints){
        switch(refinement->type) {
        case PointToPoint:	{surface_func->computeModel(surface_residuals,surface_s_nrp,3);} 	break;
        case PointToPlane:	{surface_func->computeModel(surface_residuals,surface_s_nrp,1);}	break;
        default:  			{printf("type not set\n");} break;
        }
        lastModelNoise = std::min(lastModelNoise,surface_func->getNoise());

        double maxd2 = 15;
        stopD2 = (maxd2 * lastModelNoise) * (maxd2 * lastModelNoise);
    }

    if(useKeyPoints){
        kp_func->computeModel(kp_residuals,nr_kp_residuals,3);
        lastModelNoise = std::min(lastModelNoise,kp_func->getNoise());
    }

    lastModel = result;
    stop = refinement->convergence*lastModelNoise;

    //printf("surface_func->getNoise(): %f kp_func->getNoise() %f\n",surface_func->getNoise(),kp_func->getNoise());

    if(refinement->use_timer){refinement->addTime("computeModel", getTime()-computeModelStart); }
    return 0;
}

int RegistrationRefinement3Instance::refine(){
    //Eigen::Affine3d p = Eigen::Affine3d::Identity();
    double innerStart;
    if(refinement->use_timer){ innerStart = getTime(); }

    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    typedef Eigen::Matrix<double, 6, 6> Matrix6d;

    Matrix6d ATA;
    Vector6d ATb;

    double surface_info = pow(surface_func->getNoise(),-2);
    double kp_info = pow(kp_func->getNoise(),-2);

    for(long inner=0; inner < 30 ; ++inner) {
        score = 0;

        Eigen::Matrix4d p = result;

        ATA.setZero ();
        ATb.setZero ();
        const double & m00 = p(0,0); const double & m01 = p(0,1); const double & m02 = p(0,2); const double & m03 = p(0,3);
        const double & m10 = p(1,0); const double & m11 = p(1,1); const double & m12 = p(1,2); const double & m13 = p(1,3);
        const double & m20 = p(2,0); const double & m21 = p(2,1); const double & m22 = p(2,2); const double & m23 = p(2,3);

        std::vector<double> s_weights;
        s_weights.resize(surface_s_nrp);
        double maxv = 0;

        for(unsigned int i = 0; i < surface_s_nrp; ++i) {
            s_weights[i] = 0;
            if(!surface_valid[i]){continue;}

            const double & rw = surface_rangew[i];

            const double & src_x = surface_sp[3*i+0];
            const double & src_y = surface_sp[3*i+1];
            const double & src_z = surface_sp[3*i+2];
            const double & src_nx = surface_sn[3*i+0];
            const double & src_ny = surface_sn[3*i+1];
            const double & src_nz = surface_sn[3*i+2];

            const double & dx = surface_Qp[3*i+0];
            const double & dy = surface_Qp[3*i+1];
            const double & dz = surface_Qp[3*i+2];
            const double & dnx = surface_Qn[3*i+0];
            const double & dny = surface_Qn[3*i+1];
            const double & dnz = surface_Qn[3*i+2];

            const double & sx = m00*src_x + m01*src_y + m02*src_z + m03;
            const double & sy = m10*src_x + m11*src_y + m12*src_z + m13;
            const double & sz = m20*src_x + m21*src_y + m22*src_z + m23;

            const double & nx = m00*src_nx + m01*src_ny + m02*src_nz;
            const double & ny = m10*src_nx + m11*src_ny + m12*src_nz;
            const double & nz = m20*src_nx + m21*src_ny + m22*src_nz;

            const double & angle = nx*dnx+ny*dny+nz*dnz;
            if(angle < 0){continue;}

            double diffX = sx-dx;
            double diffY = sy-dy;
            double diffZ = sz-dz;

            double d2 = diffX*diffX + diffY*diffY + diffZ*diffZ;
            if(d2 > stopD2){continue;}

            double di = (nx*diffX + ny*diffY + nz*diffZ)*rw;

//            printf("di: %f\n",di);

            double prob = surface_func->getProb(di);
            double weight = surface_info * prob*rw*rw;
            score += prob;

            maxv = std::max(rw,maxv);
            s_weights[i] = prob;
            //continue;

            const double & a = nz*sy - ny*sz;
            const double & b = nx*sz - nz*sx;
            const double & c = ny*sx - nx*sy;

            ATA.coeffRef ( 0) += weight * a  * a;
            ATA.coeffRef ( 1) += weight * a  * b;
            ATA.coeffRef ( 2) += weight * a  * c;
            ATA.coeffRef ( 3) += weight * a  * nx;
            ATA.coeffRef ( 4) += weight * a  * ny;
            ATA.coeffRef ( 5) += weight * a  * nz;
            ATA.coeffRef ( 7) += weight * b  * b;
            ATA.coeffRef ( 8) += weight * b  * c;
            ATA.coeffRef ( 9) += weight * b  * nx;
            ATA.coeffRef (10) += weight * b  * ny;
            ATA.coeffRef (11) += weight * b  * nz;
            ATA.coeffRef (14) += weight * c  * c;
            ATA.coeffRef (15) += weight * c  * nx;
            ATA.coeffRef (16) += weight * c  * ny;
            ATA.coeffRef (17) += weight * c  * nz;
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

        std::vector<double> weights;
        weights.resize(kp_s_nrp);

        double wsum = 0;
        double wsx = 0;
        double wsy = 0;
        double wsz = 0;
        for(unsigned int i=0; i < kp_s_nrp; ++i) {
            weights[i] = 0;
            if(!kp_valid[i]){continue;}

            const double & rw = kp_rangew[i];

            const double & src_x = kp_sp[3*i+0];
            const double & src_y = kp_sp[3*i+1];
            const double & src_z = kp_sp[3*i+2];

            const double & dx = kp_Qp[3*i+0];
            const double & dy = kp_Qp[3*i+1];
            const double & dz = kp_Qp[3*i+2];

            const double & sx = m00*src_x + m01*src_y + m02*src_z + m03;
            const double & sy = m10*src_x + m11*src_y + m12*src_z + m13;
            const double & sz = m20*src_x + m21*src_y + m22*src_z + m23;

            const double diffX = dx-sx;
            const double diffY = dy-sy;
            const double diffZ = dz-sz;

            const double de = sqrt((diffX*diffX+diffY*diffY+diffZ*diffZ));

            double probX = kp_func->getProb(diffX*rw);
            double probY = kp_func->getProb(diffY*rw);
            double probZ = kp_func->getProb(diffZ*rw);
            double prob = probX*probY*probZ/(probX*probY*probZ + (1-probX)*(1-probY)*(1-probZ));
            //double prob = kp_func->getProb(de*rw);
//if(refinement->visualizationLvl == 6 && inner == 0){
//printf("%4.4i -> diff: [%5.5f %5.5f %5.5f] -> de: %5.5f ",i,fabs(diffX),fabs(diffY),fabs(diffZ),de);
//printf("-> diff*rw: [%5.5f %5.5f %5.5f] -> de*rw: %5.5f ",fabs(diffX*rw),fabs(diffY*rw),fabs(diffZ*rw),de*rw);
//printf("->prob: [%5.5f %5.5f %5.5f] ->prob de: %5.5f -> prob full: %5.5f noise: %5.5f\n",probX,probY,probZ,func->getProb(de*rw),prob,func->getNoise());
//}
           // weights[i] = prob;//func->getProb(de*rw);
            //continue;
            if(prob < 0.000001){continue;}





            //double weight = kpInfo*prob*rw*rw;
            double weight = kp_info*prob*rw*rw;

//            if(prob > 0.5 && de > 1.0){
//                printf("----> id: %i : de:%f prob: %f -> src: %f dst: %f rw: %f weight: %f wde: %f \n",i,de,prob,src_z,dz,rw,weight,weight*de*de);
//            }else{
//                printf("id: %i : de:%f prob: %f -> src: %f dst: %f rw: %f weight: %f wde: %f\n",i,de,prob,src_z,dz,rw,weight,weight*de*de);
//            }

            weights[i] = weight;

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

            ATb.coeffRef (0) += weight * (sy*diffZ -sz*diffY);
            ATb.coeffRef (1) += weight * (-sx*diffZ + sz*diffX);
            ATb.coeffRef (2) += weight * (sx*diffY -sy*diffX);
            ATb.coeffRef (3) += weight * diffX;
            ATb.coeffRef (4) += weight * diffY;
            ATb.coeffRef (5) += weight * diffZ;
        }

        if((refinement->visualizationLvl == 6 && inner == 0) || refinement->visualizationLvl == 7){
            double mw = 0;
            for(unsigned int i = 0; i < weights.size(); i++){mw = std::max(mw,weights[i]);}
            for(unsigned int i = 0; i < weights.size(); i++){weights[i] /= mw;}
            //for(unsigned int i = 0; i < weights.size(); i++){printf("%3.3i -> %f\n",i,weights[i]);}
            //printf("refine: %i\n",inner);
            refinement->show(refinement->src_kp,refinement->dst_kp,result,refinement->keypoint_matches,weights,false);
            //refinement->show(refinement->src,refinement->dst,result,s_weights,true);
            refinement->show(surface_sp,surface_s_nrp,surface_dp,surface_d_nrp,result,true,s_weights);
        }

        ATA.coeffRef (4)  -= wsz;
        ATA.coeffRef (9)  += wsz;
        ATA.coeffRef (5)  += wsy;
        ATA.coeffRef (15) -= wsy;
        ATA.coeffRef (11) -= wsx;
        ATA.coeffRef (16) += wsx;
        ATA.coeffRef (21) += wsum;
        ATA.coeffRef (28) += wsum;
        ATA.coeffRef (35) += wsum;

        ATA.coeffRef (6)  = ATA.coeff (1);
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

        for(long d = 0; d < 6; d++){ATA(d,d) += 0.000000001;}

        Vector6d x = static_cast<Vector6d> (ATA.inverse () * ATb);
        Eigen::Affine3d transformation = Eigen::Affine3d(constructTransformationMatrix(x(0,0),x(1,0),x(2,0),x(3,0),x(4,0),x(5,0)));

        result = transformation*result;

        if(refinement->getChange(transformation.matrix(),meandist) < stop){ break; }
    }

//    result = p.matrix() * result;
    transform(result);

    if(refinement->use_timer){ refinement->addTime("inner", getTime()-innerStart); }
    //if(refinement->visualizationLvl == 4){refinement->show(sp, s_nrp, dp, d_nrp);}
    return 0;
}

int RegistrationRefinement3Instance::update(){
    double funcUpdateStart;
    if(refinement->use_timer){ funcUpdateStart = getTime(); }

    bool surface_converged = true;
    if(useSurfacePoints){
        double surface_noise_before = surface_func->getNoise();
        surface_func->update();
        double surface_noise_after = surface_func->getNoise();
        surface_converged = fabs(1.0 - surface_noise_after/surface_noise_before) < 0.01;

        //printf("surface : before %7.7f after %7.7f -> %i ",surface_noise_before,surface_noise_after,surface_converged);
    }

    bool kp_converged = true;
    if(useKeyPoints){
        double kp_noise_before = kp_func->getNoise();
        kp_func->update();
        double kp_noise_after = kp_func->getNoise();
        kp_converged = fabs(1.0 - kp_noise_after/kp_noise_before) < 0.01;

        //printf("keypoint : before %7.7f after %7.7f -> %i",kp_noise_before,kp_noise_after,kp_converged);
    }
    //printf("\n");

    if(refinement->use_timer){refinement->addTime("funcUpdate", getTime()-funcUpdateStart);}

    if(kp_converged && surface_converged){return -1;}

    model(true);
    refine();

    return 0;
}

FusionResults RegistrationRefinement3Instance::getTransform(){
    if(refinement->visualizationLvl > 3){
        refinement->show(refinement->src,refinement->dst,result);
        refinement->show(refinement->src_kp,refinement->dst_kp,result);
    }
    if(refinement->visualizationLvl == 2){refinement->show(refinement->src,refinement->dst,result);}


    //return FusionResults(result,score);

    //printf("--------------------------------------\n");
    rematch(true);
    model(true);
    refine();

    //STATEMACHINE V1.0
    double maxtime = refinement->maxtime;
    double start = getTime();
    enum RegistrationState { modelstate, rematchstate, updatestate, finishedstate };
    RegistrationState state = modelstate;
    bool go = true;
    int remodels = 0;
    int rematches = 0;
    int updates = 0;
//    printf("STATEMACHINE V1.0\n");
//    while(go){
//        if( (getTime()-start) > maxtime ){timestopped = true; break;}
//        switch(state) {
//            case modelstate:	{
//                if(remodels > 5){state = rematchstate; remodels = 0; break;}
//                remodels++;
//                if(model() != -1){refine();}
//                else{state = rematchstate;}
//            } 	break;
//            case rematchstate:	{
//                if(rematches > 10){state = updatestate; rematches = 0; break;}
//                rematches++;
//                if(rematch() != -1){
//                    refine();
//                    state = modelstate;
//                }else{
//                    state = updatestate;
//                }
//            } 	break;
//            case updatestate:	{
//                if(updates > 50){
//                    updates = 0;
//                    state = finishedstate;
//                    go = false;
//                    break;
//                }
//                updates++;
//                if(update() != -1){
//                    state = modelstate;
//                }else{
//                    state = finishedstate;
//                    go = false;
//                }
//            } 	break;
//            default:  			{printf("state not coded\n");} break;
//        }
//    }


    int total_rematches = 0;
    int total_remodels = 0;
    int total_updates = 0;

    //printf("STATEMACHINE V2.0\n");
    while(go){
        if( (getTime()-start) > maxtime ){timestopped = true; break;}
        if(refinement->visualizationLvl == 5){
            refinement->show(refinement->src,refinement->dst,result);
            refinement->show(refinement->src_kp,refinement->dst_kp,result);
        }
        switch(state) {
            case modelstate:	{
                if(remodels > refinement->max_remodels){state = updatestate; remodels = 0; break;}
                remodels++;
                total_remodels++;
                if(model() != -1){refine(); state = rematchstate;}
                else{state = updatestate;}
            } 	break;
            case rematchstate:	{
                if(rematches > refinement->max_rematches){state = modelstate; rematches = 0; break;}
                rematches++;
                total_rematches++;
                if(rematch() != -1){refine(); }
                else{               state = modelstate;}
            } 	break;
            case updatestate:	{

                if(refinement->visualizationLvl == 4){
                    refinement->show(refinement->src,refinement->dst,result);
                    refinement->show(refinement->src_kp,refinement->dst_kp,result);
                }

                //printf("rematch: %i remodel: %i updates: %i surface: %5.5f kp: %5.5f \n",total_rematches,total_remodels,total_updates,surface_func->getNoise(),kp_func->getNoise());
                if(updates > 50){
                    updates = 0;
                    state = finishedstate;
                    go = false;
                    break;
                }
                updates++;
                total_updates++;
                if(update() != -1){
                    state = modelstate;
                }else{
                    state = finishedstate;
                    go = false;
                }
            } 	break;
            default:  			{printf("state not coded\n");} break;
        }
    }

    //printf("rematch: %i remodel: %i updates: %i\n",total_rematches,total_remodels,total_updates);


//    //printf("STATEMACHINE V3.0\n");
//    while(go){
//        if( (getTime()-start) > maxtime ){timestopped = true; break;}
//        if(refinement->visualizationLvl == 5){
//            refinement->show(refinement->src,refinement->dst,result);
//            refinement->show(refinement->src_kp,refinement->dst_kp,result);
//        }
//        switch(state) {
//            case modelstate:	{
//                if(remodels > 25){printf("remodels: %i\n",remodels); state = updatestate; remodels = 0; break;}
//                remodels++;

//                int rma1 = model();
//                refine();
//                if(rma1 == -1){
//                    printf("remodels: %i\n",remodels);
//                    state = updatestate;
//                }

//                for(int i = 0; i < 10; i++){
//                    int rma = rematch();
//                    refine();
//                    if(rma == -1){break;}
//                }
//            } 	break;
//            case updatestate:	{

//                if(refinement->visualizationLvl == 4){
//                    refinement->show(refinement->src,refinement->dst,result);
//                    refinement->show(refinement->src_kp,refinement->dst_kp,result);
//                }
//                if(updates > 50){
//                    updates = 0;
//                    state = finishedstate;
//                    go = false;
//                    break;
//                }
//                updates++;
//                if(update() != -1){
//                    state = modelstate;
//                }else{
//                    state = finishedstate;
//                    go = false;
//                }
//            } 	break;
//            default:  			{printf("state not coded\n");} break;
//        }
//    }

    if(refinement->visualizationLvl == 1){refinement->show(refinement->src,refinement->dst,result);}
    if(refinement->visualizationLvl == 2){refinement->show(refinement->src,refinement->dst,result);}
    if(refinement->visualizationLvl == 3){refinement->show(refinement->src,refinement->dst,result);}


    FusionResults fr = FusionResults(result,score);
    fr.timeout = timestopped;
    fr.stop = stop;
    return fr;
}

RegistrationRefinement3Instance::~RegistrationRefinement3Instance(){
    double timeStart;
    if(refinement->use_timer){ timeStart = getTime(); }

    delete[] ret_indexes;
    delete[] out_dists_sqr;

    if(surface_s_nrp > 0){
        delete[] surface_sp;
        delete[] surface_sn;
        delete[] surface_tsp;
        delete[] surface_tsn;
        delete[] surface_si;
        delete[] surface_matches;
        delete[] surface_rangew;
        delete[] surface_valid;
        delete[] surface_Qp;
        delete[] surface_Qn;
        delete[] surface_residuals;
    }
    delete   surface_func;

    if(kp_s_nrp > 0){
        delete[] kp_sp;
        delete[] kp_tsp;
        delete[] kp_si;
        delete[] kp_matches;
        delete[] kp_rangew;
        delete[] kp_valid;
        delete[] kp_Qp;
        delete[] kp_residuals;
    }
    delete   kp_func;


    if(refinement->use_timer){ refinement->addTime("cleanup", getTime()-timeStart); }
}

RegistrationRefinement3::RegistrationRefinement3(DistanceWeightFunction2 * func_){

    max_rematches = 10;
    max_remodels = 10;

    useKeyPoints = true;
    useSurfacePoints = true;

    func = func_;

    d_nrp = 0;
    dp = 0;
    dn = 0;
    di = 0;
    dvalid = 0;

    kp_d_nrp = 0;
    kp_dp    = 0;
    kp_di    = 0;

    trees3d                 = 0;
    a3d                     = 0;
    only_initial_guess		= false;
    type					= PointToPlane;
    visualizationLvl        = 1;

    target_points           = 250;
    dst_points              = 2500;
    allow_regularization    = true;
    maxtime                 = 9999999;
    use_timer               = true;
    regularization = 0.1;
}

RegistrationRefinement3::~RegistrationRefinement3(){
    kp_d_nrp = 0;
    d_nrp = 0;
    if(kp_dp    != 0){delete[] kp_dp;   kp_dp = 0;}
    if(kp_di    != 0){delete[] kp_di;   kp_di = 0;}
    if(dp       != 0){delete[] dp;      dp = 0;}
    if(dn       != 0){delete[] dn;      dn = 0;}
    if(di       != 0){delete[] di;      di = 0;}
    if(dvalid   != 0){delete[] dvalid;  dvalid = 0;}
    if(trees3d  != 0){delete trees3d;   trees3d = 0;}
    if(a3d      != 0){delete a3d;       a3d = 0;}
    if(func     != 0){delete func;      func = 0;}
}

void RegistrationRefinement3::setDst(std::vector<superpoint> & dst_){
    double startTime;
    if(use_timer){ startTime = getTime(); }
    dst = dst_;
    unsigned int d_nr_data = dst.size();
    std::vector<int> inds;
    inds.resize(d_nr_data);
    for(unsigned int i = 0; i < d_nr_data; i++){inds[i] = i;}


    for(unsigned int i = 0; i < d_nr_data; i++){
        int rind = rand()%d_nr_data;
        int tmp = inds[i];
        inds[i] = inds[rind];
        inds[rind] = tmp;
    }

    d_nrp = std::min(int(d_nr_data),int(dst_points));
    if(d_nrp == 0){return ;}

    if(dp != 0){    delete[] dp;        dp = 0;}
    if(dn != 0){    delete[] dn;        dn = 0;}
    if(di != 0){    delete[] di;        di = 0;}
    if(dvalid != 0){delete[] dvalid;    dvalid = 0;}

    dp      = new double[3*d_nrp];
    dn      = new double[3*d_nrp];
    di      = new double[  d_nrp];
    dvalid  = new bool  [  d_nrp];

    for(unsigned int i = 0; i < d_nrp; i++){
        superpoint & p = dst[inds[i]];
        dp[3*i+0]   = p.x;
        dp[3*i+1]   = p.y;
        dp[3*i+2]   = p.z;
        dn[3*i+0]   = p.nx;
        dn[3*i+1]   = p.ny;
        dn[3*i+2]   = p.nz;
        di[i]       = p.point_information;
        dvalid[i]   = true;//!p.is_boundry;
    }
if(useSurfacePoints){
    if(trees3d != 0){delete trees3d;}
    if(a3d != 0){delete a3d;}

    a3d = new ArrayData3D<double>;
    a3d->data	= dp;
    a3d->rows	= d_nrp;
    trees3d	= new Tree3d(3, *a3d, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    trees3d->buildIndex();
}
    if(use_timer){ addTime("setDst", getTime()-startTime); }
}

void RegistrationRefinement3::setSrc(std::vector<superpoint> & src_){
    double startTime;
    if(use_timer){ startTime = getTime(); }
    src = src_;
    /*
    unsigned int d_nr_data = dst.size();
    std::vector<int> inds;
    inds.resize(d_nr_data);
    for(unsigned int i = 0; i < d_nr_data; i++){inds[i] = i;}


    for(unsigned int i = 0; i < d_nr_data; i++){
        int rind = rand()%d_nr_data;
        int tmp = inds[i];
        inds[i] = inds[rind];
        inds[rind] = tmp;
    }

    d_nrp = std::min(int(d_nr_data),int(dst_points));
    if(d_nrp == 0){return ;}

    if(dp != 0){delete[] dp; dp = 0;}
    if(dn != 0){delete[] dn; dn = 0;}
    if(di != 0){delete[] di; di = 0;}

    dp = new double[3*d_nrp];
    dn = new double[3*d_nrp];
    di = new double[d_nrp];

    for(unsigned int i = 0; i < d_nrp; i++){
        superpoint & p = dst[inds[i]];
        dp[3*i+0]   = p.x;
        dp[3*i+1]   = p.y;
        dp[3*i+2]   = p.z;
        dn[3*i+0]   = p.nx;
        dn[3*i+1]   = p.ny;
        dn[3*i+2]   = p.nz;
        di[i]       = p.point_information;
    }

    if(trees3d != 0){delete trees3d;}
    if(a3d != 0){delete a3d;}

    a3d = new ArrayData3D<double>;
    a3d->data	= dp;
    a3d->rows	= d_nrp;
    trees3d	= new Tree3d(3, *a3d, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    trees3d->buildIndex();
*/
    if(use_timer){ addTime("setSrcc", getTime()-startTime); }
}


void RegistrationRefinement3::setSrc(Model * src, bool recompute){
    src_kp = src->keypoints;
    setSrc(src->points);

    double startTime;
    if(use_timer){ startTime = getTime(); }
    if(recompute && useKeyPoints){
        std::vector<double> dmat;
        dmat.resize(src_kp.size()*dst_kp.size());

        std::vector<double> src_nn_dist;
        src_nn_dist.resize(src_kp.size());

        std::vector<int> src_nn_ind;
        src_nn_ind.resize(src_kp.size());

        for(unsigned int i = 0; i < src_kp.size(); i++){
            src_nn_dist[i] = 10000;
            src_nn_ind[i] = -1;
        }

        std::vector<double> dst_nn_dist;
        dst_nn_dist.resize(dst_kp.size());
        for(unsigned int i = 0; i < dst_kp.size(); i++){dst_nn_dist[i] = 10000;}

        keypoint_matches.resize(src_kp.size());
        for(unsigned int i = 0; i < src_kp.size(); i++){
            keypoint_matches[i] = -1;
            for(unsigned int j = 0; j < dst_kp.size(); j++){
                double d = src_kp[i].descriptor.distance(dst_kp[j].descriptor);
                if( d < src_nn_dist[i]){
                    src_nn_dist[i] = d;
                    src_nn_ind[i] = j;
                }

                if( d < dst_nn_dist[j]){
                    dst_nn_dist[j] = d;
                }
            }
        }

        for(unsigned int i = 0; i < src_kp.size(); i++){
            int j = src_nn_ind[i];
            if(src_nn_dist[i] < 0.10 && src_nn_dist[i] == dst_nn_dist[j]){
                keypoint_matches[i] = j;
            }
        }
    }
    if(use_timer){ addTime("featuresSetDst", getTime()-startTime); }
}
void RegistrationRefinement3::setDst(Model * dst, bool recompute){
    printf("%s\n",__PRETTY_FUNCTION__);
    dst_kp = dst->keypoints;
    setDst(dst->points);

    double startTime;
    if(use_timer){ startTime = getTime(); }

    kp_d_nrp = dst_kp.size();
    if(kp_d_nrp == 0){return ;}

    if(kp_dp != 0){delete[] kp_dp; kp_dp = 0;}
    if(kp_di != 0){delete[] kp_di; kp_di = 0;}

    kp_dp = new double[3*kp_d_nrp];
    kp_di = new double[kp_d_nrp];

    for(unsigned int i = 0; i < kp_d_nrp; i++){
        superpoint & p = dst_kp[i].point;
        kp_dp[3*i+0]   = p.x;
        kp_dp[3*i+1]   = p.y;
        kp_dp[3*i+2]   = p.z;
        kp_di[i]       = p.point_information;
    }

    if(recompute && useKeyPoints){
        std::vector<double> dmat;
        dmat.resize(src_kp.size()*dst_kp.size());

        std::vector<double> src_nn_dist;
        src_nn_dist.resize(src_kp.size());

        std::vector<int> src_nn_ind;
        src_nn_ind.resize(src_kp.size());

        for(unsigned int i = 0; i < src_kp.size(); i++){
            src_nn_dist[i] = 10000;
            src_nn_ind[i] = -1;
        }

        std::vector<double> dst_nn_dist;
        dst_nn_dist.resize(dst_kp.size());
        for(unsigned int i = 0; i < dst_kp.size(); i++){dst_nn_dist[i] = 10000;}

        keypoint_matches.resize(src_kp.size());
        for(unsigned int i = 0; i < src_kp.size(); i++){
            keypoint_matches[i] = -1;
            for(unsigned int j = 0; j < dst_kp.size(); j++){
                double d = src_kp[i].descriptor.distance(dst_kp[j].descriptor);
                if( d < src_nn_dist[i]){
                    src_nn_dist[i] = d;
                    src_nn_ind[i] = j;
                }
                if( d < dst_nn_dist[j]){ dst_nn_dist[j] = d; }
            }
        }

        for(unsigned int i = 0; i < src_kp.size(); i++){
            int j = src_nn_ind[i];
            if(src_nn_dist[i] < 0.10 && src_nn_dist[i] == dst_nn_dist[j]){keypoint_matches[i] = j;}
        }
    }
    if(use_timer){ addTime("featuresSetSrc", getTime()-startTime); }
}

double RegistrationRefinement3::getChange(Eigen::Matrix4d & change, double meandist){
    double change_t = 0;
    double change_r = 0;
    for(unsigned long k = 0; k < 3; k++){
        change_t += change(k,3)*change(k,3);
        for(unsigned long l = 0; l < 3; l++){
            if(k == l){ change_r += fabs(1-change(k,l));}
            else{		change_r += fabs(change(k,l));}
        }
    }
    change_t = sqrt(change_t);
    //printf("t: %5.5f r: %f\n",change_t,change_r);
    double total_change = change_t + meandist*change_r;
    return total_change;
}

double RegistrationRefinement3::getChange(Eigen::Matrix4d & before, Eigen::Matrix4d & after, double meandist){
    Eigen::Matrix4d change = after*before.inverse();
    return getChange(change,meandist);
}

FusionResults RegistrationRefinement3::getTransform(Eigen::MatrixXd guess){
    RegistrationRefinement3Instance ri ( this, guess);
    return ri.getTransform();
}


std::string RegistrationRefinement3::getString(){
    return func->getString();
}

}
