#include "registration/RegistrationRefinement2.h"
#include <iostream>
#include <fstream>

namespace reglib
{

RegistrationRefinement2Instance::RegistrationRefinement2Instance( RegistrationRefinement2 * ref, Eigen::MatrixXd guess){
    refinement = ref;

    double timeStart;
    if(refinement->use_timer){ timeStart = getTime(); }
    result = guess;
    lastRematch = result;
    lastModel = result;

    func                    = new DistanceWeightFunction2PPR3(new GaussianDistribution());
    if(refinement->allow_regularization){       func->startreg			= refinement->regularization;}
    else{                                       func->startreg			= 0.0;}
    func->debugg_print                                      = false;
    func->noise_min                                         = 0.0001;
    func->reset();

    stop		= 0.00001;
    stopD2      = 100000;
    lastModelNoise = 99999999999999999999999999;

    double m00 = result(0,0); double m01 = result(0,1); double m02 = result(0,2); double m03 = result(0,3);
    double m10 = result(1,0); double m11 = result(1,1); double m12 = result(1,2); double m13 = result(1,3);
    double m20 = result(2,0); double m21 = result(2,1); double m22 = result(2,2); double m23 = result(2,3);


    std::vector<superpoint> & src = refinement->src;

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

    s_nrp      = std::min(int(s_nr_data),int(refinement->target_points));
    sp         = new double        [3*s_nrp];
    sn         = new double        [3*s_nrp];
    si         = new double        [  s_nrp];
    matches    = new unsigned int  [  s_nrp];
    rangew     = new double        [  s_nrp];
    Qp         = new double        [3*s_nrp];
    Qn         = new double        [3*s_nrp];

    capacity  = 1;
    ret_indexes = new size_t[capacity];
    out_dists_sqr = new double[capacity];

    timestopped = false;

    meandist = 0;
    double meandist_wsum = 0;

    for(unsigned int i = 0; i < s_nrp; i++){
        superpoint & p = src[inds[i]];
        double x = p.x;
        double y = p.y;
        double z = p.z;
        double xn = p.nx;
        double yn = p.ny;
        double zn = p.nz;
        sp[3*i+0] = m00*x + m01*y + m02*z + m03;
        sp[3*i+1] = m10*x + m11*y + m12*z + m13;
        sp[3*i+2] = m20*x + m21*y + m22*z + m23;
        sn[3*i+0] = m00*xn + m01*yn + m02*zn;
        sn[3*i+1] = m10*xn + m11*yn + m12*zn;
        sn[3*i+2] = m20*xn + m21*yn + m22*zn;
        si[i]     = p.point_information;
        rangew[i] = 0;
        meandist_wsum += p.point_information;
        meandist += sqrt(x*x+y*y+z*z)*p.point_information;
    }
    meandist /= meandist_wsum;

    residuals   = 0;
    switch(refinement->type) {
        case PointToPoint:	{
            residuals   = new double[3*s_nrp];
        } 						break;
        case PointToPlane:	{
            residuals   = new double[s_nrp];
        }break;
        default:			{printf("type not set\n");}					break;
    }

    d_nrp = refinement->d_nrp;
    dp = refinement->dp;
    dn = refinement->dn;
    di = refinement->di;
    trees3d = refinement->trees3d;

    if(refinement->use_timer){ refinement->addTime("init", getTime()-timeStart); }
}
void RegistrationRefinement2Instance::transform(Eigen::Matrix4d p){
    const double & m00 = p(0,0); const double & m01 = p(0,1); const double & m02 = p(0,2); const double & m03 = p(0,3);
    const double & m10 = p(1,0); const double & m11 = p(1,1); const double & m12 = p(1,2); const double & m13 = p(1,3);
    const double & m20 = p(2,0); const double & m21 = p(2,1); const double & m22 = p(2,2); const double & m23 = p(2,3);
    for(unsigned int i = 0; i < s_nrp; i++){
        double x  = sp[3*i+0];
        double y  = sp[3*i+1];
        double z  = sp[3*i+2];
        double xn = sn[3*i+0];
        double yn = sn[3*i+1];
        double zn = sn[3*i+2];
        sp[3*i+0] = m00*x + m01*y + m02*z + m03;
        sp[3*i+1] = m10*x + m11*y + m12*z + m13;
        sp[3*i+2] = m20*x + m21*y + m22*z + m23;
        sn[3*i+0] = m00*xn + m01*yn + m02*zn;
        sn[3*i+1] = m10*xn + m11*yn + m12*zn;
        sn[3*i+2] = m20*xn + m21*yn + m22*zn;
    }
}
void RegistrationRefinement2Instance::computeResiduals(){
    double computeResidualsStart;
    if(refinement->use_timer){ computeResidualsStart = getTime(); }
    switch(refinement->type) {
        case PointToPoint:	{
            for(unsigned int i=0; i<s_nrp; ++i) {
                double rw = 1.0/rangew[i];
                residuals[3*i+0] = rw*(sp[3*i+0]-Qp[3*i+0]);
                residuals[3*i+1] = rw*(sp[3*i+1]-Qp[3*i+1]);
                residuals[3*i+2] = rw*(sp[3*i+2]-Qp[3*i+2]);
            }
        }break;
        case PointToPlane:	{
            for(unsigned int i=0; i<s_nrp; ++i) {
                float dx = sp[3*i+0]-Qp[3*i+0];
                float dy = sp[3*i+1]-Qp[3*i+1];
                float dz = sp[3*i+2]-Qp[3*i+2];
                float qx = Qn[3*i+0];
                float qy = Qn[3*i+1];
                float qz = Qn[3*i+2];
                float di = qx*dx + qy*dy + qz*dz;
                residuals[i] = di/rangew[i];
            }
        }break;
        default:			{printf("type not set\n");}					break;
    }
    if(refinement->use_timer){ refinement->addTime("computeResiduals", getTime()-computeResidualsStart); }
}

int RegistrationRefinement2Instance::rematch(bool force){
    if(!force && refinement->getChange(lastRematch,result,meandist) < stop){return -1;}

    double matchStart;
    if(refinement->use_timer){ matchStart = getTime(); }

    nanoflann::KNNResultSet<double> resultSet(capacity);
    for(unsigned int i=0; i< s_nrp; ++i) {
        resultSet.init(ret_indexes, out_dists_sqr);
        trees3d->findNeighbors(resultSet, sp + 3*i, nanoflann::SearchParams(10));
        unsigned int id = ret_indexes[0];
        matches[i]      = id;
        Qp[3*i+0]       = dp[3*id+0];
        Qp[3*i+1]       = dp[3*id+1];
        Qp[3*i+2]       = dp[3*id+2];
        Qn[3*i+0]       = dn[3*id+0];
        Qn[3*i+1]       = dn[3*id+1];
        Qn[3*i+2]       = dn[3*id+2];
        rangew[i]       = sqrt( 1.0/(1.0/si[i]+1.0/di[id]) );//1.0/(1.0/SRC_INORMATION(i)+1.0/DST_INORMATION(id));
    }
    lastRematch = result;

    if(refinement->use_timer){ refinement->addTime("matching", getTime()-matchStart); }
    return 0;
}

int RegistrationRefinement2Instance::model(bool force){
    double noise_current = func->getNoise()*func->dist->getNoise()/double(func->histogram_size);

    double ratio = fabs(1.0 - noise_current/lastModelNoise);
    if(!force && (refinement->getChange(lastModel,result,meandist) < stop) ){return -1;}

    computeResiduals();

    double computeModelStart;
    if(refinement->use_timer){ computeModelStart = getTime(); }

    switch(refinement->type) {
    case PointToPoint:	{func->computeModel(residuals,s_nrp,3);} 	break;
    case PointToPlane:	{func->computeModel(residuals,s_nrp,1);}	break;
    default:  			{printf("type not set\n");} break;
    }

    lastModel = result;
    lastModelNoise = func->getNoise()*func->dist->getNoise()/double(func->histogram_size);
    stop = refinement->convergence*lastModelNoise;

    double maxd2 = 10;
    stopD2 = (maxd2 * lastModelNoise) * (maxd2 * lastModelNoise);

    if(refinement->use_timer){refinement->addTime("computeModel", getTime()-computeModelStart); }
    return 0;
}

int RegistrationRefinement2Instance::refine(){
    Eigen::Affine3d p = Eigen::Affine3d::Identity();
    double innerStart;
    if(refinement->use_timer){ innerStart = getTime(); }

    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    typedef Eigen::Matrix<double, 6, 6> Matrix6d;

    Matrix6d ATA;
    Vector6d ATb;

    for(long inner=0; inner < 15 ; ++inner) {
        score = 0;

        ATA.setZero ();
        ATb.setZero ();
        const double & m00 = p(0,0); const double & m01 = p(0,1); const double & m02 = p(0,2); const double & m03 = p(0,3);
        const double & m10 = p(1,0); const double & m11 = p(1,1); const double & m12 = p(1,2); const double & m13 = p(1,3);
        const double & m20 = p(2,0); const double & m21 = p(2,1); const double & m22 = p(2,2); const double & m23 = p(2,3);

        for(unsigned int i=0; i<s_nrp; ++i) {
            const double & src_x = sp[3*i+0];
            const double & src_y = sp[3*i+1];
            const double & src_z = sp[3*i+2];
            const double & src_nx = sn[3*i+0];
            const double & src_ny = sn[3*i+1];
            const double & src_nz = sn[3*i+2];

            const double & dx = Qp[3*i+0];
            const double & dy = Qp[3*i+1];
            const double & dz = Qp[3*i+2];
            const double & dnx = Qn[3*i+0];
            const double & dny = Qn[3*i+1];
            const double & dnz = Qn[3*i+2];

            const double & rw = rangew[i];

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

            double di = (nx*diffX + ny*diffY + nz*diffZ)/rw;

            double prob = func->getProb(di);
            double weight = prob*rw*rw;
            score += prob;

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

        for(long d = 0; d < 6; d++){ATA(d,d) += 1;}

        Vector6d x = static_cast<Vector6d> (ATA.inverse () * ATb);
        Eigen::Affine3d transformation = Eigen::Affine3d(constructTransformationMatrix(x(0,0),x(1,0),x(2,0),x(3,0),x(4,0),x(5,0)));
        p = transformation*p;
        if(refinement->getChange(transformation.matrix(),meandist) < stop){break;}
    }

    result = p.matrix() * result;

    transform(p.matrix());

    if(refinement->use_timer){ refinement->addTime("inner", getTime()-innerStart); }
    //if(refinement->visualizationLvl == 4){refinement->show(sp, s_nrp, dp, d_nrp);}
    return 0;
}

int RegistrationRefinement2Instance::update(){
    double funcUpdateStart;
    if(refinement->use_timer){ funcUpdateStart = getTime(); }

    double noise_before = func->getNoise()*func->dist->getNoise()/double(func->histogram_size);
    func->update();
    double noise_after = func->getNoise()*func->dist->getNoise()/double(func->histogram_size);

    if(refinement->use_timer){refinement->addTime("funcUpdate", getTime()-funcUpdateStart);}

    if(fabs(1.0 - noise_after/noise_before) < 0.01){return -1;}

    model(true);
    refine();

    return 0;
}

FusionResults RegistrationRefinement2Instance::getTransform(){
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

    //printf("STATEMACHINE V2.0\n");
    while(go){
        if( (getTime()-start) > maxtime ){timestopped = true; break;}
        switch(state) {
            case modelstate:	{
                if(remodels > 10){state = updatestate; remodels = 0; break;}
                remodels++;
                if(model() != -1){refine(); state = rematchstate;}
                else{state = updatestate;}
            } 	break;
            case rematchstate:	{
                if(rematches > 5){state = modelstate; rematches = 0; break;}
                rematches++;
                if(rematch() != -1){
                    refine();
                }else{
                    state = modelstate;
                }
            } 	break;
            case updatestate:	{
                if(updates > 50){
                    updates = 0;
                    state = finishedstate;
                    go = false;
                    break;
                }
                updates++;
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

    if(refinement->visualizationLvl == 1){refinement->show(sp, s_nrp, dp, d_nrp,false);}
    if(refinement->visualizationLvl == 2){refinement->show(sp, s_nrp, dp, d_nrp,true);}


    FusionResults fr = FusionResults(result,score);
    fr.timeout = timestopped;
    fr.stop = stop;
    return fr;
}

RegistrationRefinement2Instance::~RegistrationRefinement2Instance(){
    double timeStart;
    if(refinement->use_timer){ timeStart = getTime(); }

    delete[] ret_indexes;
    delete[] out_dists_sqr;
    delete[] sp;
    delete[] sn;
    delete[] si;
    delete[] matches;
    delete[] rangew;
    delete[] Qp;
    delete[] Qn;
    delete[] residuals;
    delete   func;

    if(refinement->use_timer){ refinement->addTime("cleanup", getTime()-timeStart); }
}

RegistrationRefinement2::RegistrationRefinement2(){
    d_nrp = 0;
    dp = 0;
    dn = 0;
    di = 0;

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

RegistrationRefinement2::~RegistrationRefinement2(){
    d_nrp = 0;
    if(dp != 0){delete[] dp; dp = 0;}
    if(dn != 0){delete[] dn; dn = 0;}
    if(di != 0){delete[] di; di = 0;}
    if(trees3d != 0){delete trees3d; trees3d = 0;}
    if(a3d != 0){delete a3d; a3d = 0;}
}

void RegistrationRefinement2::setDst(std::vector<superpoint> & dst_){
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

    if(use_timer){ addTime("setDst", getTime()-startTime); }
}

double RegistrationRefinement2::getChange(Eigen::Matrix4d & change, double meandist){
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
    double total_change = change_t + meandist*change_r;
    //printf("change_t %5.5f change_r %5.5f -> total_change: %5.5f\n",change_t,change_r,total_change);
    return total_change;
}

double RegistrationRefinement2::getChange(Eigen::Matrix4d & before, Eigen::Matrix4d & after, double meandist){
    Eigen::Matrix4d change = after*before.inverse();
    return getChange(change,meandist);
}

FusionResults RegistrationRefinement2::getTransform(Eigen::MatrixXd guess){


    RegistrationRefinement2Instance ri ( this, guess);
    return ri.getTransform();

    double funcUpdateStart;
    double matchStart;
    double computeResidualsStart;
    double computeModelStart;
    double innerStart;
    double initStart;
    if(use_timer){ initStart = getTime(); }


    DistanceWeightFunction2PPR3 *   func                    = new DistanceWeightFunction2PPR3(new GaussianDistribution());
    if(allow_regularization){       func->startreg			= 0.1;}
    else{                           func->startreg			= 0.0;}
    func->debugg_print                                      = false;
    func->noise_min                                         = 0.0001;

    double stop		= 0.00001;

    func->reset();

    Eigen::Matrix4d result = guess;

    double m00 = result(0,0); double m01 = result(0,1); double m02 = result(0,2); double m03 = result(0,3);
    double m10 = result(1,0); double m11 = result(1,1); double m12 = result(1,2); double m13 = result(1,3);
    double m20 = result(2,0); double m21 = result(2,1); double m22 = result(2,2); double m23 = result(2,3);

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

    unsigned int s_nrp      = std::min(int(s_nr_data),int(target_points));
    double * sp             = new double        [3*s_nrp];
    double * sn             = new double        [3*s_nrp];
    double * si             = new double        [  s_nrp];
    unsigned int * matches  = new unsigned int  [  s_nrp];
    double * rangew         = new double        [  s_nrp];


    double * Qp             = new double        [3*s_nrp];
    double * Qn             = new double        [3*s_nrp];

    double meandist = 0;
    double meandist_wsum = 0;

    for(unsigned int i = 0; i < s_nrp; i++){
        superpoint & p = src[inds[i]];
        double x = p.x;
        double y = p.y;
        double z = p.z;
        double xn = p.nx;
        double yn = p.ny;
        double zn = p.nz;
        sp[3*i+0] = m00*x + m01*y + m02*z + m03;
        sp[3*i+1] = m10*x + m11*y + m12*z + m13;
        sp[3*i+2] = m20*x + m21*y + m22*z + m23;
        sn[3*i+0] = m00*xn + m01*yn + m02*zn;
        sn[3*i+1] = m10*xn + m11*yn + m12*zn;
        sn[3*i+2] = m20*xn + m21*yn + m22*zn;
        si[i]     = 1.0/p.point_information;//sqrt(1.0/p.point_information);//Stddiv
        rangew[i] = 0;
        meandist_wsum += p.point_information;
        meandist += sqrt(x*x+y*y+z*z)*p.point_information;
    }
    meandist /= meandist_wsum;

    if(visualizationLvl > 0){show(sp, s_nrp, dp, d_nrp);}

    double * residuals   = 0;
    switch(type) {
        case PointToPoint:	{
            residuals   = new double[3*s_nrp];
        } 						break;
        case PointToPlane:	{
            residuals   = new double[s_nrp];
        }break;
        default:			{printf("type not set\n");}					break;
    }

    double score = 0;

    if(use_timer){ addTime("init", getTime()-initStart); }

    int capacity  = 1;
    size_t *  ret_indexes = new size_t[capacity];
    double * out_dists_sqr = new double[capacity];
    nanoflann::KNNResultSet<double> resultSet(capacity);

    bool timestopped = false;
    double start = getTime();
    for(int funcupdate=0; funcupdate < 100; ++funcupdate) {
        if( (getTime()-start) > maxtime ){timestopped = true; break;}
        Eigen::Matrix4d startResultFunc = result;
//        printf("RegistrationRefinement2::funcupdate: %i\n",funcupdate);

        for(int rematching=0; rematching < 30; ++rematching) {
            if( (getTime()-start) > maxtime ){timestopped = true; break;}
            Eigen::Matrix4d startResultRematch = result;
//            printf("RegistrationRefinement2::funcupdate: %i rematching: %i\n",funcupdate,rematching);

            if(use_timer){ matchStart = getTime(); }
            for(unsigned int i=0; i< s_nrp; ++i) {
                resultSet.init(ret_indexes, out_dists_sqr);
                trees3d->findNeighbors(resultSet, sp + 3*i, nanoflann::SearchParams(10));
                unsigned int id = ret_indexes[0];
                matches[i]      = id;
                Qp[3*i+0]       = dp[3*id+0];
                Qp[3*i+1]       = dp[3*id+1];
                Qp[3*i+2]       = dp[3*id+2];
                Qn[3*i+0]       = dn[3*id+0];
                Qn[3*i+1]       = dn[3*id+1];
                Qn[3*i+2]       = dn[3*id+2];
                rangew[i]       = 1.0/(1.0/si[i]+1.0/di[id]);
            }
            if(use_timer){ addTime("matching", getTime()-matchStart); }

            for(int outer=0; outer< 1; ++outer) {
                if( (getTime()-start) > maxtime ){timestopped = true; break;}
//                printf("RegistrationRefinement2::funcupdate: %i rematching: %i outer: %i\n",funcupdate,rematching,outer);

                if(use_timer){ computeResidualsStart = getTime(); }
                switch(type) {
                    case PointToPoint:	{
                        for(unsigned int i=0; i<s_nrp; ++i) {
                            double rw = rangew[i];
                            residuals[3*i+0] = rw*(sp[3*i+0]-Qp[3*i+0]);
                            residuals[3*i+1] = rw*(sp[3*i+1]-Qp[3*i+1]);
                            residuals[3*i+2] = rw*(sp[3*i+2]-Qp[3*i+2]);
                        }
                    }break;
                    case PointToPlane:	{
                        for(unsigned int i=0; i<s_nrp; ++i) {
                            float dx = sp[3*i+0]-Qp[3*i+0];
                            float dy = sp[3*i+1]-Qp[3*i+1];
                            float dz = sp[3*i+2]-Qp[3*i+2];
                            float qx = Qn[3*i+0];
                            float qy = Qn[3*i+1];
                            float qz = Qn[3*i+2];
                            float di = qx*dx + qy*dy + qz*dz;
                            residuals[i] = di;//*rangew[i];
                        }
                    }break;
                    default:			{printf("type not set\n");}					break;
                }
                if(use_timer){ addTime("computeResiduals", getTime()-computeResidualsStart); }
                if(use_timer){ computeModelStart = getTime(); }
                switch(type) {
                case PointToPoint:	{func->computeModel(residuals,s_nrp,3);} 	break;
                case PointToPlane:	{func->computeModel(residuals,s_nrp,1);}	break;
                default:  			{printf("type not set\n");} break;
                }
                if(use_timer){ addTime("computeModel", getTime()-computeModelStart); }

                stop = 0.2*func->getNoise()*func->dist->getNoise()/double(func->histogram_size);

                Eigen::Affine3d p = Eigen::Affine3d::Identity();
                if(use_timer){ innerStart = getTime(); }

                typedef Eigen::Matrix<double, 6, 1> Vector6d;
                typedef Eigen::Matrix<double, 6, 6> Matrix6d;

                Matrix6d ATA;
                Vector6d ATb;

                for(long inner=0; inner < 15 ; ++inner) {
                    score = 0;

                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr scloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr dcloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);

                    ATA.setZero ();
                    ATb.setZero ();
                    const double & m00 = p(0,0); const double & m01 = p(0,1); const double & m02 = p(0,2); const double & m03 = p(0,3);
                    const double & m10 = p(1,0); const double & m11 = p(1,1); const double & m12 = p(1,2); const double & m13 = p(1,3);
                    const double & m20 = p(2,0); const double & m21 = p(2,1); const double & m22 = p(2,2); const double & m23 = p(2,3);

                    for(unsigned int i=0; i<s_nrp; ++i) {
                        const double & src_x = sp[3*i+0];
                        const double & src_y = sp[3*i+1];
                        const double & src_z = sp[3*i+2];
                        const double & src_nx = sn[3*i+0];
                        const double & src_ny = sn[3*i+1];
                        const double & src_nz = sn[3*i+2];

                        const double & dx = Qp[3*i+0];
                        const double & dy = Qp[3*i+1];
                        const double & dz = Qp[3*i+2];
                        const double & dnx = Qn[3*i+0];
                        const double & dny = Qn[3*i+1];
                        const double & dnz = Qn[3*i+2];

                        const double & rw = rangew[i];

                        const double & sx = m00*src_x + m01*src_y + m02*src_z + m03;
                        const double & sy = m10*src_x + m11*src_y + m12*src_z + m13;
                        const double & sz = m20*src_x + m21*src_y + m22*src_z + m23;

                        const double & nx = m00*src_nx + m01*src_ny + m02*src_nz;
                        const double & ny = m10*src_nx + m11*src_ny + m12*src_nz;
                        const double & nz = m20*src_nx + m21*src_ny + m22*src_nz;

                        const double & angle = nx*dnx+ny*dny+nz*dnz;
                        if(angle < 0){continue;}

                        double di = rw*(nx*(sx-dx) + ny*(sy-dy) + nz*(sz-dz));

                        double prob = func->getProb(di);
                        double weight = prob*rw*rw;
                        score += prob;

                        if(false && visualizationLvl == 5){
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

                    for(long d = 0; d < 6; d++){ATA(d,d) += 0.1;}

//                    if(visualizationLvl == 5){
//                        printf("change_t: %10.10f change_r: %10.10f stopval: %10.10f\n",change_t,change_r,stopval);
//                        viewer->removeAllPointClouds();
//                        viewer->addPointCloud<pcl::PointXYZRGBNormal> (scloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(scloud), "scloud");
//                        viewer->addPointCloud<pcl::PointXYZRGBNormal> (dcloud, pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>(dcloud), "dcloud");
//                        viewer->spin();
//                        viewer->removeAllPointClouds();
//                        viewer->removeAllShapes();
//                    }

                    // Solve A*x = b
                    Vector6d x = static_cast<Vector6d> (ATA.inverse () * ATb);
                    Eigen::Affine3d transformation = Eigen::Affine3d(constructTransformationMatrix(x(0,0),x(1,0),x(2,0),x(3,0),x(4,0),x(5,0)));
                    p = transformation*p;
                    if(getChange(transformation.matrix(),meandist) < stop){break;}
                }

                result = p.matrix() * result;
                const double & m00 = p(0,0); const double & m01 = p(0,1); const double & m02 = p(0,2); const double & m03 = p(0,3);
                const double & m10 = p(1,0); const double & m11 = p(1,1); const double & m12 = p(1,2); const double & m13 = p(1,3);
                const double & m20 = p(2,0); const double & m21 = p(2,1); const double & m22 = p(2,2); const double & m23 = p(2,3);
                for(unsigned int i = 0; i < s_nrp; i++){
                    double x  = sp[3*i+0];
                    double y  = sp[3*i+1];
                    double z  = sp[3*i+2];
                    double xn = sn[3*i+0];
                    double yn = sn[3*i+1];
                    double zn = sn[3*i+2];
                    sp[3*i+0] = m00*x + m01*y + m02*z + m03;
                    sp[3*i+1] = m10*x + m11*y + m12*z + m13;
                    sp[3*i+2] = m20*x + m21*y + m22*z + m23;
                    sn[3*i+0] = m00*xn + m01*yn + m02*zn;
                    sn[3*i+1] = m10*xn + m11*yn + m12*zn;
                    sn[3*i+2] = m20*xn + m21*yn + m22*zn;
                }
                if(use_timer){ addTime("inner", getTime()-innerStart); }


                if(visualizationLvl == 4){show(sp, s_nrp, dp, d_nrp);}

                if(getChange(p.matrix(),meandist) < stop){break;}
            }

            if(visualizationLvl == 3){show(sp, s_nrp, dp, d_nrp);}
            if(getChange(startResultRematch,result,meandist) < stop){
                //printf("funcupdate: %i rematching: %i\n",funcupdate,rematching);
                break;
            }
        }


        //for(unsigned int i=0; i<s_nrp; i+=10) {printf("%4.4f ",residuals[i]);}printf("\n");

        if(visualizationLvl == 2){show(sp, s_nrp, dp, d_nrp);}
        if(use_timer){ funcUpdateStart = getTime(); }
        double noise_before = func->getNoise()*func->dist->getNoise()/double(func->histogram_size);//func->dist->getNoise();
        //func->debugg_print = true;
        func->update();
        //func->debugg_print = false;
        double noise_after = func->getNoise()*func->dist->getNoise()/double(func->histogram_size);//func->dist->getNoise();
        //printf("before: %5.5f after: %5.5f\n",noise_before,noise_after);

        if(use_timer){addTime("funcUpdate", getTime()-funcUpdateStart);}
        if(fabs(1.0 - noise_after/noise_before) < 0.01){break;}
    }

    if(visualizationLvl == 1){show(sp, s_nrp, dp, d_nrp);}

    delete[] ret_indexes;
    delete[] out_dists_sqr;
    delete[] sp;
    delete[] sn;
    delete[] si;
    delete[] matches;
    delete[] rangew;
    delete[] Qp;
    delete[] Qn;
    delete func;

    FusionResults fr = FusionResults(result,score);
    fr.timeout = timestopped;
    fr.stop = stop;
    return fr;
}

}
