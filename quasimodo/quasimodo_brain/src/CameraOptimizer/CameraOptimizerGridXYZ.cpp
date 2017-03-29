#include "CameraOptimizerGridXYZ.h"

int CameraOptimizerGridXYZ::getInd(int w, int h, int z){return (h*grid_w+w)*grid_z + z;}

CameraOptimizerGridXYZ::CameraOptimizerGridXYZ(int gw, int gh, int gz, double bias, double mz){
    grid_w = gw;
    grid_h = gh;
    grid_z = gz;
    max_z = mz;
    int nr_corners = (1+grid_w)*(1+grid_h)*(1+grid_z);
    mul.resize(nr_corners);
    sum.resize(nr_corners);
    for(unsigned long i = 0; i < nr_corners;i++){
        sum[i] = bias;
        mul[i] = sum[i];
    }
    visualizationLvl = 0;
}

CameraOptimizerGridXYZ::~CameraOptimizerGridXYZ(){}

void CameraOptimizerGridXYZ::addConstraint(double w, double h, double z, double z2, double weight){
    double ratio = z2/z;

    double wg = double(grid_w)*w;
    double hg = double(grid_h)*h;
    double zg = double(grid_z)*z/max_z;
    zg = std::min(zg,double(grid_z)-0.0001);

    double dw = wg-std::floor(wg);
    double dh = hg-std::floor(hg);
    double dz = zg-std::floor(zg);
    int ind000 = getInd(wg+0,hg+0,zg+0);
    int ind010 = getInd(wg+0,hg+1,zg+0);
    int ind100 = getInd(wg+1,hg+0,zg+0);
    int ind110 = getInd(wg+1,hg+1,zg+0);
    int ind001 = getInd(wg+0,hg+0,zg+1);
    int ind011 = getInd(wg+0,hg+1,zg+1);
    int ind101 = getInd(wg+1,hg+0,zg+1);
    int ind111 = getInd(wg+1,hg+1,zg+1);
    double mul000 = (1-dw)*(1-dh)*(1-dz);
    double mul010 = (1-dw)*(dh  )*(1-dz);
    double mul100 = (dw  )*(1-dh)*(1-dz);
    double mul110 = (dw  )*(dh  )*(1-dz);
    double mul001 = (1-dw)*(1-dh)*(  dz);
    double mul011 = (1-dw)*(dh  )*(  dz);
    double mul101 = (dw  )*(1-dh)*(  dz);
    double mul111 = (dw  )*(dh  )*(  dz);

    sum[ind000] +=       weight*mul000;
    mul[ind000] += ratio*weight*mul000;

    sum[ind010] +=       weight*mul010;
    mul[ind010] += ratio*weight*mul010;

    sum[ind100] +=       weight*mul100;
    mul[ind100] += ratio*weight*mul100;

    sum[ind110] +=       weight*mul110;
    mul[ind110] += ratio*weight*mul110;

    sum[ind001] +=       weight*mul001;
    mul[ind001] += ratio*weight*mul001;

    sum[ind011] +=       weight*mul011;
    mul[ind011] += ratio*weight*mul011;

    sum[ind101] +=       weight*mul101;
    mul[ind101] += ratio*weight*mul101;

    sum[ind111] +=       weight*mul111;
    mul[ind111] += ratio*weight*mul111;

    //printf("%f %f %f-> %f %f %f\n",w,h,z,wg,hg,zg);
}

double CameraOptimizerGridXYZ::getRange(double w, double h, double z, bool debugg){

    double wg = double(grid_w)*w;
    double hg = double(grid_h)*h;
    double zg = double(grid_z)*z/max_z;
    zg = std::min(zg,double(grid_z)-0.0001);

    double dw = wg-std::floor(wg);
    double dh = hg-std::floor(hg);
    double dz = zg-std::floor(zg);
    int ind000 = getInd(wg+0,hg+0,zg+0);
    int ind010 = getInd(wg+0,hg+1,zg+0);
    int ind100 = getInd(wg+1,hg+0,zg+0);
    int ind110 = getInd(wg+1,hg+1,zg+0);
    int ind001 = getInd(wg+0,hg+0,zg+1);
    int ind011 = getInd(wg+0,hg+1,zg+1);
    int ind101 = getInd(wg+1,hg+0,zg+1);
    int ind111 = getInd(wg+1,hg+1,zg+1);
    double mul000 = (1-dw)*(1-dh)*(1-dz);
    double mul010 = (1-dw)*(dh  )*(1-dz);
    double mul100 = (dw  )*(1-dh)*(1-dz);
    double mul110 = (dw  )*(dh  )*(1-dz);
    double mul001 = (1-dw)*(1-dh)*(  dz);
    double mul011 = (1-dw)*(dh  )*(  dz);
    double mul101 = (dw  )*(1-dh)*(  dz);
    double mul111 = (dw  )*(dh  )*(  dz);

    double r000 = mul[ind000]/sum[ind000];
    double r010 = mul[ind010]/sum[ind010];
    double r100 = mul[ind100]/sum[ind100];
    double r110 = mul[ind110]/sum[ind110];

    double r001 = mul[ind001]/sum[ind001];
    double r011 = mul[ind011]/sum[ind011];
    double r101 = mul[ind101]/sum[ind101];
    double r111 = mul[ind111]/sum[ind111];

    double ratio = mul000*r000 + mul010*r010  + mul100*r100 + mul110*r110 + mul001*r001 + mul011*r011  + mul101*r101 + mul111*r111;

    //if(debugg){ printf("%f %f %f -> max_z %f -> %f %f %f -> %f\n",w,h,z,max_z,wg,hg,zg,ratio); }
    return z*ratio;
}


void CameraOptimizerGridXYZ::save(std::string path){
    long size = 6*sizeof(int)+sizeof(double)*(2+2*mul.size());
    char* buffer = new char[size];
    int * intbuf = (int*)buffer;
    double * doublebuf = (double*)buffer;
    intbuf[0] = 2;
    intbuf[1] = grid_w;
    intbuf[2] = grid_h;
    intbuf[3] = grid_z;
    intbuf[4] = mul.size();
    doublebuf[3] = bias;
    doublebuf[4] = max_z;

    for(unsigned long i = 0; i < mul.size();i++){
        doublebuf[5+2*i+0] = mul[i];
        doublebuf[5+2*i+1] = sum[i];
    }

    std::ofstream outfile (path,std::ofstream::binary);
    outfile.write (buffer,size);
    outfile.close();

    delete[] buffer;
}

void CameraOptimizerGridXYZ::loadInternal(std::string path){
    std::streampos size;
    char * memblock;

    std::ifstream file (path, std::ios::in|std::ios::binary|std::ios::ate);
    if (file.is_open()){
        size = file.tellg();
        memblock = new char [size];

        file.seekg (0, ios::beg);
        file.read (memblock, size);
        file.close();
        int * intbuf = (int*)memblock;
        double * doublebuf = (double*)memblock;

        grid_w = intbuf[1];
        grid_h = intbuf[2];
        grid_z = intbuf[3];
        int nr_mul = intbuf[4];
        bias = doublebuf[3];
        max_z = doublebuf[4];

        mul.resize(nr_mul);
        sum.resize(nr_mul);
        for(unsigned long i = 0; i < mul.size();i++){
            mul[i] = doublebuf[5+2*i+0];
            sum[i] = doublebuf[5+2*i+1];
        }

        delete[] memblock;
    }

}
