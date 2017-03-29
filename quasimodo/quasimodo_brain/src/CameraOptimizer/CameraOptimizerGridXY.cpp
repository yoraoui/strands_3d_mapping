#include "CameraOptimizerGridXY.h"

int CameraOptimizerGridXY::getInd(int w, int h){return h*grid_w+w;}

CameraOptimizerGridXY::CameraOptimizerGridXY(int gw, int gh, double bias){
    grid_w = gw;
    grid_h = gh;
    int nr_corners = (1+grid_w)*(1+grid_h);
    mul.resize(nr_corners);
    sum.resize(nr_corners);
    for(unsigned long i = 0; i < nr_corners;i++){
        sum[i] = bias;
        mul[i] = sum[i];
    }
    visualizationLvl = 0;
}

CameraOptimizerGridXY::~CameraOptimizerGridXY(){}


void CameraOptimizerGridXY::addConstraint(double w, double h, double z, double z2, double weight){
    double ratio = z2/z;

    double wg = double(grid_w)*w;
    double hg = double(grid_h)*h;

    double dw = wg-std::floor(wg);
    double dh = hg-std::floor(hg);
    int ind00 = getInd(wg  , hg  );
    int ind01 = getInd(wg  , hg+1);
    int ind10 = getInd(wg+1, hg  );
    int ind11 = getInd(wg+1, hg+1);
    double mul00 = (1-dw)*(1-dh);
    double mul01 = (1-dw)*(  dh);
    double mul10 = (  dw)*(1-dh);
    double mul11 = (  dw)*(  dh);

    sum[ind00] +=       weight*mul00;
    mul[ind00] += ratio*weight*mul00;

    sum[ind01] +=       weight*mul01;
    mul[ind01] += ratio*weight*mul01;

    sum[ind10] +=       weight*mul10;
    mul[ind10] += ratio*weight*mul10;

    sum[ind11] +=       weight*mul11;
    mul[ind11] += ratio*weight*mul11;
}

double CameraOptimizerGridXY::getRange(double w, double h, double z, bool debugg){
    double wg = double(grid_w)*w;
    double hg = double(grid_h)*h;

    double dw = wg-std::floor(wg);
    double dh = hg-std::floor(hg);
    int ind00 = getInd(wg  , hg  );
    int ind01 = getInd(wg  , hg+1);
    int ind10 = getInd(wg+1, hg  );
    int ind11 = getInd(wg+1, hg+1);
    double mul00 = (1-dw)*(1-dh);
    double mul01 = (1-dw)*(  dh);
    double mul10 = (dw  )*(1-dh);
    double mul11 = (dw  )*(  dh);

    double r00 = mul[ind00]/sum[ind00];
    double r01 = mul[ind01]/sum[ind01];
    double r10 = mul[ind10]/sum[ind10];
    double r11 = mul[ind11]/sum[ind11];
    double ratio = (mul00*r00 + mul01*r01  + mul10*r10 + mul11*r11);
    return z*ratio;
}


void CameraOptimizerGridXY::print(){
    for(unsigned int w = 0; w < grid_w; w++){
        for(unsigned int h = 0; h < grid_h; h++){
            unsigned int ind = h * grid_w + w;
            printf("%i %i -> %5.5f / %5.5f -> %5.5f\n",w,h,mul[ind],sum[ind],mul[ind]/sum[ind]);
        }
    }
}

void CameraOptimizerGridXY::save(std::string path){
    long size = 4*sizeof(int)+sizeof(double)*(1+2*mul.size());
    char* buffer = new char[size];
    int * intbuf = (int*)buffer;
    double * doublebuf = (double*)buffer;
    intbuf[0] = 1;
    intbuf[1] = grid_w;
    intbuf[2] = grid_h;
    intbuf[3] = mul.size();
    doublebuf[2] = bias;

    for(unsigned long i = 0; i < mul.size();i++){
        doublebuf[3+2*i+0] = mul[i];
        doublebuf[3+2*i+1] = sum[i];
    }

    std::ofstream outfile (path,std::ofstream::binary);
    outfile.write (buffer,size);
    outfile.close();

    delete[] buffer;
}

void CameraOptimizerGridXY::loadInternal(std::string path){
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
        int nr_mul = intbuf[3];
        bias = doublebuf[2];

        mul.resize(nr_mul);
        sum.resize(nr_mul);
        for(unsigned long i = 0; i < mul.size();i++){
            mul[i] = doublebuf[3+2*i+0];
            sum[i] = doublebuf[3+2*i+1];
        }

        delete[] memblock;
    }

}
