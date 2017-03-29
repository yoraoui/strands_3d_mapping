#ifndef CameraOptimizerGridXYZ_H
#define CameraOptimizerGridXYZ_H

#include "CameraOptimizerGrid.h"

class CameraOptimizerGridXYZ : public CameraOptimizerGrid{
public:
    int grid_w;
    int grid_h;
    int grid_z;
    double max_z;

    int getInd(int w, int h, int z);

    CameraOptimizerGridXYZ(int gw = 6, int gh = 4, int gz = 10, double bias = 100, double mz = 10.0);

    ~CameraOptimizerGridXYZ();

    virtual void addConstraint(double w, double h, double z, double z2, double weight);

    virtual double getRange(double w, double h, double z, bool debugg = false);


    virtual void save(std::string path);
    virtual void loadInternal(std::string path);
};

#endif
