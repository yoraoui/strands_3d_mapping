#ifndef CameraOptimizerGridXY_H
#define CameraOptimizerGridXY_H

#include "CameraOptimizerGrid.h"

class CameraOptimizerGridXY : public CameraOptimizerGrid{
public:
    int grid_w;
    int grid_h;

    int getInd(int w, int h);

    CameraOptimizerGridXY(int gw = 6, int gh = 4, double bias_ = 100);

    ~CameraOptimizerGridXY();

    virtual void addConstraint(double w, double h, double z, double z2, double weight);
    virtual double getRange(double w, double h, double z, bool debugg = false);
    virtual void print();

    virtual void save(std::string path);
    virtual void loadInternal(std::string path);
};

#endif
