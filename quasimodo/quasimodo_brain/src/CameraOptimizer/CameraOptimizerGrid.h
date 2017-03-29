#ifndef CameraOptimizerGrid_H
#define CameraOptimizerGrid_H

#include "CameraOptimizer.h"

class CameraOptimizerGrid : public CameraOptimizer{
public:
    std::vector<double> mul;
    std::vector<double> sum;
    double bias;

    CameraOptimizerGrid();
    ~CameraOptimizerGrid();

    virtual void shrink();
    virtual void normalize();
    virtual double getMax();
    virtual double getMin();
};

#include "CameraOptimizerGridXY.h"
#include "CameraOptimizerGridXYZ.h"

#endif
