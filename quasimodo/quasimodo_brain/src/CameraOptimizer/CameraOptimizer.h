#ifndef CameraOptimizer_H
#define CameraOptimizer_H

#include "../Util/Util.h"
#include <iostream>
#include <fstream>

class CameraOptimizer{
public:
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
    int visualizationLvl;
    int z_slider;

    CameraOptimizer();
    ~CameraOptimizer();

    void setVisualization(boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_, int visualizationLvl_ = 1);

    cv::Mat getPixelReliability(reglib::RGBDFrame * src);

    cv::Mat improveDepth(cv::Mat depth, double idepth);

    virtual void addConstraint(double w, double h, double z, double z2, double weight);

    virtual void redraw();

    virtual void show(bool stop = false);

    virtual void show(reglib::RGBDFrame * src, reglib::RGBDFrame * dst, Eigen::Matrix4d p);

    virtual void normalize();

    virtual double getRange(double w, double h, double z, bool debugg = false);

    virtual void addTrainingData( reglib::RGBDFrame * src, reglib::RGBDFrame * dst, Eigen::Matrix4d p);
    virtual void print();
    virtual void shrink();
    virtual double getMax();
    virtual double getMin();

    virtual void save(std::string path);
    virtual void loadInternal(std::string path);
    static CameraOptimizer * load(std::string path);
};

#include "CameraOptimizerGrid.h"

#endif
