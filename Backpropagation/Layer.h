#include <armadillo>
#include <vector>
#include <iostream>


using namespace arma;
using namespace std;


#ifndef LAYER_H
#define LAYER_H


class Layer
{
public:
    Layer(int size, int nIn, int nOut, double (*activationFunction) (double),double (*gradientFunction)(double));
    ~Layer (){}


    void ramdomWeitghs(int nIn, int nOut);



    mat mWeights;
    vec mOutput;

    vec mError;

    vec mGradient;

    mat mDeltaWeights;

    mat mDeltaWeightsSum;


    int mSize=0;


    void computeOutput(mat input);

    int size();

    double (*mActivationFunction) (double )=0;
    double (*mGradientFunction) (double )=0;




};

#endif // LAYER_H
