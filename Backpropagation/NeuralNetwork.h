#include "Layer.h"

#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H


class NeuralNetwork
{
public:
    NeuralNetwork(int nInputs,  int hiddenLayerSize, int nHiddenLayers, int nOutputs, double (*activationFunction)(double),double (*gradientFunction)(double));
    ~NeuralNetwork(){}

    int computeOutput(vec input);
    void backpropagation(vector<vec> &trainingSet);


    vector<Layer> mLayers;

    int mNInputs=0;
    int mNOutputs=0;
    double mError=100;

    double mLearningRate=0.1;

};

#endif // NEURALNETWORK_H
