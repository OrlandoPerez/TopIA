#include "Layer.h"

#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H


class NeuralNetwork
{
public:
    NeuralNetwork(int nInputs,  vector<int> hiddenLayers, int nOutputs, double (*activationFunction)(double), double (*gradientFunction)(double));
    ~NeuralNetwork(){}

    int computeOutput(vec &input);
    void backpropagation(vector<vec> &trainingInput, vector<vec> &trainingOutput, double momentum=0);


    vector<Layer> mLayers;

    int mNInputs=0;
    int mNOutputs=0;
    double mError=100;

    double mLearningRate=0.1;



    void backpropagationMiniBatch(vector<vec> &trainingInput, vector<vec> &trainingOutput, int batchSize, double momentum=0);
};

#endif // NEURALNETWORK_H
