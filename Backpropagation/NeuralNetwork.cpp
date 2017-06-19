#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(int nInputs, vector<int> hiddenLayers, int nOutputs, double (*activationFunction)(double),double (*gradientFunction)(double))
{
    mLayers.clear(); mNInputs=nInputs;mNOutputs=nOutputs;

    if (hiddenLayers.size())
        mLayers.push_back(Layer (nInputs,0,hiddenLayers.front(),activationFunction,gradientFunction));
    else
        mLayers.push_back(Layer (nInputs,0,nOutputs,activationFunction,gradientFunction));

    if (hiddenLayers.size()>1){
        for (unsigned int i=0;i<hiddenLayers.size()-1;++i)
            mLayers.push_back(Layer (hiddenLayers[i],mLayers.back().mSize,hiddenLayers[i+1],activationFunction,gradientFunction));
    }

    if (hiddenLayers.size())
        mLayers.push_back(Layer (hiddenLayers.back(),mLayers.back().mSize,nOutputs,activationFunction,gradientFunction));


    mLayers.push_back(Layer (nOutputs,mLayers.back().mSize,0,activationFunction,gradientFunction));
}

int NeuralNetwork::computeOutput(vec& input)
{
    mLayers.front().computeOutput(input);

    for (unsigned int i =1;i<mLayers.size();++i){
        mLayers[i].computeOutput(mLayers[i-1].mOutput);
    }

    double max=mLayers.back().mOutput[0];
    int pos=0;
    for (unsigned int i=1;i<mLayers.back().mOutput.size();++i){
        if(mLayers.back().mOutput[i]>max)
        {
            max=mLayers.back().mOutput[i];
            pos=i;
        }
    }

   // mLayers.back().mOutput.print("Output Final");
    return pos;
}




void NeuralNetwork::backpropagation(vector<vec>& trainingInput, vector<vec> &trainingOutput,double momentum)
{
    if (trainingInput.size()!=trainingOutput.size()){
        cout<<"input not match with output in training set"<<endl;
    }

    int iteration=0;
    mError=1000;

    while(mError>0.001 &&  iteration<1000){
        cout<<iteration<<endl;
        mError=0;
        for (unsigned int i=0;i< trainingInput.size();++i){
            if (trainingInput[i].size()== mNInputs && trainingOutput[i].size()==mNOutputs){


                //FORDWARD

                computeOutput(trainingInput[i]);

                //BACKWARD
                vec error = trainingOutput[i] - mLayers.back().mOutput;
                mLayers.back().mError = mLayers.back().mGradient % error;

                mError+=sum(square(error));

                for (int j =mLayers.size()-2;j>=0;--j){

                    vec output=mLayers[j].mOutput;
                    output.insert_rows(0,1);
                    output.row(0)=1;

                    // gradient descent with momentum

                    mLayers[j+1].mDeltaWeights=(momentum*mLayers[j+1].mDeltaWeights) + ((1-momentum)*mLearningRate*(mLayers[j+1].mError*trans(output)));

                    mLayers[j+1].mWeights+=mLayers[j+1].mDeltaWeights;


                    mat pesos= trans(mLayers[j+1].mWeights);

                    mLayers[j].mError=(pesos.submat(1,0,pesos.n_rows-1,pesos.n_cols-1)*mLayers[j+1].mError)%mLayers[j].mGradient;

                }

            }
            else{
                cout<<"error input and neural network not match"<<endl;
                return;
            }

        }

        mError*=1.0/(2*trainingInput.size());
        cout<<"error : "<<mError<<endl;
        ++iteration;
    }
}

void NeuralNetwork::backpropagationMiniBatch(vector<vec>& trainingInput, vector<vec> &trainingOutput,int batchSize,double momentum)
{
    if (trainingInput.size()!=trainingOutput.size()){
        cout<<"input not match with output in training set"<<endl;
    }


    int iteration=0;
    mError=1000;

    while(mError>0.0001 &&  iteration<1000){
        cout<<iteration<<endl;
        mError=0;
        for (unsigned int i=0;i< trainingInput.size();i+=batchSize){
            unsigned int j=i;
            for (;j < i+batchSize;++j){
                if (j > trainingInput.size()-1) break;
                if (trainingInput[j].size()== mNInputs && trainingOutput[j].size()==mNOutputs){
                    //FORDWARD

                    computeOutput(trainingInput[j]);

                    //BACKWARD
                    vec error = trainingOutput[j] - mLayers.back().mOutput;
                    mLayers.back().mError = mLayers.back().mGradient % error;

                    mError=sum(square(error));


                    for (int k =mLayers.size()-2;k>=0;--k){

                        vec output=mLayers[k].mOutput;
                        output.insert_rows(0,1);
                        output.row(0)=1;

                        // gradient descent with momentum



                        mLayers[k+1].mDeltaWeights=mLayers[k+1].mError*trans(output);

                        mLayers[k+1].mDeltaWeightsSum+=mLayers[k+1].mDeltaWeights;

                        mat pesos= trans(mLayers[k+1].mWeights+mLayers[k+1].mDeltaWeights);

                        mLayers[k].mError=(pesos.submat(1,0,pesos.n_rows-1,pesos.n_cols-1)*mLayers[k+1].mError)%mLayers[k].mGradient;
                    }


                }
                else{
                    cout<<"error input and neural network not match"<<endl;
                    return;
                }
            }

            cout<<"batch size"<<j-i<<endl;
            for (int k =mLayers.size()-2;k>=0;--k){
                mLayers[k+1].mWeights+=mLearningRate*(1.0/(j-i))* mLayers[k+1].mDeltaWeightsSum;
                mLayers[k+1].mDeltaWeightsSum*=0;
            }
        }

        mError*=1.0/(2*trainingInput.size());
        cout<<"error : "<<mError<<endl;

        ++iteration;
    }
}




