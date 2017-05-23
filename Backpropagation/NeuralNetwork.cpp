#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(int nInputs, int hiddenLayerSize, int nHiddenLayers, int nOutputs, double (*activationFunction)(double),double (*gradientFunction)(double))
{
    mLayers.clear(); mNInputs=nInputs;mNOutputs=nOutputs;

    mLayers.push_back(Layer (nInputs,0,hiddenLayerSize,activationFunction,gradientFunction));

    for (int i=0;i<nHiddenLayers;++i){
        if (i== 0)
            mLayers.push_back(Layer (hiddenLayerSize,nInputs,hiddenLayerSize,activationFunction,gradientFunction));
        else if(i<nHiddenLayers-1)
            mLayers.push_back(Layer (hiddenLayerSize,hiddenLayerSize,hiddenLayerSize,activationFunction,gradientFunction));
        else if (i==nHiddenLayers-1)
            mLayers.push_back(Layer (hiddenLayerSize,hiddenLayerSize,nOutputs,activationFunction,gradientFunction));
    }

    mLayers.push_back(Layer (nOutputs,hiddenLayerSize,0,activationFunction,gradientFunction));
}

int NeuralNetwork::computeOutput(vec input)
{
    mLayers.front().computeOutput(input);

    for (unsigned int i =1;i<mLayers.size();++i){
        mLayers[i].computeOutput(mLayers[i-1].mOutput);
    }

    double max=mLayers.back().mOutput[0];
    int pos=0;
    for (int i=1;i<mLayers.back().mOutput.size();++i){
        if(mLayers.back().mOutput[i]>max)
        {
            max=mLayers.back().mOutput[i];
            pos=i;
        }
    }
    mLayers.back().mOutput.print("Output Final");
    return pos;
}




void NeuralNetwork::backpropagation(vector<vec>& trainingSet)
{

    int iteration=0;
    while(mError>0.0001 &&  iteration<1000){
        mError=0;
        for (vec &example: trainingSet){
            if (example.size() != (mNInputs + mNOutputs)){ cout<<"error input and neural network not match"<<endl;return;}


            //FORDWARD

            computeOutput(example.subvec(0,mNInputs-1));



            //BACKWARD


            vec error = example.subvec(mNInputs,example.size()-1) - mLayers.back().mOutput;
            mLayers.back().mError = mLayers.back().mGradient % error;
            //cout<<"dfasdasd"<<endl;

            mError+=sum(square(error));

            for (int i =mLayers.size()-2;i>=0;--i){

                vec output=mLayers[i].mOutput;
                output.insert_rows(0,1);
                output.row(0)=1;

                //mLayers[i+1].mWeights.print("w de la sgte "+to_string(mLayers[i+1].mWeights.n_rows)+"x"+to_string(mLayers[i+1].mWeights.n_cols));

                //mLayers[i+1].mError.print("error de la sgte "+to_string(mLayers[i+1].mError.n_rows)+"x"+to_string(mLayers[i+1].mError.n_cols));
               // output.print("mi salida "+to_string(output.n_rows)+"x"+to_string(output.n_cols));

                //cout<<"rate "<<mLearningRate<<endl;
                mat multi1=mLayers[i+1].mError*trans(output);
                //multi1.print("salida*error");

                mLayers[i+1].mWeights+=mLearningRate*mLayers[i+1].mError*trans(output);



                //mLayers[i].mGradient.print("derivate");

                //mLayers[i+1].mWeights.print("pesitos");

                mat pesos= trans(mLayers[i+1].mWeights);

                //pesos.print("trans de pesos");


                //pesos.submat(1,0,pesos.n_rows-1,pesos.n_cols-1).print("sin bias");

                mat mult=(pesos.submat(1,0,pesos.n_rows-1,pesos.n_cols-1))*mLayers[i+1].mError;

                //mult.print("pesos *  error");


                mLayers[i].mError=(pesos.submat(1,0,pesos.n_rows-1,pesos.n_cols-1)*mLayers[i+1].mError)%mLayers[i].mGradient;

                //mLayers[i].mError.print("myerror");


            }
        }

        mError*=0.5*(1.0/trainingSet.size());

        //cout<<"_Current Error: "<<mError<<endl;
        ++iteration;
    }


}
