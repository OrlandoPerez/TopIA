#include "Layer.h"

Layer::Layer(int size, int nIn, int nOut, double (*activationFunction)(double),double (*gradientFunction)(double))
{
    mSize=size;

    mActivationFunction=activationFunction;


    mGradientFunction=gradientFunction;

    ramdomWeitghs(nIn,size);

    mDeltaWeights=mat(size,nIn+1,fill::zeros);

    mDeltaWeightsSum=mat(size,nIn+1,fill::zeros);


}

void Layer::ramdomWeitghs(int nIn,int nOut)
{
    mWeights.clear();

    if (nIn){
        double eInit=sqrt(6)/sqrt(nIn+nOut);

        mWeights=mat(nOut,nIn+1,fill::randu)* 2 *eInit -eInit;
        //mWeights=mat(nOut,nIn+1,fill::randu);
        //mWeights=mat(nOut,nIn+1,fill::zeros);
    }

}

void Layer::computeOutput(mat input)
{
    if (!mWeights.is_empty()){
       // mWeights.print("weight");

        input.insert_rows(0,1);
        input.row(0)=1;

        mOutput=mWeights*input;

        mGradient=mOutput;

        mOutput.for_each( [this](vec::elem_type & element) { element=mActivationFunction(element);  } );
        mGradient.for_each( [this](vec::elem_type & element) { element=mGradientFunction(element);  } );


    }
    else{
        mOutput=input;

        mGradient=mOutput;
        mGradient.for_each( [this](vec::elem_type & element) { element=mGradientFunction(element);  } );
    }

}

int Layer::size()
{
    return mSize;
}
