#include "mainwindow.h"
#include <QApplication>
#include "NeuralNetwork.h"
#include <stdio.h>
#include <CImg.h>


using namespace cimg_library;



const int inputSize =784;
const int outputSize =10;

//const int inputSize =4;
//const int outputSize =3;


double sigmoid(double x){
    return 1.0/(1+exp(-x));
}

double sigmoidGradient(double x){
    return sigmoid(x)*(1-sigmoid(x));
}


void loadData(string fileName, vector<vec>& trainingInput,vector<vec>& trainingOutput ){


    string line;
    ifstream fileIn(fileName);
    if (fileIn.is_open())
    {
        while ( getline (fileIn,line) )
        {
            std::stringstream pixels(line);
            int index = 0;
            vec input = zeros(inputSize);
            vec output= zeros(outputSize);

            for (std::string dato; std::getline(pixels,dato, ','); )
            {
                if(index==0)
                //if (index<inputSize)
                {
                    output[atoi(dato.c_str())] = 1  ;
                   // input[index] = atof(dato.c_str())  ;
                    ++index;

                }
                else
                {
                    //output[index-inputSize] = atof(dato.c_str());
                    input[index-1] = atof(dato.c_str());
                    ++index;
                }

            }
            trainingInput.push_back(input);
            trainingOutput.push_back(output);

        }
        fileIn.close();
    }

    else cout << "No se encuentra el archivo..."<<endl;

}

void normalize(vector<vec>& set){

    vector<double> min(inputSize,100000.0);
    vector<double> max(inputSize,-100000.0);

    double d1=0;
    double d2=1;

    for (unsigned int i =0;i<set.size();++i){
        for (int j =0;j<inputSize;++j){
            if (min[j]> set[i][j]){
                min[j]=set[i][j];
            }
            if (max[j]< set[i][j])
                max[j]=set[i][j];
        }
    }
    for(unsigned int i =0;i<set.size();++i){
        for (int j =0;j<inputSize;++j){
            if (max[j]-min[j])
                set[i][j]=(((set[i][j]-min[j])*(d2-d1))/(max[j]-min[j]))+d1;
            else
                set[i][j]=d1;

        }
    }

}



int main(int argc, char *argv[])
{
    //QApplication a(argc, argv);
    //MainWindow w;
    //w.show();
    //return a.exec();

    arma_rng::set_seed_random();
    NeuralNetwork n(inputSize,{35},outputSize,&sigmoid,&sigmoidGradient);

    //NeuralNetwork n(2,{2},1,&sigmoid,&sigmoidGradient);

    //NeuralNetwork n(inputSize,{8,8},outputSize,&sigmoid,&sigmoidGradient);

   /* vector<vec> inputSet={
        {0,0},{0,1},{1,0},{1,1}
    };

    vector<vec> outputSet={
        {0},{1},{1},{0}
    };*/

    vector<vec> inputTraining;
    vector<vec> outputTraining;

    loadData("DataSet/mnist_train_100.csv",inputTraining,outputTraining);
    //loadData("DataSet/irisdataTrain",inputTraining,outputTraining);
    //loadData("DataSet/mnist_train.csv",inputTraining,outputTraining);

    normalize(inputTraining);


    //n.backpropagationMiniBatch(inputTraining,outputTraining,10);
    n.backpropagation(inputTraining,outputTraining);

    //n.backpropagation(inputSet,outputSet);

    vector<vec> inputTest;
    vector<vec> outputTest;

/*    vector<vec> inputTest={
        {0,0},{1,0},{0,1},{1,1}
    };

    vector<vec> outputTest={
        {0},{1},{1},{0}
    };
*/

    loadData("DataSet/mnist_test_10.csv",inputTest,outputTest);
    //loadData("DataSet/irisdataTest",inputTest,outputTest);
    //loadData("DataSet/mnist_test.csv",inputTest,outputTest);


    normalize(inputTest);

    cout<<"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"<<endl;
    double count=0;
     for (int i=0;i<inputTest.size();++i){
         int best=n.computeOutput(inputTest[i]);

         for (int j =0;j<outputTest[i].size();++j){
             if(outputTest[i][j]){
                 if(best==j)
                     ++count;
             }
         }


       /* CImg<unsigned char> digit(28,28,1,1,1);

        for(int i=0;i<28;++i)
            for(int j=0;j<28;++j)
                digit(j,i)=t[(i*28)+j];

        digit.display();*/

    }

    cout<<inputTraining.size()<<" test : "<<inputTest.size()<<" Accuracy: "<<count/inputTest.size()<<endl;



    return 0;
}

