#include "mainwindow.h"
#include <QApplication>
#include "NeuralNetwork.h"
#include <stdio.h>
#include <CImg.h>


using namespace cimg_library;


const int mySize =784;


double sigmoid(double x){
    return 1.0/(1+exp(-x));
}

double sigmoidGradient(double x){
    return sigmoid(x)*(1-sigmoid(x));
}


void loadData(string fileName, vector<vec>& set ){


    string line;
    ifstream fileIn(fileName);
    if (fileIn.is_open())
    {
        while ( getline (fileIn,line) )
        {
            std::stringstream pixels(line);
            int index = 0;
            vec input = zeros(mySize+10);

            for (std::string dato; std::getline(pixels,dato, ','); )
            {
                if(index==0)
                {
                    input[input.size()-(10-atoi(dato.c_str()))] = 1  ;
                    ++index;
                }
                else
                {
                    //cout<<dato<<endl;
                    input[index-1] = atoi(dato.c_str())>127;
                    ++index;
                }

            }
            set.push_back(input);
        }
        fileIn.close();
    }

    else cout << "No se encuentra el archivo..."<<endl;

}



int main(int argc, char *argv[])
{
    //QApplication a(argc, argv);
    //MainWindow w;
    //w.show();

    //return a.exec();
    //cout<<sigmoid(1)<<endl;
    //cout<<sigmoidGradient(1)<<endl;
    arma_rng::set_seed_random();
    NeuralNetwork n(mySize,35,1,10,&sigmoid,&sigmoidGradient);

   /* vector<vec> set={
        {1,3,4},{5,6,3},{200,1,7},{0,5,9}
    };*/

    vector<vec> dataSet;

    //loadData("DataSet/mnist_train_100.csv",dataSet);
    loadData("DataSet/mnist_train.csv",dataSet);


    //dataSet[0].print("vamo a ver");


    //cout<<dataSet.size()<<endl;computeOutput

    //cout<<dataSet.front().size()<<endl;
    n.backpropagation(dataSet);

    vector<vec> test;
    loadData("DataSet/mnist_test_10.csv",test);
    //loadData("DataSet/mnist_test.csv",test);


    int count=0;
     for (vec &t:test){
        //cout<<test.size()<<endl;
        int best=n.computeOutput(t.subvec(0,t.size()-11));


        for (int i=1;i<t.subvec(t.size()-10,t.size()-1).n_elem;++i){
            if(t.subvec(t.size()-10,t.size()-1)[i]){
                if(best==i)
                    ++count;
            }
        }



       /* CImg<unsigned char> digit(28,28,1,1,1);

        for(int i=0;i<28;++i)
            for(int j=0;j<28;++j)
                digit(j,i)=t[(i*28)+j];

        digit.display();*/

    }
     cout<<"Accuracy: "
           ""<<count/10.0<<endl;

    /*vec a={{1,2,3}};
    mat b={{4,2,3}};

    a.print("a");
    b.print("b");
    mat c=a*b;
    c.print("c");*/
    //+=4;
    //a.print("a");
    //cout<<a.<<endl;


    //n.computeOutput({1,0});


    return 0;
}
