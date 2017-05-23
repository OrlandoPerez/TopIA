#-------------------------------------------------
#
# Project created by QtCreator 2017-05-07T09:21:52
#
#-------------------------------------------------

QT       += core gui
CONFIG += c++14

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = Backpropagation
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    NeuralNetwork.cpp \
    Layer.cpp

HEADERS  += mainwindow.h \
    NeuralNetwork.h \
    Layer.h

FORMS    += mainwindow.ui

LIBS+= -larmadillo
#para CImg :
LIBS += -lX11
LIBS += -pthread
