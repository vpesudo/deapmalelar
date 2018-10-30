#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

__author__ = ' AUTOR:    Miguel Cardenas Montes'

''' CNN unidimensional en Keras para O3 '''

"""Imports: librerias"""
import os
import math
from math import sqrt
import sys
from sys import argv
import numpy
import numpy as np
import random
import scipy.stats
from pylab import *
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from time import time

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import load_model

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

from sklearn.metrics import confusion_matrix

import pandas as pd  

import tensorflow as tf

##Para cambiar las fuentes
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 'large'
mpl.rcParams['ytick.labelsize'] = 'large'
mpl.rcParams['axes.labelsize'] = 'large'

import matplotlib as mpl
np.set_printoptions(threshold=np.inf)

import pdb # Para depurar
import copy

# command options: 
# This command creates a new model with 3 layers of neurons and runs it for a "nepochs" number of epochs. The number of neurons in each layer is chosen from the command.
# $ python3.5 ann_charge_isNeck_mcmvp_load.py nepochs neurons[0] neurons[1] neurons[2]

# This command loads trained model with 3 layers of neurons and that has run for a "nepochsold" number of epochs. It runs over it for "nepochsnow" number of epochs.
# $ python3.5 ann_charge_isNeck_mcmvp_load.py nepochsold neurons[0] neurons[1] neurons[2] load nepochsnow


##########################################
############## Codigo ####################
##########################################

"""Codigo principal"""
def main():

        ####### Some Parameters ###########

        nepochs =int(argv[1])
        # Energy cuts
        cutlow=40
        cuthigh=260

        # 2 capas ocultas con numero de neuronas, definidas en la variable siguiente
        #neurons = [128, 32, 8] 
        #nneurons = len(argv)-3
        neurons = [int(argv[2]),int(argv[3]),int(argv[4])] 
        print("neurons: ",argv[2],argv[3],argv[4])
        nepochstot = nepochs
        modelloaded = False




        ####### Estructura de la FFNN ###########
        ####### Comprobamos si la cargamos o la creamos  
        if( len(argv)==7 and argv[5]=='load'):
          ifolder = "/tmp/ml_outputs/testsize0.25sizeofbatch0.75"
          filename = "%s/model_ANN_epochs%d_2HL_%d_%d_%d__%d_%d.h5" % (ifolder,nepochs,neurons[0],neurons[1],neurons[2],cutlow,cuthigh)
          #filename = "%s" % (ifile)
          print('Loading model : ',filename)
          model = load_model(filename)
          modelloaded = True
          # epochs that want to be run in this iteration.
          nepochs = int(argv[6])
          # total number of epochs that the ouput model has gone through
          nepochstot += nepochs
        ####### Estructura de la FFNN ###########
          
          
        ####### Reading the data ###########

        #data = pd.read_csv('../rawdata/sample.txt', sep="\t", header=None)
        #data = pd.read_csv('../rawdata/lightpatternXYZar40_nominal_30000.txt', sep="\t", header=None)
        data01 = pd.read_csv('/tmp/data/sk_ar40_0_100.txt', sep="\t", header=None)
        data02 = pd.read_csv('/tmp/data/sk_ar40_100_200.txt', sep="\t", header=None)
        data03 = pd.read_csv('/tmp/data/sk_ar40_200_300.txt', sep="\t", header=None)
        data04 = pd.read_csv('/tmp/data/sk_ar40_300_400.txt', sep="\t", header=None)
        data05 = pd.read_csv('/tmp/data/sk_ar40_400_500.txt', sep="\t", header=None)
        data06 = pd.read_csv('/tmp/data/sk_ar40_500_600.txt', sep="\t", header=None)
        data07 = pd.read_csv('/tmp/data/sk_ar40_600_700.txt', sep="\t", header=None)
        data08 = pd.read_csv('/tmp/data/sk_ar40_700_800.txt', sep="\t", header=None)
        data09 = pd.read_csv('/tmp/data/sk_ar40_800_900.txt', sep="\t", header=None)
        data10 = pd.read_csv('/tmp/data/sk_ar40_900_1000.txt', sep="\t", header=None)
        data1 = pd.concat([data01,data02,data03,data04,data05,data06,data07,data08,data09,data10], ignore_index=True)
        data1 = shuffle(data1)

        data2 = pd.read_csv('/tmp/data/sk_neck_0_500.txt', sep="\t", header=None)
        data3 = pd.read_csv('/tmp/data/sk_neck_500_1000.txt', sep="\t", header=None)
        data4 = pd.read_csv('/tmp/data/sk_neck_1000_1500.txt', sep="\t", header=None)
        data5 = pd.read_csv('/tmp/data/sk_neck_1500_2000.txt', sep="\t", header=None)
        datan = pd.concat([data2,data3,data4,data5], ignore_index=True)

        print(shape(data1))
        print(shape(datan))

        # Adding summed Charge column
        datan['sumPMTs']=datan.iloc[:,0:255].sum(axis=1)

        # energy cuts: Selecting data of neck events
        datan = datan[(datan['sumPMTs'] >= cutlow) & (datan['sumPMTs'] <= cuthigh)]
        print('neck aftercuts ',shape(datan))

        data1['sumPMTs']=data1.iloc[:,0:255].sum(axis=1)
        # energy cuts: Performing same cuts on neck and Ar  events
        data1 = data1[(data1['sumPMTs'] >= cutlow) & (data1['sumPMTs'] <= cuthigh)]
        print('LAr aftercuts ',shape(data1))
        # Cutting Ar-40 data to avoid biasing
        datanum = datan.shape[0]*3
        print('LAr accepted ',datanum)
        datag= data1[:datanum]

        data = pd.concat([datag,datan], ignore_index=True)

        data = shuffle(data)

        # data = data[(data['sumPMTs'] >= cutlow) & (data['sumPMTs'] <= cuthigh)]
        
        print(shape(data))
        print(data.columns)

        # Selecting the last column: 1 if neck, 0 if Ar-40 recoil
        Y= data.iloc[:,513:514]
        print("Y.shape", Y.shape)
        
        # Enconde binary option A:
        #Y = to_categorical(Y,num_classes=2)
        #print("Y", Y.shape)

        # Encode binary option B: class values as integers
        #encoder = LabelEncoder()
        #encoder.fit(Y)
        #encoded_Y = encoder.transform(Y)
        #print("Y",Y)

        X= data.iloc[:,0:255]
        print(X.shape)
        
        pos= data.iloc[:,510:513]
        mbpos= data.iloc[:,514:517]
        print(pos.shape)

        X = np.asarray(X)
        Y = np.asarray(Y)
        pos = np.asarray(pos)
        mbpos = np.asarray(mbpos)

        # Creamos los conjuntos de datos de entrenamiento y de evaluacion. 
        #test_size = 100 
        test_size = int(np.floor(0.25*X.shape[0]) )
        #sizeofbatch =int(np.floor(0.25*test_size))
        sizeofbatch =int(np.floor(0.75*test_size))
         
        trainX, testX = X[:-test_size], X[-test_size:]
        #trainX = np.reshape(trainX, (1, trainX.shape[0], trainX.shape[1]))
        #testX = np.reshape(testX, (1, testX.shape[0], testX.shape[1]))
        trainY, testY = Y[:-test_size], Y[-test_size:]
        #trainY = np.reshape(trainY, (1, trainY.shape[0], trainY.shape[1]))
        #testYX = np.reshape(testY, (1, testY.shape[0], testY.shape[1]))
        testpos = pos[-test_size:]
        mbtestpos = mbpos[-test_size:]

        chargeT=data['sumPMTs']
        chargeT=chargeT[-test_size:]
        print("lengths:", len(chargeT), len(testY))

        #trainX=trainX[1:]
        #trainY=trainY[1:]

        trainX = np.reshape(trainX, trainX.shape + (1,))
        testX = np.reshape(testX, testX.shape + (1,))


        print(trainX.shape, testX.shape,trainY.shape, testY.shape)
        print("trainX.shape", trainX.shape[0], trainX.shape[1])
        print("trainY.shape", trainY.shape[0], trainY.shape[1])

        
        ####### Estructura de la FFNN ###########

        ####### Si no está cargada la creamos ###########

        if (not modelloaded):
          print('Loading model')
          # 2 capas ocultas con numero de neuronas, definidas en la variable siguiente
          #neurons = [128, 32, 8] 
          #nneurons = len(argv)-3
          #neurons = [int(argv[2]),int(argv[3]),int(argv[4])] 
          #  print("neurons: ",argv[2],argv[3],argv[4])
  
          # Creamos la base del modelo
          model = Sequential() 

          # Ponemos una primera capa oculta 
          model.add(Dense(neurons[0], activation='relu', input_shape=(trainX.shape[1], 1)))
          #print(model.layers[-1].output_shape)
        
          # Incorporamos una segunda capa oculta 
          model.add(Dense(neurons[1], activation='relu'))
          #print(model.layers[-1].output_shape)

          # Incorporamos una terceraa capa oculta 
          model.add(Dense(neurons[2], activation='relu'))
        
          # Aplanamos los datos para reducir la dimensionalidad en la salida
          model.add(Flatten())

          # A\~nadimos la capa de salida de la red con activacion lineal
          #model.add(Dense( trainY.shape[1], activation='tanh'))
          model.add(Dense( trainY.shape[1], activation='sigmoid'))


          # Compilamos el modelo usando el optimizador Adam
          model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
          #model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['binary_crossentropy']) 

        print(model.layers[-1].output_shape)

        # Entrenamos la red
        #model.fit(trainX, trainY, epochs=nepochs, batch_size=1, verbose=2) 
        history=model.fit(trainX, trainY, epochs=nepochs, batch_size=sizeofbatch, validation_data=(testX, testY), verbose=2) 

        # Pronosticos 
        pred = model.predict(testX)

        print("pred.shape: ",pred.shape)
        print("testY.shape: ",testY.shape)

        errors= np.empty([test_size])

        #print('\n\nReal', '        ', 'Pronosticado')
        #for actual, predicted in zip(testY, pred.squeeze()):
                #print(actual.squeeze(), '\t', predicted, '\t',actual.squeeze()-predicted)
        R= np.empty([test_size])
        mbR= np.empty([test_size])

        print(R.shape)
        print('sumPMT format ',chargeT.shape)
        filename = "./r_error_ANN_epochs%d_2HL_%d_%d_%d__%d_%d.txt" % (nepochstot, neurons[0],neurons[1],neurons[2],cutlow,cuthigh)
        fh = open(filename, "w")

                #errors.append(actual.squeeze()-predicted)
        for i in range(test_size):
                #print("X:", testY[i,0],pred[i,0], testY[i,0]-pred[i,0], "\t Y:", testY[i,1],pred[i,1],testY[i,1]-pred[i,1], "\t Z:")
                errors[i]= testY[i]-pred[i]
                #errorsY[i]= testY[i,1]-pred[i,1]
                #errorsZ[i]= testY[i,2]-pred[i,2]
                R[i]= np.sqrt(testpos[i][0]*testpos[i][0]+testpos[i][1]*testpos[i][1]+testpos[i][2]*testpos[i][2])        
                mbR[i]= np.sqrt(mbtestpos[i][0]*mbtestpos[i][0]+mbtestpos[i][1]*mbtestpos[i][1]+mbtestpos[i][2]*mbtestpos[i][2])        
                #print(chargeT.iloc[i])
                #print(float(chargeT.iloc[i]))
                #print(Y[i])
                #print(pred[i])
                myline = "%f %d %f %f %f\n" % (R[i], testY[i], pred[i], float(chargeT.iloc[i]), mbR[i])
                #myline = "%f %d %f %f\n" % (R[i], testY[i], pred[i], float(chargeT.iloc[i]))
                fh.write(myline)
        fh.close()


        # Guardar el modelo, si se quiere
        filename = "./model_ANN_epochs%d_2HL_%d_%d_%d__%d_%d.h5" % (nepochstot, neurons[0],neurons[1],neurons[2],cutlow,cuthigh)
        model.save(filename)

        # Calcular ECM y EAM
        testScoreECM = mean_squared_error(testY, pred)
        print('ECM: %.4f' % (testScoreECM))

        testScoreEAM = mean_absolute_error(testY, pred)
        print('EAM: %.4f' % (testScoreEAM))
        
        # histogram errors 
        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,8))# 6,6
        plt.figure(1)
        #plt.ylim(0,500)
        ax.set_yscale("log")
        plt.hist(errors, 40)
        plt.title('Error X')
        filename = "./histerrors_ANN_epochs%d_2HL_%d_%d_%d__%d_%d.eps" % (nepochstot, neurons[0],neurons[1],neurons[2],cutlow,cuthigh)
        plt.savefig(filename)

        # summarize history for loss
        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,8))# 6,6
        plt.figure(2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        ax.set_yscale("log")
        #plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        filename = "./loss_ANN_epochs%d_2HL_%d_%d_%d__%d_%d.eps" % (nepochstot, neurons[0],neurons[1],neurons[2],cutlow,cuthigh)
        plt.savefig(filename)

        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,8))# 6,6
        plt.figure(3)
        plt.scatter(R, pred, s=0.1, color='black')
        plt.scatter(R, testY, s=0.1, color='red')
        #plt.title('model loss')
        plt.ylabel('IsNeck')
        plt.xlabel('radius')
        plt.ylim(-1.1,1.1)
        plt.legend(['train', 'test'], loc='upper left')
        filename = "./scatterplotr_ANN_epochs%d_2HL_%d_%d_%d__%d_%d.eps" % (nepochstot, neurons[0],neurons[1],neurons[2],cutlow,cuthigh)
        plt.savefig(filename)

        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,8))# 6,6
        plt.figure(4)
        plt.scatter( testY, pred, s=0.1, color='black')
        plt.ylabel('Pred')
        plt.xlabel('True')
        plt.xlim(-0.1,1.1)
        plt.legend(['pred', 'test'], loc='upper left')
        filename = "./scatterplot_TruevsPred_ANN_epochs%d_2HL_%d_%d_%d__%d_%d.eps" % (nepochstot, neurons[0],neurons[1],neurons[2],cutlow,cuthigh)
        plt.savefig(filename)


        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,8))# 6,6
        plt.figure(5)
        plt.scatter(chargeT, errors, s=0.1, color='black')
        #plt.scatter(sumPMTs, testY, s=0.1, color='red')
        #plt.title('model loss')
        plt.ylabel('error')
        plt.xlabel('charge')
        plt.ylim(-1.1,1.1)
        plt.legend(['train', 'test'], loc='upper left')
        filename = "./scatterplotch_ANN_epochs%d_2HL_%d_%d_%d__%d_%d.eps" % (nepochstot, neurons[0],neurons[1],neurons[2],cutlow,cuthigh)
        plt.savefig(filename)


        #plt.show()

"""Invoking the main."""
main()
