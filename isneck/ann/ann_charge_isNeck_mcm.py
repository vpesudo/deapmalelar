#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

__author__ = ' AUTOR:    Miguel Cardenas Montes'

''' CNN unidimensional en Keras para O3 '''

"""Imports: librerias"""
import os
import math
from math import sqrt
import sys
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

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

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



##########################################
############## Codigo ####################
##########################################

"""Codigo principal"""
def main():

	# Some parameters
	nepochs =75

	# Reading the data
	#data = pd.read_csv('../rawdata/sample.txt', sep="\t", header=None)
	#data = pd.read_csv('../rawdata/lightpatternXYZar40_nominal_30000.txt', sep="\t", header=None)
	data1 = pd.read_csv('../rawdata/sk_ar40_0_100.txt', sep="\t", header=None)
	data2 = pd.read_csv('../rawdata/sk_neck_0_200.txt', sep="\t", header=None)
	data = pd.concat([data1,data2], ignore_index=True)

	print(shape(data1))
	print(shape(data2))

	data = shuffle(data)
	data['sumPMTs']=data.iloc[:,0:255].sum(axis=1)


	print(shape(data))

	cutlow=80
	cuthigh=240

	data = data[(data['sumPMTs'] >= cutlow) & (data['sumPMTs'] <= cuthigh)]


	print(shape(data))
	print(data.columns)


	# Selecting the last column: 1 if neck, 0 if Ar-40 recoil
	Y= data.iloc[:,513:514]
	print(Y.shape)

	X= data.iloc[:,0:255]
	print(X.shape)
	
	pos= data.iloc[:,510:513]
	print(pos.shape)

	X = np.asarray(X)
	Y = np.asarray(Y)
	pos = np.asarray(pos)

	# Creamos los conjuntos de datos de entrenamiento y de evaluacion. 
	#test_size = 100 
	test_size = int(np.floor(0.25*X.shape[0]) )
	#sizeofbatch =int(np.floor(0.25*test_size))
	sizeofbatch =int(np.floor(0.5*test_size))
 	
	trainX, testX = X[:-test_size], X[-test_size:]
	#trainX = np.reshape(trainX, (1, trainX.shape[0], trainX.shape[1]))
	#testX = np.reshape(testX, (1, testX.shape[0], testX.shape[1]))
	trainY, testY = Y[:-test_size], Y[-test_size:]
	#trainY = np.reshape(trainY, (1, trainY.shape[0], trainY.shape[1]))
	#testYX = np.reshape(testY, (1, testY.shape[0], testY.shape[1]))
	testpos = pos[-test_size:]

	#trainX=trainX[1:]
	#trainY=trainY[1:]

	trainX = np.reshape(trainX, trainX.shape + (1,))
	testX = np.reshape(testX, testX.shape + (1,))


	print(trainX.shape, testX.shape,trainY.shape, testY.shape)
	print("trainX.shape", trainX.shape[0], trainX.shape[1])
	print("trainY.shape", trainY.shape[0], trainY.shape[1])

	####### Creamos la estructura de la FFNN ###########
	
	# 2 capas ocultas con nuemro de neuronas, definidas en la variable siguiente
	neurons = [64, 32] 

	# Creamos la base del modelo
	model = Sequential() 

	# Ponemos una primera capa oculta 
	model.add(Dense(neurons[0], activation='relu', input_shape=(trainX.shape[1], 1)))
	print(model.layers[-1].output_shape)
	
	# Incorporamos una segunda capa oculta 
	model.add(Dense(neurons[1], activation='relu'))
	print(model.layers[-1].output_shape)
	
	# Aplanamos los datos para reducir la dimensionalidad en la salida
	model.add(Flatten())

	# A\~nadimos la capa de salida de la red con activacion lineal
	#model.add(Dense( trainY.shape[1], activation='linear'))
	#model.add(Dense( trainY.shape[1], activation='tanh'))
	model.add(Dense( trainY.shape[1], activation='sigmoid'))
	print(model.layers[-1].output_shape)

	# Compilamos el modelo usando el optimizador Adam
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
	#model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['binary_crossentropy']) 

	# Entrenamos la red
	#model.fit(trainX, trainY, epochs=nepochs, batch_size=1, verbose=2) 
	history=model.fit(trainX, trainY, epochs=nepochs, batch_size=sizeofbatch, validation_data=(testX, testY), verbose=2) 
	
	# Pronosticos 
	pred = model.predict(testX)
	print("pred.shape: ",pred.shape)

	errors= np.empty([test_size])

	#print('\n\nReal', '	', 'Pronosticado')
	#for actual, predicted in zip(testY, pred.squeeze()):
		#print(actual.squeeze(), '\t', predicted, '\t',actual.squeeze()-predicted)
		#errors.append(actual.squeeze()-predicted)
	for i in range(test_size):
		#print("X:", testY[i,0],pred[i,0], testY[i,0]-pred[i,0], "\t Y:", testY[i,1],pred[i,1],testY[i,1]-pred[i,1], "\t Z:")
		errors[i]= testY[i]-pred[i]
		#errorsY[i]= testY[i,1]-pred[i,1]
		#errorsZ[i]= testY[i,2]-pred[i,2]

	# Calcular ECM y EAM
	testScoreECM = mean_squared_error(testY, pred)
	print('ECM: %.4f' % (testScoreECM))

	testScoreEAM = mean_absolute_error(testY, pred)
	print('EAM: %.4f' % (testScoreEAM))
	
	R= np.empty([testpos.shape[0]])
	print(R.shape)
	filename = "./r_error_ANN_epochs%d_2HL_%d_%d_%d_%d.txt" % (nepochs, neurons[0],neurons[1],cutlow,cuthigh)
	fh = open(filename, "w")

	for i in range(int(testpos.shape[0])):
		R[i]= np.sqrt(testpos[i][0]*testpos[i][0]+testpos[i][1]*testpos[i][1]+testpos[i][2]*testpos[i][2])	
		#print(R[i])
		#print(Y[i])
		#print(pred[i])
		myline = "%f %d %f\n" % (R[i], testY[i], pred[i])
		fh.write(myline)
		
	fh.close()

	# histogram errors 
	fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,8))# 6,6
	plt.figure(1)
	#plt.ylim(0,500)
	ax.set_yscale("log")
	plt.hist(errors, 40)
	plt.title('Error X')
	filename = "./histerrors_ANN_epochs%d_2HL_%d_%d_%d_%d.eps" % (nepochs, neurons[0],neurons[1],cutlow,cuthigh)
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
	filename = "./loss_ANN_epochs%d_2HL_%d_%d.eps" % (nepochs, neurons[0],neurons[1])
	plt.savefig(filename)

	fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,8))# 6,6
	plt.figure(3)
	plt.scatter(R, errors, s=0.1, color='black')
	plt.scatter(R, testY, s=0.1, color='red')
	#plt.title('model loss')
	plt.ylabel('error')
	plt.xlabel('radious')
	plt.ylim(-1.1,1.1)
	plt.legend(['train', 'test'], loc='upper left')
	filename = "./scatterplot_ANN_epochs%d_2HL_%d_%d_%d_%d.eps" % (nepochs, neurons[0],neurons[1],cutlow,cuthigh)
	plt.savefig(filename)

	fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,8))# 6,6
	plt.figure(4)
	plt.scatter( pred,testY, s=0.1, color='black')
	plt.ylabel('True')
	plt.xlabel('Pred')
	plt.xlim(-0.1,1.1)
	plt.legend(['pred', 'test'], loc='upper right')
	filename = "./scatterplot_TruevsPred_ANN_epochs%d_2HL_%d_%d.eps" % (nepochs, neurons[0],neurons[1])



	plt.show()

"""Invoking the main."""
main()
