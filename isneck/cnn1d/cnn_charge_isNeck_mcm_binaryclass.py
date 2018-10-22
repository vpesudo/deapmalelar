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
from keras.layers import Convolution1D, MaxPooling1D

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



##########################################
############## Codigo ####################
##########################################

"""Codigo principal"""
def main():

	# Some parameters
	nepochs =50

	# Reading the data
	#data = pd.read_csv('../rawdata/sample.txt', sep="\t", header=None)
	#data = pd.read_csv('../rawdata/lightpatternXYZar40_nominal_30000.txt', sep="\t", header=None)
	data1 = pd.read_csv('../rawdata/mini_sk_ar40_0_100.txt', sep="\t", header=None)
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
	print("Y.shape", Y.shape)
	
	# Enconde binary option A:
	#Y = to_categorical(Y,num_classes=2)
	#print("Y", Y.shape)

	# Encode binary option B: class values as integers
	encoder = LabelEncoder()
	encoder.fit(Y)
	encoded_Y = encoder.transform(Y)
	print("Y",Y)

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
	
	####### Creamos la estructura de la CNN ###########
	####(usamos ReLU's como funciones de activacion)###
	
	# Numero de filtros de convolucion que se aplicaran a las muestras.
	filters = [32, 64, 128] 

	kernel_sz1 = 9 
	kernel_sz2 = 5
	kernel_sz3 = 3

	# Creamos la base del modelo.
	model = Sequential() 
	
	''' La red comienza con entradas de la forma (12,1). Esto se
	debe a que partimos de una entrada con el siguiente formato:
	(muestras, sample_size, dimensionalidad). El primer parametro
	esta implicito, y por tanto nos quedamos con 
	(sample_size, dimensionalidad). Puesto que estamos trabajando 
	con arrays 1D, la dimensionalidad es 1.
	'''
	model.add(Convolution1D(filters[0], kernel_sz1, activation='relu', 
					 input_shape=(trainX.shape[1], 1)))
	'''Cuando se aplica el filtro, obtenemos una salida con dimensiones:
	(sample_size-kernel_sz+1, filters[0]), en este caso (10,32). Ahora 
	agrupamos haciendo uso de un pool_size de 2.'''	

	model.add(MaxPooling1D(pool_size=2))

	'''Dividiendo la primera dimension de la salida anterior por el
	pool_size, obtenemos la dimension actual, que es de (5,32).'''


	'''Ahora a\~nadimos una segunda capa de convolucion, pero partiendo
	de una dimensionalidad inicial de (5,32). Asi, debemos tener cuidado
	en los tama\~nos de filtro y de pool_size en esta segunda capa. 
	Sin embargo, podemos aplicar los mismos tama\~nos que antes sin 
	problemas, como se muestra a continuacion.'''
	
	model.add(Convolution1D(filters[1], kernel_sz2, activation='relu'))
	
	'''Siguiendo lo comentado, ahora obtenemos una salida de la 
	forma (5-3+1,64), o (3,64). Esto nos permite aplicar ahora un 
	pool_size de 3 como mucho. Usamos 2 de nuevo.''' 

	model.add(MaxPooling1D(pool_size=2))

	'''Ahora a\~nadimos una segunda capa de convolucion, pero partiendo
	de una dimensionalidad inicial de (5,32). Asi, debemos tener cuidado
	en los tama\~nos de filtro y de pool_size en esta segunda capa. 
	Sin embargo, podemos aplicar los mismos tama\~nos que antes sin 
	problemas, como se muestra a continuacion.'''

	model.add(Convolution1D(filters[2], kernel_sz3, activation='relu'))
	
	'''Siguiendo lo comentado, ahora obtenemos una salida de la 
	forma (5-3+1,64), o (3,64). Esto nos permite aplicar ahora un 
	pool_size de 3 como mucho. Usamos 2 de nuevo.''' 

	model.add(MaxPooling1D(pool_size=2))

	
	'''Aplanamos los datos para construir una red neuronal propiamente 
	dicha y completamente conectada.'''
	
	model.add(Flatten()) 

	'''Finalmente, a\~nadimos la capa de salida de la red con una funcion 
	de activacion lineal.'''

	# A\~nadimos la capa de salida de la red con activacion sigmoid
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
	print("testY.shape: ",testY.shape)

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
	filename = "./r_error_CNN_epochs%d_2HL_%d_%d_binaryclass.txt" % (nepochs, cutlow,cuthigh)
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
	filename = "./histerrors_CNN_epochs%d_2HL_%d_%d_binaryclass.eps" % (nepochs,cutlow,cuthigh)
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
	filename = "./loss_CNN_epochs%d_2HL_binaryclass.eps" % (nepochs)
	#plt.savefig(filename)

	fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,8))# 6,6
	plt.figure(3)
	plt.scatter(R, errors, s=0.1, color='black')
	plt.scatter(R, testY, s=0.1, color='red')
	#plt.title('model loss')
	plt.ylabel('error')
	plt.xlabel('radious')
	plt.ylim(-1.1,1.1)
	plt.legend(['train', 'test'], loc='upper left')
	filename = "./scatterplot_CNN_epochs%d_2HL_%d_%d_binaryclass.eps" % (nepochs, cutlow,cuthigh)
	#plt.savefig(filename)

	fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,8))# 6,6
	plt.figure(4)
	plt.scatter( testY, pred, s=0.1, color='black')
	plt.ylabel('True')
	plt.xlabel('Pred')
	plt.xlim(-0.1,1.1)
	plt.legend(['pred', 'test'], loc='upper left')
	filename = "./scatterplot_TruevsPred_CNN_epochs%d_2HL_%d_%d_binaryclass.eps" % (nepochs, cutlow, cuthigh)



	plt.show()

"""Invoking the main."""
main()
