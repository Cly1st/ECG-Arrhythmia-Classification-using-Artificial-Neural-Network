import numpy as np 
import h5py
import matplotlib.pyplot as plt 
import os
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras import layers, Input
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from tensorflow.math import confusion_matrix

def Load_Data(File):
	File = h5py.File(f'{File}.hdf5', 'r')
	Normal = File['Normal']
	Sup = File['Sup']
	Ven = File['Ven']
	Fusion = File['Fusion']
	Unknown = File['Unknown']
	return Normal, Sup, Ven, Fusion, Unknown

def create_labels(N, S, V, F, U):
	N_Labels = np.zeros(len(N), dtype=int)
	S_Labels = np.ones(len(S), dtype=int)
	V_Labels = np.full((len(V)), 2, dtype=int)
	F_Labels = np.full((len(F)), 3, dtype=int)
	U_Labels = np.full((len(U)), 4, dtype=int)

	N_Labels = np.reshape(N_Labels, (-1, 1))
	S_Labels = np.reshape(S_Labels, (-1, 1))
	V_Labels = np.reshape(V_Labels, (-1, 1))
	F_Labels = np.reshape(F_Labels, (-1, 1))
	U_Labels = np.reshape(U_Labels, (-1, 1))

	return N_Labels, S_Labels, V_Labels, F_Labels, U_Labels

def preprocessing(Dimension=None, File=None):
	N, S, V, F, U = Load_Data(File=File)
	NSet, SSet, VSet, FSet, USet = N[:], S[:], V[:], F[:], U[:]
	N_L, S_L, V_L, F_L, U_L = create_labels(NSet, SSet, VSet, FSet, USet)

	if Dimension is not None:
		N_L = np.reshape(N_L, (-1, 1))
		S_L = np.reshape(S_L, (-1, 1))
		V_L = np.reshape(V_L, (-1, 1))
		F_L = np.reshape(F_L, (-1, 1))
		U_L = np.reshape(U_L, (-1, 1))
	Dataset = np.concatenate((NSet, SSet, VSet, FSet, USet), axis=0)
	Labels = np.concatenate((N_L, S_L, V_L, F_L, U_L), axis=0)

	return Dataset, Labels
	#X_train, X_test, y_train, y_test = train_test_split(Dataset, Labels, test_size=0.1)
	#return X_train, X_test, y_train, y_test

def Load_Testing(File):
	N, S, V, F, U = Load_Data(File=File)
	NSet, SSet, VSet, FSet, USet = N[:], S[:], V[:], F[:], U[:]
	N_L, S_L, V_L, F_L, U_L = create_labels(NSet, SSet, VSet, FSet, USet)

	Dataset = np.concatenate((NSet, SSet, VSet, FSet, USet), axis=0)
	Labels = np.concatenate((N_L, S_L, V_L, F_L, U_L), axis=0)

	return Dataset, Labels

def ANN(input_shape=None):
	if input_shape is None:
		raise ("Input shape size for traning!")
	model = Sequential()
	model.add(layers.Dense(128, activation='relu', input_shape=(input_shape)))
	model.add(layers.Dense(128, activation='relu'))
	model.add(layers.Dense(68, activation='relu'))
	model.add(layers.Dense(32, activation='relu'))
	model.add(layers.Dense(16, activation='relu'))
	model.add(layers.Dense(5, activation='softmax'))
	model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
	return model

def plotting(his):
	train_loss = his.history['loss']
	val_loss = his.history['val_loss']
	train_acc = his.history['accuracy']
	val_acc = his.history['val_accuracy']

	fig = plt.figure()
	ax = plt.subplot(111)
	ax.plot(train_loss, label='Train Loss')
	ax.plot(val_loss, label='Validation Loss')
	ax.legend()
	plt.title("Train and Validation Loss")
	plt.show()

	fig = plt.figure()
	ax = plt.subplot(111)
	ax.plot(train_acc, label='Train Accuracy')
	ax.plot(val_acc, label='Validation Accuracy')
	plt.legend()
	plt.title("Train and Validation Accuracy")
	plt.show()

def main():
	clear = lambda: os.system('cls')
	
	#Load training dataset format hdf5
	Path = 'Dataset(Train)'
	Dataset, Labels = preprocessing(File=Path)


	#Train and Validation Dataset Split into 80 | 10
	X_train, X_validation, y_train, y_validation = train_test_split(Dataset, Labels, test_size=0.2)
	
	#Early stopping for prevent overfitting
	earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=20, mode='min', restore_best_weights=True)
	
	input_shape = X_train.shape
	batch_size = 128
	epoch = 150

	#Model
	model = ANN(input_shape=input_shape)

	clear()
	model.summary()

	weight_path = 'AnnWeight.h5'

	#Evaulate testing dataset
	if os.path.exists(weight_path):
		model.load_weights(weight_path)
		path = 'Dataset(Test)'
		Data, Label = preprocessing(File=path)
		loss, acc = model.evaluate(Data, Label)
	#Training training dataset
	else:
		history = model.fit(X_train, y_train, batch_size = batch_size, epochs=epoch, verbose=1, callbacks=[earlystop],
	 						validation_data=(X_validation, y_validation))
	
		plotting(history) #Plot loss and accuracy of training

		model.save('AnnWeight.h5') #Save training weight


if __name__ == '__main__':
	main()