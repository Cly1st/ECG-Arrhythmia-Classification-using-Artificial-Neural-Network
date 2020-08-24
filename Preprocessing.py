import csv
import os 
import glob
import numpy as np
import collections
import h5py
from scipy.signal import resample
from sklearn.model_selection import train_test_split

def load_data(path):
	ecg = list()
	with open(path, 'r') as f:
		csv_reader = csv.reader(f, delimiter=',')
		next(csv_reader) #Skip header
		for row in csv_reader:
			ecg.append(int(row[1]))
	return ecg
		
def load_ann(path):
	with open(path, 'r') as f:
		indexs = list()
		symbols = list()
		next(f)
		for row in f:
			inf = row.split()
			indexs.append(int(inf[1]))
			symbols.append(inf[2])
		return indexs, symbols

def patients_filename():
	#Path
	data_path = 'mitbih_database/*.csv' #Dataset folder path "mitbih_database"
	#Retreive File
	Data_Files = [os.path.basename(x) for x in glob.glob(data_path)]
	#Get Files Name
	Patients = list()
	for f in Data_Files:
		p = f.split(".")
		Patients.append(int(p[0]))
	return Patients

def beats_types():
	normal_beats = ['N', 'L', 'R', 'e', 'j']
	sup_beats = ['A', 'a', 'J', 'S']
	ven_beats = ['V', 'E']
	fusion_beats = ['F']
	unknown_beat = ['/', 'f', 'Q']

	return normal_beats, sup_beats, ven_beats, fusion_beats, unknown_beat

def excluded(Dataset):
	Excludes = [102, 104, 107, 217]
	for i in Excludes:
		Dataset.remove(i)
	return Dataset

def split_dataset():
	'''
		Split dataset into training and testing follow AAMI Standard
	'''
	training = [101, 106, 108, 109, 112, 114, 
			115, 116, 118, 119, 122, 124,
			201, 203, 205, 207, 208, 209, 
			215, 220, 223, 230]
	testing = [100, 103, 105, 111, 113, 117, 
			121, 123, 200, 202, 210,212, 
			213, 214, 219, 221, 222, 228, 
			231, 232, 233, 234]

	return training, testing

def Split_Dataset_Types(Dataset):
	if Dataset is None:
		raise ("Input specific dataset file name.")

	Normal, Sup, Ven, Fusion, Unknown = list(), list(), list(), list(), list()
	Beat_types = beats_types()
	Resample = 128
	for t in Dataset:
		Ann_path = f'mitbih_database/{t}annotations.txt'
		Data_path = f'mitbih_database/{t}.csv'

		Raw_ECG = load_data(Data_path)
		R_Indexs, Symbols = load_ann(Ann_path)
		
		Length_RRI = len(R_Indexs)
		
		for L in range(Length_RRI-2):
			Ind1 = int((R_Indexs[L] + R_Indexs[L+1]) / 2) 
			Ind2 = int((R_Indexs[L+1] + R_Indexs[L+2]) / 2)

			Symb = Symbols[L+1]
			Sign = Raw_ECG[Ind1:Ind2]
			Resamp = resample(Sign, Resample)
			plt.plot(Sign)
			
			plt.show()

			plt.plot(Resamp)
			plt.show()
			if Symb in Beat_types[0]:
				Normal.append(np.array(Resamp))
			elif Symb in Beat_types[1]:
				Sup.append(np.array(Resamp))
			elif Symb in Beat_types[2]:
				Ven.append(np.array(Resamp))
			elif Symb in Beat_types[3]:
				Fusion.append(np.array(Resamp))
			elif Symb in Beat_types[4]:
				Unknown.append(np.array(Resamp))

	Normal = np.asarray(Normal, dtype=int)
	Sup = np.asarray(Sup, dtype=int)
	Ven = np.asarray(Ven, dtype=int)
	Fusion = np.asarray(Fusion, dtype=int)
	Unknown = np.asarray(Unknown, dtype=int)

	return Normal, Sup, Ven, Fusion, Unknown

def extract():
	#Remove dataset follow AAMI Standard
	#patients = patients_filename()
	#aami_patients = excluded(patients)

	#Splitted Dataset Names Followed AAMI Standard
	Training, Testing = split_dataset()

	#Segmentation Datasets
	Normal_Train, Sup_Train, Ven_Train, Fusion_Train, Unknown_Train = Split_Dataset_Types(Training)
	Normal_Test, Sup_Test, Ven_Test, Fusion_Test, Unknown_Test = Split_Dataset_Types(Testing)

	print("Training Data Shape: ", Normal_Train.shape, Sup_Train.shape, Ven_Train.shape, Fusion_Train.shape, Unknown_Train.shape)
	print("Testing Data Shape: ",Normal_Test.shape, Sup_Test.shape, Ven_Test.shape, Fusion_Test.shape, Unknown_Test.shape)

	Save_H5py(Normal_Train, Sup_Train, Ven_Train, Fusion_Train, Unknown_Train, Filename="Dataset(Train)")
	Save_H5py(Normal_Test, Sup_Test, Ven_Test, Fusion_Test, Unknown_Test, Filename="Dataset(Test)")
	print("Dataset saving completed!")

def Save_H5py(Normal, Sup, Ven, Fusion, Unknown, Filename):
	f1 = h5py.File(f'{Filename}.hdf5', "w")
	f1.create_dataset("Normal", Normal.shape, dtype='i', data=Normal)
	f1.create_dataset("Sup", Sup.shape, dtype='i', data=Sup)
	f1.create_dataset("Ven", Ven.shape, dtype='i', data=Ven)
	f1.create_dataset("Fusion", Fusion.shape, dtype='i', data=Fusion)
	f1.create_dataset("Unknown", Unknown.shape, dtype='i', data=Unknown)
	f1.close()

def main():
	extract()

if __name__ == '__main__':
	main()