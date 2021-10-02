# ❤️ ECG-Arrhythmia-Classification-using-Artificial-Neural-Network

# 📝Noted

This project is following the AMMI Standard which used only 44 out of 48 dataset, the dataset were deleted included 102, 104, 107, 217.

Moreover, the AAMI Standard merged dataset from 15 classes into 5 classes contains Normal beat, Supreventricular Ectopic Beat, Ventricular Ectopic beat, Fusion Beat and Unknown Beat.

📘Training (22 Records)

101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230

📘Testing (22 Records) 

100, 103, 105, 11, 113, 117, 121, 123, 200, 202, 210,212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234

✨You can using the already split train dataset and testing dataset in hdf5 format. (You don't need to run the preprocessing code and download dataset).

# 📚Dependency

Numpy

Matplotlib

h5py

Keras 2.4.3

Sklearn 0.22.1 

Tensorflow cpu 2.1.0

# 💾Dataset
The dataset was downloaded from kaggle website; which data store in csv file and annotation file in txt, not original format from physionet website.
https://www.kaggle.com/taejoongyoon/mitbit-arrhythmia-database

# Training results
<img src="https://github.com/Cly1st/ECG-Arrhythmia-Classification-using-Artificial-Neural-Network/blob/master/Images/Accuracy.png" width=350 height= 150>
![alt text](https://github.com/Cly1st/ECG-Arrhythmia-Classification-using-Artificial-Neural-Network/blob/master/Images/Accuracy.png)

# 💌Achknowledgement

https://www.kaggle.com/taejoongyoon/mitbit-arrhythmia-database 

https://www.physionet.org/content/mitdb/1.0.0/
