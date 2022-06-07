from cgi import test
from fileinput import filename
import os
import shutil
import numpy as np
import glob
import pandas as pd 



def split_data(folder_name, fileNames,label):
    data_csv = np.array([['filename', 'words']])
    for fileName in fileNames:
        shutil.copy(fileName,os.path.join('trainer','all_data',folder_name,folder_name))
        file_name = fileName.split('\\')[-1]
        x = np.where(label == file_name)
        data = np.array([[file_name, label[x[0][0]][1]]])
    #     #print(len(data_csv))
        data_csv = np.append(data_csv,data, axis=0)
    # print(data_csv[1])
    csv = pd.DataFrame(data_csv)
    # csv.to_csv(os.path.join(raw_data_path,folder_name,'labels.csv'),index=None,header=None)
    csv.to_csv(os.path.join('trainer','all_data',folder_name,folder_name,'labels.csv'),index=None,header=None)


raw_data_path = os.path.join('raw_data','train_images')
list_file = glob.glob(os.path.join(raw_data_path,'*.jpg'))

ratio = [0.85, 0.15]
classes = ['training','validation']


#os.makedirs(os.path.join('trainer','all_data',folder_name,folder_name))
for cls in classes:
    if not os.path.exists(os.path.join(os.path.join('trainer','all_data',cls,cls))):
        os.mkdir(os.path.join('trainer','all_data',cls,cls))
    else:
        #if(input('The directory already exists, do you want to delete it?(y/n)') == 'y'):
        shutil.rmtree(os.path.join('trainer','all_data',cls,cls))
        os.mkdir(os.path.join('trainer','all_data',cls,cls))
        #else:exit()

np.random.shuffle(list_file)
training_fileNames = list_file[:int(len(list_file)*ratio[0])]
validation_fileNames = list_file[int(len(list_file)*ratio[0]):]
# testing_fileNames = list_file[int(len(list_file)*(1-ratio[2])):]

print(len(list_file),len(training_fileNames),len(validation_fileNames))

label_path = os.path.join(raw_data_path,'label.txt')

with open(label_path, encoding='utf8') as file:
    label = np.loadtxt(file,dtype=str)

split_data('training',training_fileNames,label)
split_data('validation',validation_fileNames,label)
# split_data('testing',testing_fileNames,label)




test = os.path.join('trainer','all_data')

print(os.listdir(test))