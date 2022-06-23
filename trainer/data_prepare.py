import os
import shutil
import numpy as np
import glob
import pandas as pd 
from tqdm import tqdm

def _copyfileobj_patched(fsrc, fdst, length=16*1024*1024):
    """Patches shutil method to hugely improve copy speed"""
    while 1:
        buf = fsrc.read(length)
        if not buf:
            break
        fdst.write(buf)
shutil.copyfileobj = _copyfileobj_patched

# non-dict txt
# def split_data(folder_name, fileNames,label):
#     data_csv = np.array([['filename', 'words']])
#     for fileName in tqdm(fileNames):
#         shutil.copy(fileName,os.path.join('trainer','all_data',folder_name,folder_name))
#         file_name = fileName.split('\\')[-1]
#         x = np.where(label == file_name)
#         data = np.array([[file_name, label[x[0][0]][1]]])
#         data_csv = np.append(data_csv,data, axis=0)
#     csv = pd.DataFrame(data_csv)
#     # csv.to_csv(os.path.join(raw_data_path,folder_name,'labels.csv'),index=None,header=None)
#     csv.to_csv(os.path.join('trainer','all_data',folder_name,folder_name,'labels.csv'),index=None,header=None)

# dict txt
def split_data(folder_name, fileNames,label):
    data_csv = np.array([['filename', 'words']])
    for fileName in tqdm(fileNames):
        shutil.copy(fileName,os.path.join(os.getcwd(),'all_data',folder_name,folder_name))
        file_name = fileName.split('\\')[-1]
        for d in label:
            if d['filename'] == file_name:
                text = d['text']
                break
        data = np.array([[file_name, text]])
    #     #print(len(data_csv))
        data_csv = np.append(data_csv,data, axis=0)
    # print(data_csv[1])
    csv = pd.DataFrame(data_csv)
    # csv.to_csv(os.path.join(raw_data_path,folder_name,'labels.csv'),index=None,header=None)
    csv.to_csv(os.path.join(os.getcwd(),'all_data',folder_name,folder_name,'labels.csv'),index=None,header=None)


def prepare_data(image_path):
    # raw_data_path = os.path.join('raw_data','train_images_30k')
    # list_file = glob.glob(os.path.join(image_path,'*.jpg'))

    # ratio = [0.90, 0.10]
    # classes = ['training','validation']
    # print()
    # for cls in classes:
    #     if not os.path.exists(os.path.join(os.getcwd(),'all_data',cls,cls)):
    #         os.mkdir(os.path.join(os.getcwd(),'all_data',cls,cls))
    #     else:
    #         if(input('The directory already exists, do you want to delete it?(y/n)') == 'y'):
    #             shutil.rmtree(os.path.join(os.getcwd(),'all_data',cls,cls))
    #             os.mkdir(os.path.join(os.getcwd(),'all_data',cls,cls))
    #         else:exit()

    # np.random.shuffle(list_file)
    # training_fileNames = list_file[:int(len(list_file)*ratio[0])]
    # validation_fileNames = list_file[int(len(list_file)*ratio[0]):]
    # # testing_fileNames = list_file[int(len(list_file)*(1-ratio[2])):]

    label_path = os.path.join(image_path,'label.txt')

    label_list = []
    with open(label_path, encoding='utf8') as file:

        lines = file.readlines()
        for line in lines:
            label_list.append(eval(line))

    split_data('training',training_fileNames,label_list)
    split_data('validation',validation_fileNames,label_list)
