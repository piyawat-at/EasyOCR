import easyocr
import os

path = os.path.join(os.getcwd(),'my_model','new_model')
reader = easyocr.Reader(lang_list = ['en'],
                        model_storage_directory = os.path.join('my_model','model'),
                        user_network_directory = os.path.join('my_model','user_network'),
                        recog_network  ='new_model',
                        gpu = False )
result = reader.readtext('trainer/all_data/traning/en_sample/131.jpg') 
print(result)