import easyocr
import os
import matplotlib.pyplot as plt
path = os.path.join(os.getcwd(),'my_model','new_model')
# reader = easyocr.Reader(lang_list = ['en'],
#                         model_storage_directory = os.path.join('my_model','model'),
#                         user_network_directory = os.path.join('my_model','user_network'),
#                         recog_network  ='new_model',
#                         gpu = False )
reader = easyocr.Reader(lang_list = ['th'],gpu = True )
result = reader.readtext('examples\\thai.jpg') 
print(result[0][1])
img = plt.imread('examples\\thai.jpg')
plt.imshow(img)