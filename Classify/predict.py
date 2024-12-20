import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle

model = load_model('insect_classifier.h5')  #path to the model

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  
    img_array = img_to_array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array

image_folder = 'data_to_test' #path to the images folder to test
results = [] 

for image_name in os.listdir(image_folder):
    if image_name.endswith(( '.jpg')):  
        image_path = os.path.join(image_folder, image_name)
        processed_image = load_and_preprocess_image(image_path)
        prediction = model.predict(processed_image)
        predicted_class_index = np.argmax(prediction) 
        predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]  
        
        results.append({'Image_Name': image_name, 'Predicted_Species': predicted_class}) #to change if you want to make another classification, 'Predicted_Species' to 'Order' or 'Family' etc.

results_df = pd.DataFrame(results)
results_df.to_csv('result.csv', index=False) #path for the result csv

print('Predictions saved')
