import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle

#loading excel 
excel_file = 'dataset_classifier.xlsx' #this file should be modified if you use other images
df = pd.read_excel(excel_file)

print(df.head())

#loading images
image_dir = 'dataset_for_training'  #path to the images folder

#map images and labels
image_names = df['Image_Name'].values
labels = df['Scientific_name'].values   #to change if you want to make a classification by order, genus or family

image_label_map = dict(zip(image_names, labels))


#saving of label
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_classes = len(np.unique(encoded_labels))
categorical_labels = to_categorical(encoded_labels, num_classes=num_classes)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)


def load_and_preprocess_image(image_name):
    img_path = os.path.join(image_dir, image_name)
    img = load_img(img_path, target_size=(224, 224))  
    img_array = img_to_array(img) / 255.0 
    return img_array


X = np.array([load_and_preprocess_image(name) for name in image_names])
y = categorical_labels

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

optimizer = Adam(learning_rate=1e-4) 

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

history = model.fit(datagen.flow(X_train, y_train, batch_size=32), 
                    validation_data=(X_val, y_val), 
                    epochs=20,
                    callbacks=[lr_reduction])

loss, accuracy = model.evaluate(X_val, y_val)
print(f'Accuracy on validation set: {accuracy:.2f}')


model.save('insect_classifier.h5') #path for the result model

