from zenml import step
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


@step
def data_preprocessor(df):
    def preproc(image_path, target_size=(64, 64)):
        image = load_img(image_path, target_size=target_size, color_mode='grayscale')  # Resize and convert to grayscale
        image = img_to_array(image)
        image = image.flatten()  # Flatten the image
        image /= 255.0  # Normalize pixel values to [0, 1]
        return image

    df['features'] = df['image_path'].apply(preproc)
    return df