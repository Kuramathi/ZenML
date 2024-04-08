from zenml import step
import os
import pandas as pd

@step
def data_loader(base_path:str) -> pd.DataFrame:
    folders = ['normal', 'lung_opacity', 'covid', 'pneumonia']
    data = []

    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                file_path = os.path.join(folder_path, filename)
                data.append((file_path, folder))

    df_tuples = pd.DataFrame(data, columns=['image_path', 'label'])

    return df_tuples