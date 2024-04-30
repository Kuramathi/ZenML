from zenml import step
import os
from pathlib import Path
import pandas as pd


@step
def data_loader(base_path: str) -> pd.DataFrame:
    base_path = Path(base_path)  # Convert to Path object
    folders = ['normal', 'lung_opacity', 'covid', 'pneumonia']
    data = []

    for folder in folders:
        folder_path = base_path / folder  # Using '/' operator to join paths
        if not folder_path.exists():
            print(f"Folder {folder_path} not found.")
            continue
        for filename in folder_path.iterdir():
            if filename.suffix in ['.jpg', '.png']:
                data.append((str(filename), folder))

    df_tuples = pd.DataFrame(data, columns=['image_path', 'label'])

    return df_tuples
