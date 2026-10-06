import kaggle
from pathlib import Path

def download_dataset(dataset_identifier: str, target_folder: str):
    target_path = Path(target_folder)
    target_path.mkdir(parents=True, exist_ok=True)
    
    kaggle.api.authenticate()
    kaggle.api.dataset_download_cli(
        dataset=dataset_identifier,
        path=target_path,
        unzip=True
    )

if __name__ == "__main__":
    # Recuerda cambiar 'tu-usuario' por tu handle real de Kaggle
    DATASET = "javierruanohdez/leafscans-orchard-balanced-fruit-tree-leaves"
    DESTINATION = "data/raw"
    
    download_dataset(DATASET, DESTINATION)