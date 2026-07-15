import mne
import moabb
from moabb.datasets import Dreyer2023
from train_moabb import train_and_evaluate_moabb
from mne import get_config, set_config

if __name__ == "__main__":
    # Point MNE to the downloaded data location
    original_path = get_config("MNE_DATA")
    print(f"The download directory is currently {original_path}")


    # Initialize the Cho2017 dataset
    dataset = Dreyer2023()
    
    # Select subjects to run on. Cho2017 has 52 subjects.
    # We select the first 2 subjects for demonstration/testing.
    subject_list = list(range(1,110))
    
    # Define frequency bands (8-30 Hz is standard for motor imagery)
    bands = [(8, 30)]
    
    # Define channels (empty list would use all, but here we pick standard Motor Imagery channels)
    selected_channels = ["C1",
      "C2",
      "C5",
      "C3",
      "C4",
      "C6",
      "FC3",
      "CP3",
      "FC4",
      "CP4",
      "Cz"]
    
    # Run the training
    print(f"Running classifiers on dataset: {dataset.code}")
    train_and_evaluate_moabb(
        dataset=dataset,
        subject_list=subject_list,
        bands=bands,
        selected_channels=selected_channels,
        output_dir="trained_pipelines"
    )
