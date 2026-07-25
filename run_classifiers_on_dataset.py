from moabb.datasets import Dreyer2023A, Cho2017, PhysionetMI, Dreyer2023, Stieger2021, Lee2019_MI, Yang2025, GuttmannFlury2025_MI, GuttmannFlury2025_ME
import mne
import moabb
from train_moabb import train_and_evaluate_moabb
from mne import get_config, set_config

if __name__ == "__main__":
    # Point MNE to the downloaded data location
    original_path = get_config("MNE_DATA")
    print(f"The download directory is currently {original_path}")


    # Initialize the dataset
    # dataset = Cho2017()
    # dataset = PhysionetMI()
    dataset = Dreyer2023()
    # dataset = Dreyer2023A()
    # dataset = Stieger2021()
    # dataset = Lee2019_MI()
    # dataset = Yang2025()
    # dataset = GuttmannFlury2025_MI()
    # dataset = GuttmannFlury2025_ME()
    
    # Select subjects to run on. Cho2017 has 52 subjects.
    # We select the first 2 subjects for demonstration/testing.
    subject_list = list(range(1,32))
    subject_list=[15]
    
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
