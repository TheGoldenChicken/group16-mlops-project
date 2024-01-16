from hypo_prediction.data.make_dataset import data_preprocessor
import os
import torch


def test_data_preprocessor():
    """Checks if the .pt files are saved correctly."""

    data_preprocessor()

    file_paths = ['./data/processed/train.pt', './data/processed/test.pt']

    # Assert that the files are saved
    all_files_saved = True
    for file in file_paths:
        if not os.path.exists(file):
            all_files_saved = False
            break
    assert all_files_saved, 'Not all preprocessed files have been saved'

    # Assert that the files are valid .pt files
    all_files_valid = True
    for file in file_paths:
        try:
            torch.load(file)
        except:
            all_files_valid = False
            break
    assert all_files_valid, 'Not all preprocessed files are valid .pt files'