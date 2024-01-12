from hypo_prediction.data.make_dataset import dataset_preprocessor
import os
import torch


def test_dataset_preprocessor():
    """Checks if the .pt files are saved correctly."""

    dataset_preprocessor('train')
    dataset_preprocessor('validation_matched')
    dataset_preprocessor('validation_mismatched')

    file_paths = ['./data/processed/train-input.pt', './data/processed/train-targets.pt', './data/processed/train-genre.pt', './data/processed/validation_matched-input.pt', './data/processed/validation_matched-targets.pt', './data/processed/validation_matched-genre.pt', './data/processed/validation_mismatched-input.pt', './data/processed/validation_mismatched-targets.pt', './data/processed/validation_mismatched-genre.pt']

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
