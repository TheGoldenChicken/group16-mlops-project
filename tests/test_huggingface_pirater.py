from hypo_prediction.data.make_dataset import das_huggingface_pirater
import os
import pandas as pd

def test_das_huggingface_pirater():
    """Checks if a valid CSV file is saved."""

    das_huggingface_pirater()
    assert os.path.exists('./data/raw/train.csv')
    assert os.path.exists('./data/raw/validation_matched.csv')
    assert os.path.exists('./data/raw/validation_mismatched.csv')

    # Assert that the files are valid CSV files
    all_files_valid = True
    for file in ['./data/raw/train.csv', './data/raw/validation_matched.csv', './data/raw/validation_mismatched.csv']:
        try:
            pd.read_csv(file)
        except:
            all_files_valid = False
            break
    assert all_files_valid
