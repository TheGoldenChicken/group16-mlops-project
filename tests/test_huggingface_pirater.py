from hypo_prediction.data.make_dataset import das_huggingface_pirater
import os
import pandas as pd


def test_das_huggingface_pirater():
    """Checks if a valid CSV file is saved."""

    das_huggingface_pirater()

    assert os.path.exists('./data/raw/train.csv'), 'The train.csv file has not been saved'
    assert os.path.exists('./data/raw/validation_matched.csv'), 'The validation_matched.csv file has not been saved'
    assert os.path.exists('./data/raw/validation_mismatched.csv'), 'The validation_mismatched.csv file has not been saved'

    # Assert that the files are valid CSV files
    all_files_valid = True
    for file in ['./data/raw/train.csv', './data/raw/validation_matched.csv', './data/raw/validation_mismatched.csv']:
        try:
            pd.read_csv(file)
        except:
            all_files_valid = False
            break
    assert all_files_valid, 'The CSV files are not valid, since they cannot be read by pandas.read_csv()'
