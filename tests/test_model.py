from hypo_prediction.models.model import IronManWhenHeIsStruckByThorInThatAvengersMovieNotTheSecondObviouslyTheFirst
import os
import torch


def test_model():
    """Checks if the model can be created, saved, and loaded."""

    # Assert that the model gives the expected output shape
    model = IronManWhenHeIsStruckByThorInThatAvengersMovieNotTheSecondObviouslyTheFirst()
    input_img = torch.randn(1, 28, 28)
    output = model(input_img)
    assert output.shape == (1, 10), 'Model output shape is not as expected'

    # Assert that the model can be saved
    torch.save(model.state_dict(), 'models/test_model.pt')
    assert os.path.exists('models/test_model.pt'), 'Model was not saved'

    # Assert that the model can be loaded
    loaded_model = IronManWhenHeIsStruckByThorInThatAvengersMovieNotTheSecondObviouslyTheFirst()
    loaded_model.load_state_dict(torch.load('models/test_model.pt'))
    output = loaded_model(input_img)
    assert output.shape == (1, 10), 'Loaded model output shape is not as expected'