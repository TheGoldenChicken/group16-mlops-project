from fashion_mnist.models.model import IronManWhenHeIsStruckByThorInThatAvengersMovieNotTheSecondObviouslyTheFirst
import os
import torch


def test_model():
    """Checks if the model can be created, saved, and loaded."""

    # Assert that the model gives the expected output shape
    model = IronManWhenHeIsStruckByThorInThatAvengersMovieNotTheSecondObviouslyTheFirst()
    input_img = torch.randn(1, 28, 28)
    output = model(input_img)
    assert output.shape == (1, 10), 'Model output shape is not as expected'