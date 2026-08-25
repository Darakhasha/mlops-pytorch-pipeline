import torch

from src.model import get_model


def test_model_output_shape():
    model = get_model(
        architecture="cnn",
        num_classes=10,
    )

    # Fashion-MNIST: batch × channels × height × width
    x = torch.randn(2, 1, 28, 28)

    output = model(x)

    assert output.shape == (2, 10)


def test_model_can_predict():
    model = get_model(
        architecture="cnn",
        num_classes=10,
    )

    # Fashion-MNIST input
    x = torch.randn(1, 1, 28, 28)

    output = model(x)

    prediction = torch.argmax(output, dim=1)

    assert prediction.shape == (1,)
    assert 0 <= prediction.item() < 10