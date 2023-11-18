import torch

from auto_trader.modeling import model


def test_extractor() -> None:
    layer = model.InceptionExtractor(
        in_channels=2,
        out_channels=3,
        bottleneck_channels=4,
        kernel_sizes=[1, 2],
        num_blocks=2,
        residual=True,
        batchnorm=True,
        dropout=0.1,
    )
    assert layer.output_dim == 3 * (2 + 1) * 2

    x = torch.randn(1, 10, 2)
    y = layer(x)
    assert y.shape == (1, 18)


def test_build_fc_layer() -> None:
    layer = model.build_fc_layer(
        input_dim=8, hidden_dims=[4], batchnorm=True, dropout=0.1, output_dim=2
    )
    x = torch.randn(3, 8)
    y = layer(x)
    assert y.shape == (3, 2)
