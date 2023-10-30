import torch

from auto_trader.modeling import model


def test_conv_extractor() -> None:
    layer = model.ConvExtractor(
        hist_len=10,
        input_channel=2,
        out_channels=[3, 4],
        kernel_sizes=[5, 5],
        batchnorm=True,
        dropout=0.1,
    )
    assert layer.get_output_dim() == 4 * (10 // 2 // 2)

    x = torch.randn(1, 10, 2)
    y = layer(x)
    assert y.shape == (1, 4 * (10 // 2 // 2))


def test_attention_extractor() -> None:
    layer = model.AttentionExtractor(
        hist_len=10,
        emb_dim=4,
        num_layers=2,
        num_heads=2,
        feedforward_dim=8,
        dropout=0.1,
    )
    assert layer.get_output_dim() == 4

    x = torch.randn(1, 10, 4)
    y = layer(x)
    assert y.shape == (1, 4)


def test_build_fc_layer() -> None:
    layer = model.build_fc_layer(
        input_dim=10,
        hidden_dims=[5],
        batchnorm=True,
        dropout=0.1,
        output_dim=2,
    )
    x = torch.randn(2, 10)
    y = layer(x)
    assert y.shape == (2, 2)
