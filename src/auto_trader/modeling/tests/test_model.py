import torch

from auto_trader.modeling import data, model


def test_build_fc_layer() -> None:
    layer = model.build_fc_layer(
        input_dim=8, hidden_dims=[4], batchnorm=True, dropout=0.1, output_dim=2
    )
    x = torch.randn(3, 8)
    y = layer(x)
    assert y.shape == (3, 2)


def test_net() -> None:
    layer = model.Net(
        feature_stats={"x": data.ContinuousFeatureStats(mean=0.0, std=1.0)},
        hist_len=10,
        continuous_emb_dim=4,
        periodic_activation_num_coefs=3,
        periodic_activation_sigma=1.0,
        categorical_emb_dim=4,
        out_channels=[4],
        kernel_sizes=[5],
        pooling_sizes=[5],
        batchnorm=True,
        layernorm=False,
        dropout=0.1,
        head_hidden_dims=[10],
        head_batchnorm=True,
        head_dropout=0.1,
        head_output_dim=3,
    )
    features = {"x": torch.randn(2, 10, 1)}
    actual = layer(features)
    assert actual.shape == (2, 3)
