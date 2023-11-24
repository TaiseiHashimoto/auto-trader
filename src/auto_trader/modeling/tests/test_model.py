import numpy as np
import torch

from auto_trader.modeling import data, model


def test_build_fc_layer() -> None:
    layer = model.build_fc_layer(
        input_dim=8, hidden_dims=[4], batchnorm=True, dropout=0.1, output_dim=2
    )
    x = torch.randn(3, 8)
    y = layer(x)
    assert y.shape == (3, 2)


def test_block_net() -> None:
    layer = model.BlockNet(
        feature_info={"1min": {}, "2min": {}},
        qkv_kernel_size=5,
        ff_kernel_size=1,
        channels=2,
        ff_channels=4,
        dropout=True,
    )
    x = {
        "1min": torch.randn(1, 2, 10),
        "2min": torch.randn(1, 2, 10),
    }
    actual = layer(x)
    assert list(actual.keys()) == ["1min", "2min"]
    assert actual["1min"].shape == (1, 2, 10)
    assert actual["2min"].shape == (1, 2, 10)


def test_net() -> None:
    layer = model.Net(
        symbol_num=2,
        feature_info={
            "1min": {"sma5": data.FeatureInfo(np.float32)},
            "2min": {"sma5": data.FeatureInfo(np.float32)},
        },
        hist_len=10,
        numerical_emb_dim=4,
        periodic_activation_num_coefs=3,
        periodic_activation_sigma=1.0,
        categorical_emb_dim=4,
        emb_kernel_size=3,
        num_blocks=1,
        block_qkv_kernel_size=5,
        block_ff_kernel_size=5,
        block_channels=4,
        block_ff_channels=6,
        block_dropout=0.1,
        head_hidden_dims=[10],
        head_batchnorm=True,
        head_dropout=0.1,
        head_output_dim=3,
    )
    symbol_idx = torch.tensor([0, 1], dtype=torch.int64)
    features = {
        "1min": {"sma5": torch.randn(2, 10, 1)},
        "2min": {"sma5": torch.randn(2, 10, 1)},
    }
    actual = layer(symbol_idx, features)
    assert actual.shape == (2, 3)
