import torch.nn as nn

from auto_trader.modeling import model


def test_build_cnn_layer():
    conv_layer, output_size = model.build_conv_layer(
        window_size=10,
        in_channel=2,
        out_channels=[3, 4],
        kernel_sizes=[5, 5],
        batchnorm=True,
        dropout=0.1,
    )

    assert isinstance(conv_layer[0], nn.Conv1d)
    assert conv_layer[0].in_channels == 2
    assert conv_layer[0].out_channels == 3

    assert isinstance(conv_layer[1], nn.BatchNorm1d)
    assert conv_layer[1].num_features == 3

    assert isinstance(conv_layer[2], nn.ReLU)

    assert isinstance(conv_layer[3], nn.Dropout)
    assert conv_layer[3].p == 0.1

    assert isinstance(conv_layer[4], nn.MaxPool1d)
    assert conv_layer[4].kernel_size == 2

    assert isinstance(conv_layer[5], nn.Conv1d)
    assert conv_layer[5].in_channels == 3
    assert conv_layer[5].out_channels == 4

    assert isinstance(conv_layer[6], nn.BatchNorm1d)
    assert conv_layer[6].num_features == 4

    assert isinstance(conv_layer[7], nn.ReLU)

    assert isinstance(conv_layer[8], nn.Dropout)
    assert conv_layer[8].p == 0.1

    assert isinstance(conv_layer[9], nn.MaxPool1d)
    assert conv_layer[9].kernel_size == 2

    assert len(conv_layer) == 10
    assert output_size == (10 // 2 // 2)


def test_build_fc_layer():
    fc_layer = model.build_fc_layer(
        in_dim=10,
        hidden_dims=[20, 30],
        batchnorm=True,
        dropout=0.1,
        output_dim=1,
    )

    assert isinstance(fc_layer[0], nn.Linear)
    assert fc_layer[0].in_features == 10
    assert fc_layer[0].out_features == 20

    assert isinstance(fc_layer[1], nn.BatchNorm1d)
    assert fc_layer[1].num_features == 20

    assert isinstance(fc_layer[2], nn.ReLU)

    assert isinstance(fc_layer[3], nn.Dropout)
    assert fc_layer[3].p == 0.1

    assert isinstance(fc_layer[4], nn.Linear)
    assert fc_layer[4].in_features == 20
    assert fc_layer[4].out_features == 30

    assert isinstance(fc_layer[5], nn.BatchNorm1d)
    assert fc_layer[5].num_features == 30

    assert isinstance(fc_layer[6], nn.ReLU)

    assert isinstance(fc_layer[7], nn.Dropout)
    assert fc_layer[7].p == 0.1

    assert isinstance(fc_layer[8], nn.Linear)
    assert fc_layer[8].in_features == 30
    assert fc_layer[8].out_features == 1

    assert len(fc_layer) == 9
