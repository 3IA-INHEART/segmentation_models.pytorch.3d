import os
import sys
import mock
import pytest
import torch

# mock detection module
sys.modules["torchvision._C"] = mock.Mock()
import segmentation_models_pytorch as smp


def get_encoders_3d():
    exclude_encoders = [
        "senet154",
        "resnext101_32x16d",
        "resnext101_32x32d",
        "resnext101_32x48d",
    ]
    encoders = smp.encoders.get_encoder_names()

    encoders_3D = [e for e in encoders if e not in exclude_encoders and e[-3:] == '_3D']

    return encoders_3D


ENCODERS_3D = get_encoders_3d()

DEFAULT_ENCODER_3D = "resnet18_3D"


def get_sample(model_class):

    if model_class == smp.Unet_3D:
        sample = torch.ones([2, 3, 1, 128, 128])
    else:
        raise ValueError("Not supported model class {}".format(model_class))
    return sample


def _test_forward(model, sample, test_shape=False):
    with torch.no_grad():
        out = model(sample)
    if test_shape:
        assert out.shape[2:] == sample.shape[2:]


def _test_forward_backward(model, sample, test_shape=False):
    out = model(sample)
    out.mean().backward()
    if test_shape:
        assert out.shape[2:] == sample.shape[2:]

@pytest.mark.parametrize("encoder_name", ENCODERS_3D)
@pytest.mark.parametrize("encoder_depth", [3, 5])
@pytest.mark.parametrize("model_class", [smp.Unet_3D])
def test_forward_3D(model_class, encoder_name, encoder_depth, **kwargs):
    decoder_channels = (256, 128, 64, 32, 16)

    model = model_class(
        encoder_name, encoder_depth=encoder_depth, encoder_weights=None, 
        decoder_channels=decoder_channels[:encoder_depth], **kwargs
    )
    sample = get_sample(model_class)
    model.eval()

    if encoder_depth == 5:
        test_shape = True
    else:
        test_shape = False

    _test_forward(model, sample, test_shape)

@pytest.mark.parametrize(
    "model_class",
    [smp.Unet_3D]
)
def test_forward_backward(model_class):
    sample = get_sample(model_class)
    model = model_class(DEFAULT_ENCODER_3D, encoder_weights=None)
    _test_forward_backward(model, sample)

@pytest.mark.parametrize("model_class", [smp.Unet_3D])
def test_aux_output(model_class):
    model = model_class(
        DEFAULT_ENCODER_3D, encoder_weights=None, aux_params=dict(classes=2)
    )
    sample = get_sample(model_class)
    label_size = (sample.shape[0], 2)
    mask, label = model(sample)
    assert label.size() == label_size


@pytest.mark.parametrize("model_class", [smp.Unet_3D])
@pytest.mark.parametrize("encoder_name", ENCODERS_3D)
@pytest.mark.parametrize("in_channels", [3])
@pytest.mark.parametrize("temporary", [1,2,4,5])
def test_in_channels_and_temporary(model_class, encoder_name, in_channels, temporary):
    sample = torch.ones([1, in_channels, temporary, 64, 64])
    model = model_class(DEFAULT_ENCODER_3D, encoder_weights=None, in_channels=in_channels, temporal_size=temporary)
    model.eval()
    with torch.no_grad():
        model(sample)

    assert model.encoder._in_channels == in_channels

if __name__ == "__main__":
    pytest.main([__file__])
