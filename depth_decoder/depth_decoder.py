

"""MidashNet: Network for monocular depth estimation trained by mixing several datasets.
This file contains code that is adapted from
https://github.com/thomasjpfan/pytorch_refinenet/blob/master/pytorch_refinenet/refinenet/refinenet_4cascade.py
"""
import torch
import torch.nn as nn

from depth_decoder.blocks import FeatureFusionBlock, Interpolate, _make_encoder_scratch


class MidasNet_decoder(nn.Module):
    """Network for monocular depth estimation.
    """

    def __init__(self, path=None, features=256,non_negative=True):
        super(MidasNet_decoder, self).__init__()
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.

        """
        self.scratch = _make_encoder_scratch(features)

        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
        )

        #Vignesh
        #if path:
            #self.load(path)
            #self.load_state_dict(torch.load(path),strict=False)

    def forward(self, *xs):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """

        layer_1, layer_2, layer_3, layer_4 =  [xs[0][i] for i in range(4)]


        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        print('layer_4_rn',layer_4_rn[0][0])

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)
        print('out',out.size(),out)
        print('out squeeze',torch.squeeze(out, dim=1).size(),torch.squeeze(out, dim=1))

        return torch.squeeze(out, dim=1)

