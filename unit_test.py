import unittest

import torch

from lib.conv import GP_conv2D
from lib.gp_dist import GP_dist
from lib.layer import LayerFused


class Test_GP_conv(unittest.TestCase):
    def test_output_shape(self):
        img = torch.rand(20, 150, 150, 3)
        conv2d = GP_conv2D(
            img.shape[1], img.shape[2], img.shape[3], 4, kernel_size=(3, 3)
        )
        out = conv2d.forward(GP_dist.fromTensor(img))
        self.assertTrue(out.mean.dim() == 4)
        self.assertTrue(out.mean.shape == out.var.shape)
        self.assertTrue(out.mean.shape[0] == 20)
        self.assertTrue(out.mean.shape[1] == 148)
        self.assertTrue(out.mean.shape[2] == 148)
        self.assertTrue(out.mean.shape[3] == 4)


class Test_LayerFused(unittest.TestCase):
    def test_output_shape(self):
        tens = torch.rand(20, 32)
        layer = LayerFused(32, 17)
        out = layer.forward(GP_dist.fromTensor(tens))
        self.assertTrue(out.mean.dim() == 2)
        self.assertTrue(out.mean.shape == out.var.shape)
        self.assertTrue(out.mean.shape[0] == 20)
        self.assertTrue(out.mean.shape[1] == 17)


if __name__ == "__main__":
    unittest.main()
