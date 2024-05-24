import sys

sys.path.append("components")
sys.path.append("components/stacks")


import torch
import unittest

from components.Generator import Generator
from components.TemporalDiscriminator import TemporalDiscriminator
from components.SpatialDiscriminator import SpatialDiscriminator

from components.stacks.ConditioningStack import ConditioningStack
from components.stacks.LatentConditioningStack import LatentConditioningStack
from components.stacks.OutputStack import OutputStack

from components.stacks.blocks.ConvGRU import ConvGRU
from components.stacks.blocks.D3Block import D3Block
from components.stacks.blocks.DBlock import DBlock
from components.stacks.blocks.GBlock import GBlock
from components.stacks.blocks.LBlock import LBlock
from components.stacks.blocks.S2D import S2D
from components.stacks.blocks.SpatialAttention import SpatialAttention


class TestShapes(unittest.TestCase):
    def test_ConvGRU(self):
        i, h0 = torch.zeros((5, 768, 8, 8)), torch.zeros((5, 384, 8, 8))
        h = ConvGRU(768, 384)(i, h0)
        self.assertEqual(h.shape, torch.Size([5, 384, 8, 8]))

    def test_D3Block(self):
        i = torch.zeros((5, 22, 4, 64, 64))
        o = D3Block(4, 48)(i)
        self.assertEqual(o.shape, torch.Size([5, 11, 48, 32, 32]))

    def test_DBlock(self):
        i = torch.zeros((5, 4, 8, 8))
        o1 = DBlock(4, 48)(i)
        self.assertEqual(o1.shape, torch.Size([5, 48, 4, 4]))

        o2 = DBlock(4, 48, downsampling=False)(i)
        self.assertEqual(o2.shape, torch.Size([5, 48, 8, 8]))

    def test_GBlock(self):
        i = torch.zeros((5, 96, 8, 8))
        o1 = GBlock(96, 48)(i)
        self.assertEqual(o1.shape, torch.Size([5, 48, 16, 16]))

        o2 = GBlock(96, 48, upsampling=False)(i)
        self.assertEqual(o2.shape, torch.Size([5, 48, 8, 8]))

        o3 = GBlock(96, 96)(i)
        self.assertEqual(o3.shape, torch.Size([5, 96, 16, 16]))

    def test_LBlock(self):
        i = torch.zeros((5, 8, 8, 8))
        o = LBlock(8, 24)(i)
        self.assertEqual(o.shape, torch.Size([5, 24, 8, 8]))

    def test_S2D(self):
        i1 = torch.zeros((5, 22, 1, 128, 128))
        o1 = S2D(0.5)(i1)
        self.assertEqual(o1.shape, torch.Size([5, 22, 4, 64, 64]))

        i2 = torch.zeros((5, 1, 128, 128))
        o2 = S2D(0.5)(i2)
        self.assertEqual(o2.shape, torch.Size([5, 4, 64, 64]))

    def test_SpatialAttention(self):
        i = torch.zeros((5, 192, 8, 8))
        o = SpatialAttention(192, 192)(i)
        self.assertEqual(o.shape, torch.Size([5, 192, 8, 8]))

    def test_ConditioningStack(self):
        i = torch.zeros((5, 4, 1, 256, 256))
        o = ConditioningStack()(i)
        shapes = [out.shape for out in o]
        expected_shapes = [
            torch.Size([5, 384, 8, 8]),
            torch.Size([5, 192, 16, 16]),
            torch.Size([5, 96, 32, 32]),
            torch.Size([5, 48, 64, 64]),
        ]
        self.assertEqual(shapes, expected_shapes)

    def test_LatentConditioningStack(self):
        batch_size = 5
        o = LatentConditioningStack(batch_size)()
        self.assertEqual(o.shape, torch.Size([batch_size, 768, 8, 8]))

    def test_OutputStack(self):
        hidden_shapes = [
            torch.Size([5, 384, 8, 8]),
            torch.Size([5, 192, 16, 16]),
            torch.Size([5, 96, 32, 32]),
            torch.Size([5, 48, 64, 64]),
        ]
        h0 = [torch.zeros(shape) for shape in hidden_shapes]
        o, h = OutputStack()(h0)
        self.assertEqual(o.shape, torch.Size([5, 1, 256, 256]))
        self.assertEqual([hidden.shape for hidden in h], hidden_shapes)

    def test_Generator(self):
        i = torch.zeros((2, 4, 1, 256, 256))
        o = Generator(18)(i)
        self.assertEqual(o.shape, torch.Size([2, 18, 256, 256]))

    def test_SpatialDiscriminator(self):
        i = torch.zeros((5, 22, 1, 256, 256))
        o = SpatialDiscriminator(n_frame=8)(i)
        self.assertEqual(o.shape, torch.Size([5, 1]))

    def test_TemporalDiscriminator(self):
        i = torch.zeros((5, 22, 1, 256, 256))
        o = TemporalDiscriminator()(i)
        self.assertEqual(o.shape, torch.Size([5, 1]))


if __name__ == "__main__":
    unittest.main()
