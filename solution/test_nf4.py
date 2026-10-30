"""Run on CPU with unittest, or on Kaggle with pytest/unittest."""

import importlib.util
import unittest

from solution.nf4_format import NF4_VALUES, decode_reference, qwen_projection_shapes


class FormatTests(unittest.TestCase):
    def test_every_code_and_nibble_order(self):
        packed = bytes((i << 4) | (i + 1) for i in range(0, 16, 2))
        self.assertEqual(decode_reference(packed, 16, [1.0], 64), list(NF4_VALUES))

    def test_tail_and_block_boundary(self):
        self.assertEqual(decode_reference(b"\xff\xff\xf0", 5, [2.0, 3.0], 3),
                         [2.0, 2.0, 2.0, 3.0, 3.0])

    def test_missing_scale_is_rejected(self):
        with self.assertRaises(ValueError):
            decode_reference(b"\xff", 2, [], 64)

    def test_projection_geometry(self):
        shapes = qwen_projection_shapes()
        self.assertEqual(shapes["v_proj"], (256, 1536))
        self.assertEqual(max(n * k for n, k in shapes.values()), 8960 * 1536)


GPU_AVAILABLE = False
if all(importlib.util.find_spec(name) is not None for name in ("torch", "triton", "bitsandbytes")):
    import torch

    GPU_AVAILABLE = torch.cuda.is_available()


@unittest.skipUnless(GPU_AVAILABLE, "CUDA + torch + triton + bitsandbytes required")
class CudaTests(unittest.TestCase):
    def setUp(self):
        import bitsandbytes.functional as functional
        from solution.nf4 import unpatch_dequantize

        unpatch_dequantize()
        self.functional = functional
        torch.manual_seed(31)

    def tearDown(self):
        from solution.nf4 import unpatch_dequantize

        unpatch_dequantize()

    def test_decode_all_projection_shapes_nested_and_plain(self):
        from solution.nf4 import dequantize_nf4

        shapes = set(qwen_projection_shapes().values()) | {(33, 65), (1, 17)}
        for shape in sorted(shapes):
            for nested in (False, True):
                for dtype in (torch.float16, torch.float32):
                    with self.subTest(shape=shape, nested=nested, dtype=dtype):
                        weight = torch.randn(shape, dtype=dtype, device="cuda")
                        packed, state = self.functional.quantize_4bit(
                            weight, blocksize=64, compress_statistics=nested, quant_type="nf4")
                        # Canonical packed orientation, including the one-byte edge case.
                        reference = self.functional.dequantize_4bit(packed, state)
                        if packed.shape[0] == 1:
                            reference = reference.t()
                        actual = dequantize_nf4(packed, state)
                        torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-7)

    def test_all_nibbles_and_block_sizes(self):
        from solution.nf4 import dequantize_nf4

        for block in (32, 64, 128, 256, 512, 1024, 2048, 4096):
            weight = torch.randn((33, 65), device="cuda", dtype=torch.float32)
            packed, state = self.functional.quantize_4bit(weight, blocksize=block,
                                                         quant_type="nf4")
            pattern = torch.arange(packed.numel(), device="cuda", dtype=torch.int64) % 256
            packed.copy_(pattern.to(torch.uint8).view_as(packed))
            torch.testing.assert_close(dequantize_nf4(packed, state),
                                       self.functional.dequantize_4bit(packed, state),
                                       rtol=1e-6, atol=1e-7)

    def test_out_transpose_fallback_and_idempotence(self):
        from solution.nf4 import patch_dequantize

        weight = torch.randn((32, 64), device="cuda", dtype=torch.float16)
        packed, state = self.functional.quantize_4bit(weight, quant_type="nf4",
                                                     compress_statistics=True)
        original = self.functional.dequantize_4bit
        expected = original(packed, state)
        patch_dequantize()
        once = self.functional.dequantize_4bit
        patch_dequantize()
        self.assertIs(once, self.functional.dequantize_4bit)
        out = torch.empty_like(expected)
        actual = once(packed, state, out=out)
        self.assertEqual(actual.data_ptr(), out.data_ptr())
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(once(packed.t(), state), expected.t(), rtol=1e-6, atol=1e-7)
        fp4, state4 = self.functional.quantize_4bit(weight, quant_type="fp4")
        torch.testing.assert_close(once(fp4, state4), original(fp4, state4))

    def test_scaled_backward_and_two_layers(self):
        """Exercise the actual bnb autograd path with non-contiguous upstream grads."""
        import bitsandbytes as bnb
        from solution.nf4 import patch_dequantize, unpatch_dequantize

        weights = [torch.randn((256, 128), device="cuda", dtype=torch.float16),
                   torch.randn((64, 256), device="cuda", dtype=torch.float16)]
        quantized = [self.functional.quantize_4bit(w, quant_type="nf4", compress_statistics=True)
                     for w in weights]
        x = torch.randn((2, 7, 128), device="cuda", dtype=torch.float16)
        upstream = torch.randn((2, 64, 7), device="cuda", dtype=torch.float16).transpose(1, 2)

        def run(patched):
            unpatch_dequantize()
            if patched:
                patch_dequantize()
            v = x.detach().clone().requires_grad_(True)
            p, s = quantized[0]
            h = bnb.matmul_4bit(v, p, s)
            p, s = quantized[1]
            y = bnb.matmul_4bit(h, p, s)
            y.backward(upstream * 16)  # Incoming gradient is already loss-scaled.
            return y.detach(), v.grad

        expected_y, expected_dx = run(False)
        actual_y, actual_dx = run(True)
        torch.testing.assert_close(actual_y, expected_y, rtol=0, atol=0)
        torch.testing.assert_close(actual_dx, expected_dx, rtol=2e-3, atol=2e-3)


if __name__ == "__main__":
    unittest.main()
