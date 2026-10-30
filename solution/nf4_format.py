"""CPU-readable NF4 format notes/reference, not used on the training hot path."""

# Numeric format constants, not an implementation copied from another kernel.
NF4_VALUES = (
    -1.0, -0.6961928009986877, -0.5250730514526367,
    -0.39491748809814453, -0.28444138169288635, -0.18477343022823334,
    -0.09105003625154495, 0.0, 0.07958029955625534,
    0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
)


def decode_reference(packed, count, scales, blocksize, code=NF4_VALUES):
    """Scalar oracle: first value is the HIGH nibble. Scales already decoded."""
    if count < 0 or blocksize <= 0 or len(packed) * 2 < count:
        raise ValueError("invalid packed extent/blocksize")
    if len(scales) < (count + blocksize - 1) // blocksize:
        raise ValueError("missing scales")
    return [
        code[(packed[i // 2] >> (4 if i % 2 == 0 else 0)) & 15]
        * scales[i // blocksize]
        for i in range(count)
    ]


def qwen_projection_shapes():
    """Expected [out, in]; verify these against the actual instantiated model."""
    return {
        "q_proj": (1536, 1536), "k_proj": (256, 1536),
        "v_proj": (256, 1536), "o_proj": (1536, 1536),
        "gate_proj": (8960, 1536), "up_proj": (8960, 1536),
        "down_proj": (1536, 8960),
    }
