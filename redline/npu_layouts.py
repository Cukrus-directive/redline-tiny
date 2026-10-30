"""Host-side data layouts for the bfp16ebs8 GEMM kernels (numpy ports of gemm_atb_layout.h / helper.h).

Verified element-for-element against loop translations of the C++ helpers at tiles 128x64x128 and 192x128x96.
"""

import numpy as np

V9 = np.dtype("V9")  # one v8bfp16ebs8 block: [shared exp][8 x int8]


def b_layout(b, k, n):
    """layout_transpose_L1_1x2_8x8block: L1 (k, n) tiles column-major; inside, 1x2 super-blocks of col-major 8x8."""
    K, N = b.shape
    x = b.reshape(K // k, k // 8, 8, N // n, n // 16, 2, 8)  # L1r, br, row, L1c, sbc, bis, col
    return x.transpose(3, 0, 4, 1, 5, 6, 2).reshape(-1)


def a_layout(a, m, k):
    """layout_A_L1_2x1_8x8block: L1 (m, k) tiles row-major; inside, 2x1 super-blocks of row-major 8x8."""
    M, K = a.shape
    x = a.reshape(M // m, m // 16, 2, 8, K // k, k // 8, 8)  # L1r, sbr, bis, rib, L1c, bc, cib
    return x.transpose(0, 4, 1, 5, 2, 3, 6).reshape(-1)


def c_unlayout(c, M, N, m, n):
    """layout_inverse_C_L1_2x2_8x8block: L1 (m, n) tiles row-major; inside, 2x2 super-blocks of row-major 8x8."""
    x = c.reshape(M // m, N // n, m // 16, n // 16, 2, 2, 8, 8)  # L1r, L1c, sbr, sbc, br, bc, rib, cib
    return x.transpose(0, 2, 4, 6, 1, 3, 5, 7).reshape(M, N)


def c_unlayout_atb(c, M, N, m, n, rows=4, cols=8):
    """C as the ATB bfp designs (configs 2/3) actually emit it; upstream's layout_inverse_C_L1 has the tile order wrong.

    The runtime sequence walks groups of (rows M-tiles x cols N-tiles), row-group outer; each group drains one chunk
    per core column, and a chunk stacks that column's `rows` core tiles (one per M-tile). Inside a tile: 2x2 8x8.
    """
    x = c.reshape(M // (rows * m), N // (cols * n), cols, rows, m // 16, n // 16, 2, 2, 8, 8)
    # rg, cg, col, row, sbr, sbc, br, bc, rib, cib  ->  rows: rg,row,sbr,br,rib  cols: cg,col,sbc,bc,cib
    return x.transpose(0, 3, 4, 6, 8, 1, 2, 5, 7, 9).reshape(M, N)


def c_layout_atb(c, m, n, rows=4, cols=8):
    """Inverse of c_unlayout_atb: an (M, N) matrix -> the C stream configs 2/3 emit."""
    M, N = c.shape
    x = c.reshape(M // (rows * m), rows, m // 16, 2, 8, N // (cols * n), cols, n // 16, 2, 8)
    # rg,row,sbr,br,rib,cg,col,sbc,bc,cib -> rg,cg,col,row,sbr,sbc,br,bc,rib,cib
    return x.transpose(0, 5, 6, 1, 2, 7, 3, 8, 4, 9).reshape(-1)


def b_unlayout(s, K, N, k, n):
    """Inverse of b_layout: the B stream -> a (K, N) matrix."""
    x = s.reshape(N // n, K // k, n // 16, k // 8, 2, 8, 8)  # L1c, L1r, sbc, br, bis, col, row
    return x.transpose(1, 3, 6, 0, 2, 4, 5).reshape(K, N)


def a_unlayout(s, M, K, m, k):
    """Inverse of a_layout: the A stream -> an (M, K) matrix."""
    x = s.reshape(M // m, K // k, m // 16, k // 8, 2, 8, 8)  # L1r, L1c, sbr, bc, bis, rib, cib
    return x.transpose(0, 2, 4, 5, 1, 3, 6).reshape(M, K)


def to_bfp16(x, truncate=False):
    """[shared exp][8 x int8 mantissa] per block of 8, value = int8 * 2^(exp - 133), exp = the block's largest
    float32 exponent. Rounds half to even, saturating at +127, like the NPU's accum -> bfp16ebs8 conversion
    (conv_even). truncate=True is upstream helper.h floatToBfp16(rounding=truncation): it floors, a -0.5 LSB bias."""
    if not truncate:
        xb = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, 8)
        max_exp = ((xb.view(np.uint32) >> 23) & 0xFF).max(axis=1, keepdims=True)
        val = np.clip(np.rint(xb * np.ldexp(1.0, 133 - max_exp.astype(np.int32))), -128, 127)
        out = np.empty((xb.shape[0], 9), dtype=np.uint8)
        out[:, 0] = max_exp[:, 0].astype(np.uint8)
        out[:, 1:] = val.astype(np.int8).view(np.uint8)
        return out.reshape(-1)
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).reshape(-1, 8)
    exp = (bits >> 23) & 0xFF
    sign = (bits & 0x80000000) != 0
    mant = bits & 0x7FFFFF
    mant = np.where(exp != 0, mant | 0x800000, mant).astype(np.uint32)
    mant = np.where(sign, (~mant + np.uint32(1)), mant)
    val = (mant >> np.uint32(17)).astype(np.uint8).view(np.int8)
    max_exp = exp.max(axis=1, keepdims=True)
    shift = (max_exp - exp).astype(np.int32)
    val = np.where(shift >= 32, np.where(sign, -1, 0), val.astype(np.int32) >> np.minimum(shift, 31))
    out = np.empty((bits.shape[0], 9), dtype=np.uint8)
    out[:, 0] = max_exp[:, 0].astype(np.uint8)
    out[:, 1:] = val.astype(np.int8).view(np.uint8)
    return out.reshape(-1)


def from_bfp16(raw):
    """helper.h bfp16ebs8ToFloat: value = int8 * 2^(shared_exp - 133)."""
    blk = np.asarray(raw, dtype=np.uint8).reshape(-1, 9)
    scale = np.ldexp(1.0, blk[:, :1].astype(np.int32) - 133)
    return (blk[:, 1:].view(np.int8).astype(np.float32) * scale).reshape(-1)
