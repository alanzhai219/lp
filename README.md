# lp
lower precision data type

Header-only C++17 low-precision number types and element-type metadata,
independent of OpenVINO. Each numeric type is implemented in its own header;
only common helpers live in `detail.hpp`. Add `inc/` to the compiler's include
path and include
`<bfloat16.hpp>`, `<float16.hpp>`, `<float4_e2m1.hpp>`,
`<float8_e4m3.hpp>`, `<float8_e5m2.hpp>`, `<float8_e8m0.hpp>`, or
`<element_type_traits.hpp>`. The numeric types live in the `lp` namespace; the
element types are in `lp::element`. No additional libraries need to be linked.
Floating-point sign/exponent/mantissa widths and their layout comments are
maintained centrally in `inc/float_layout.hpp`; numeric headers include it
through `detail.hpp`.

The optional `inc_ext/mxfp4.hpp` provides `lp::mxfp4`, a 17-byte block of 32
E2M1 values with one shared E8M0 scale. Add both `inc/` and `inc_ext/` to the
include path. Construct a block from `std::array<float, 32>`, access decoded
values with `operator[]` or `to_array()`, and use `scale_bits()`, `to_bits()`,
and `from_bits()` for the scale byte and 16 packed bytes (even elements in
the low nibble). The scale uses the block maximum's binary exponent minus 2;
all-zero blocks use a scale of 1, and non-finite inputs produce a NaN block.

The optional `inc_ext/mxfp8.hpp` provides `lp::mxfp8_e4m3` and
`lp::mxfp8_e5m2` (aliases of `lp::mxfp8<FP8>`). Each block has 32 FP8 bytes
and one shared E8M0 scale byte, for 33 bytes total. The interface matches
`mxfp4`: construct from `std::array<float, 32>`, use `operator[]` or
`to_array()` to decode, and `scale_bits()`, `to_bits()` and `from_bits()`
for the raw encoding. Scaling uses the block maximum's binary exponent minus
8 for E4M3 or 15 for E5M2; overflow in an FP8 element saturates to its
largest finite value. As with MXFP4, non-finite inputs produce a NaN block.

The optional `inc_ext/q4_k.hpp` provides only `lp::block_q4_K` and the
`quantize_row_q4_K_ref()` / `dequantize_row_q4_K()` functions, like GGML.
Each block holds 256 weights in 144 bytes (4.5 bits/weight). Add both
`inc/` and `inc_ext/` to the include path; pass input/output pointers and
a count divisible by 256 to the row functions. The block has `d`, `dmin`,
`scales`, and `qs` fields (without GGML's unused union).
The byte layout matches GGML Q4_K (little-endian fp16 super-block scales,
12 bytes of packed 6-bit group scales/mins, then 128 bytes of 4-bit quants).
The encoder ports llama.cpp/ggml's `quantize_row_q4_K_ref` weighted search,
six-bit scale/min quantization and fp16 conversion; the decoder follows
`dequantize_row_q4_K`. Non-finite inputs and fp16 scale overflow are rejected
instead of invoking undefined or unusable reference behavior.

Run the standalone smoke test with a C++17 compiler and the `inc/` include
directory. OpenVINO's attribute serialization adapters are intentionally not
part of this independent API.
