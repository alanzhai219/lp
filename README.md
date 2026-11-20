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

Run the standalone smoke test with a C++17 compiler and the `inc/` include
directory. OpenVINO's attribute serialization adapters are intentionally not
part of this independent API.
