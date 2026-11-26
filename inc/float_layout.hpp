#pragma once

#include <cstdint>

namespace lp {
namespace detail {
namespace layout {
// f32: sign(1) | exponent(8) | mantissa(23).
struct f32 {
	static constexpr uint32_t sign_bits = 1;
	static constexpr uint32_t exponent_bits = 8;
	static constexpr uint32_t mantissa_bits = 23;
	static constexpr uint32_t exponent_bias = (1u << (exponent_bits - 1)) - 1;
};

// bf16: sign(1) | exponent(8) | mantissa(7); shares f32's exponent.
struct bf16 {
	static constexpr uint32_t sign_bits = 1;
	static constexpr uint32_t exponent_bits = 8;
	static constexpr uint32_t mantissa_bits = 7;
};

// f16: sign(1) | exponent(5) | mantissa(10).
struct f16 {
	static constexpr uint32_t sign_bits = 1;
	static constexpr uint32_t exponent_bits = 5;
	static constexpr uint32_t mantissa_bits = 10;
	static constexpr uint32_t exponent_bias = (1u << (exponent_bits - 1)) - 1;
};

// f8e4m3: sign(1) | exponent(4) | mantissa(3).
struct f8e4m3 {
	static constexpr uint32_t sign_bits = 1;
	static constexpr uint32_t exponent_bits = 4;
	static constexpr uint32_t mantissa_bits = 3;
	static constexpr uint32_t exponent_bias = (1u << (exponent_bits - 1)) - 1;
};

// f8e5m2: sign(1) | exponent(5) | mantissa(2); shares f16's exponent.
struct f8e5m2 {
	static constexpr uint32_t sign_bits = 1;
	static constexpr uint32_t exponent_bits = 5;
	static constexpr uint32_t mantissa_bits = 2;
};

// f4e2m1: sign(1) | exponent(2) | mantissa(1).
struct f4e2m1 {
	static constexpr uint32_t sign_bits = 1;
	static constexpr uint32_t exponent_bits = 2;
	static constexpr uint32_t mantissa_bits = 1;
};

// f8e8m0: exponent(8) | mantissa(0); no sign bit.
struct f8e8m0 {
	static constexpr uint32_t sign_bits = 0;
	static constexpr uint32_t exponent_bits = 8;
	static constexpr uint32_t mantissa_bits = 0;
};
}  // namespace layout
}  // namespace detail
}  // namespace lp