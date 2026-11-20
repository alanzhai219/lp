// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "float16.hpp"

namespace lp {
namespace detail {
inline uint8_t f16_u16_bits_to_e5m2_u8_bits(uint16_t input) noexcept {
	using f16 = layout::f16;
	using f8 = layout::f8e5m2;
	// Preserve e and round away the lower eight m bits (no infinity clamp).
	// f16 and f8 share the exponent width, so only the low mantissa bits are lost.
	constexpr uint16_t discarded_mantissa_bits = f16::mantissa_bits - f8::mantissa_bits;
	constexpr uint16_t f16_exponent_mask = ((1u << f16::exponent_bits) - 1) << f16::mantissa_bits;
	constexpr uint16_t discarded_mantissa_mask = (1u << discarded_mantissa_bits) - 1;
	constexpr uint16_t halfway = 1u << (discarded_mantissa_bits - 1);

	// Preserve NaN/Inf encodings; otherwise round the discarded m bits to even.
	if ((input & f16_exponent_mask) != f16_exponent_mask) {
		const uint16_t discarded_mantissa = input & discarded_mantissa_mask;
		const uint16_t retained_mantissa_lsb = (input >> discarded_mantissa_bits) & 1u;
		if (discarded_mantissa > halfway ||
		    (discarded_mantissa == halfway && retained_mantissa_lsb)) {
			input = static_cast<uint16_t>(input + (1u << discarded_mantissa_bits));
		}
	}
	return static_cast<uint8_t>(input >> discarded_mantissa_bits);
}
}  // namespace detail

class float8_e5m2 : public detail::arithmetic<float8_e5m2> {
public:
	float8_e5m2() = default;

	float8_e5m2(uint32_t sign, uint32_t exponent, uint32_t fraction)
		// Pack sign(1) | exponent(5) | mantissa(2).
		: value_(static_cast<uint8_t>(((sign & 1u) << (detail::layout::f8e5m2::exponent_bits +
		                                            detail::layout::f8e5m2::mantissa_bits)) |
		                              ((exponent & ((1u << detail::layout::f8e5m2::exponent_bits) - 1)) <<
		                               detail::layout::f8e5m2::mantissa_bits) |
		                              (fraction & ((1u << detail::layout::f8e5m2::mantissa_bits) - 1)))) {}

	float8_e5m2(float value) : value_(detail::f16_u16_bits_to_e5m2_u8_bits(detail::f32_to_f16_u16_bits(value))) {}

	template <class T>
	explicit float8_e5m2(T value) : float8_e5m2(static_cast<float>(value)) {}

	static constexpr float8_e5m2 from_bits(uint8_t bits) noexcept {
		return float8_e5m2(bits, true);
	}

	uint8_t to_bits() const noexcept {
		return value_;
	}

	operator float() const noexcept {
		// Restore the eight truncated mantissa positions of f16; e is unchanged.
		return detail::f16_u16_bits_to_f32(static_cast<uint16_t>(value_) <<
		                             (detail::layout::f16::mantissa_bits - detail::layout::f8e5m2::mantissa_bits));
	}

private:
	constexpr float8_e5m2(uint8_t bits, bool) : value_(bits) {}

	uint8_t value_;
};

static_assert(sizeof(float8_e5m2) == 1, "float8_e5m2 must occupy one byte");
}  // namespace lp

namespace std {
template <>
class numeric_limits<lp::float8_e5m2>
	: public lp::detail::limits<lp::float8_e5m2, 0x04, 0x7b, 0xfb, 0x01, 0x38, 0x7c, 0x7f, 0x01> {
public:
	static constexpr int digits = 3;
	static constexpr int digits10 = 0;
	static constexpr int min_exponent = -13;
	static constexpr int min_exponent10 = -4;
	static constexpr int max_exponent = 16;
	static constexpr int max_exponent10 = 4;
	static constexpr bool is_signed = true;
	static constexpr bool has_infinity = true;
	static constexpr bool has_quiet_NaN = true;
	static constexpr bool has_signaling_NaN = true;
	static constexpr float_denorm_style has_denorm = denorm_present;

	static constexpr lp::float8_e5m2 signaling_NaN() noexcept {
		return lp::float8_e5m2::from_bits(0x7d);
	}
};
}  // namespace std