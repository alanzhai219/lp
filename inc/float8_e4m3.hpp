// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "float16.hpp"

#include <array>

namespace lp {
namespace detail {
inline uint8_t f16_u16_bits_to_e4m3_u8_bits(uint16_t input) noexcept {
	using f16 = layout::f16;
	using f8 = layout::f8e4m3;
	constexpr uint16_t f16_exponent_mask = ((1u << f16::exponent_bits) - 1) << f16::mantissa_bits;
	constexpr uint16_t f16_mantissa_mask = (1u << f16::mantissa_bits) - 1;
	constexpr uint8_t f8_exponent_mask = ((1u << f8::exponent_bits) - 1) << f8::mantissa_bits;
	constexpr uint8_t f8_mantissa_mask = (1u << f8::mantissa_bits) - 1;
	// Left-align the f8 sign/e/m fields in f16; align its mantissa before rounding.
	constexpr uint16_t byte_shift = (f16::sign_bits + f16::exponent_bits + f16::mantissa_bits) -
	                                (f8::sign_bits + f8::exponent_bits + f8::mantissa_bits);
	constexpr uint16_t mantissa_alignment = f16::exponent_bits - f8::exponent_bits;
	constexpr uint16_t aligned_f8_mantissa_mask = static_cast<uint16_t>(f8_mantissa_mask << byte_shift);
	constexpr uint16_t aligned_f8_exponent_mask = static_cast<uint16_t>(f8_exponent_mask << byte_shift);
	// Check the retained mantissa's LSB and discarded bits for ties-to-even.
	constexpr uint16_t round_half_mask = (1u << (byte_shift + 1)) - 1;
	constexpr uint16_t round_normal_mask = (1u << (byte_shift - 1)) - 1;
	constexpr uint16_t round_odd = 3u << (byte_shift - 1);
	constexpr uint16_t round_even = 1u << (byte_shift - 1);
	// Below the halfway exponent for the smallest subnormal, rounding yields zero.
	constexpr int minimum_subnormal_exponent = -static_cast<int>(f8::exponent_bias + f8::mantissa_bits);

	uint8_t result = static_cast<uint8_t>((input & (1u << (f16::exponent_bits + f16::mantissa_bits))) >>
	                                      byte_shift);
	const uint16_t exponent = input & f16_exponent_mask;
	if (exponent == f16_exponent_mask) {
		return static_cast<uint8_t>(result | f8_exponent_mask | f8_mantissa_mask);
	}
	if (!exponent) {
		return result;
	}

	int biased = static_cast<int>(exponent >> f16::mantissa_bits) -
	             static_cast<int>(f16::exponent_bias - f8::exponent_bias);
	uint16_t fraction = static_cast<uint16_t>((input & f16_mantissa_mask) << mantissa_alignment);
	if (biased >= 0) {
		if ((fraction & round_half_mask) == round_odd || (fraction & round_normal_mask)) {
			fraction = static_cast<uint16_t>(fraction + round_even);
			if (fraction & aligned_f8_exponent_mask) {
				++biased;
			}
		}
		fraction &= aligned_f8_mantissa_mask;
	}

	if (biased > (1 << f8::exponent_bits) - 1) {
		return static_cast<uint8_t>(result | f8_exponent_mask | f8_mantissa_mask);
	}
	if (biased > 0) {
		return static_cast<uint8_t>(result | (biased << f8::mantissa_bits) | (fraction >> byte_shift));
	}

	fraction = static_cast<uint16_t>((1u << (f8::mantissa_bits + byte_shift)) |
	                                 ((input & f16_mantissa_mask) << mantissa_alignment));
	const int unbiased = biased - static_cast<int>(f8::exponent_bias);
	const int shift = 1 - unbiased;
	const uint16_t sticky_mask = unbiased < minimum_subnormal_exponent ? 0 :
	                             static_cast<uint16_t>((1 << shift) - 1);
	const bool sticky = (fraction & sticky_mask) != 0;
	fraction = unbiased < minimum_subnormal_exponent ? 0 : static_cast<uint16_t>(fraction >> (1 - biased));
	if (((fraction & round_half_mask) == round_odd && !sticky) || (fraction & round_normal_mask) || sticky) {
		fraction += round_even;
	}
	return static_cast<uint8_t>(result | (fraction >> byte_shift));
}
}  // namespace detail

class float8_e4m3 : public detail::arithmetic<float8_e4m3> {
public:
	float8_e4m3() = default;

	float8_e4m3(uint32_t sign, uint32_t exponent, uint32_t fraction)
		// Pack sign(1) | exponent(4) | mantissa(3).
		: value_(static_cast<uint8_t>(((sign & 1u) << (detail::layout::f8e4m3::exponent_bits +
		                                            detail::layout::f8e4m3::mantissa_bits)) |
		                              ((exponent & ((1u << detail::layout::f8e4m3::exponent_bits) - 1)) <<
		                               detail::layout::f8e4m3::mantissa_bits) |
		                              (fraction & ((1u << detail::layout::f8e4m3::mantissa_bits) - 1)))) {}

	float8_e4m3(float value) : value_(detail::f16_u16_bits_to_e4m3_u8_bits(detail::f32_to_f16_u16_bits(value))) {}

	template <class T>
	explicit float8_e4m3(T value) : float8_e4m3(static_cast<float>(value)) {}

	static constexpr float8_e4m3 from_bits(uint8_t bits) noexcept {
		return float8_e4m3(bits, true);
	}

	uint8_t to_bits() const noexcept {
		return value_;
	}

	operator float() const noexcept {
		using format = detail::layout::f8e4m3;
		// Each index is the encoded exponent and mantissa without the sign bit.
		// Exponent zero is subnormal; the last encoding denotes NaN.
		static constexpr std::array<float, 1u << (format::exponent_bits + format::mantissa_bits)> values{
		    0.0f,     0.001953125f, 0.00390625f, 0.005859375f, 0.0078125f, 0.009765625f, 0.01171875f, 0.013671875f,
		    0.015625f, 0.017578125f, 0.01953125f, 0.021484375f, 0.0234375f, 0.025390625f, 0.02734375f, 0.029296875f,
		    0.03125f,  0.03515625f,  0.0390625f,  0.04296875f,  0.046875f,  0.05078125f,  0.0546875f,  0.05859375f,
		    0.0625f,   0.0703125f,   0.078125f,   0.0859375f,   0.09375f,   0.1015625f,   0.109375f,   0.1171875f,
		    0.125f,    0.140625f,    0.15625f,    0.171875f,    0.1875f,    0.203125f,   0.21875f,    0.234375f,
		    0.25f,     0.28125f,     0.3125f,     0.34375f,     0.375f,     0.40625f,    0.4375f,     0.46875f,
		    0.5f,      0.5625f,      0.625f,      0.6875f,      0.75f,      0.8125f,     0.875f,      0.9375f,
		    1.0f,      1.125f,       1.25f,       1.375f,       1.5f,       1.625f,      1.75f,       1.875f,
		    2.0f,      2.25f,        2.5f,        2.75f,        3.0f,       3.25f,       3.5f,        3.75f,
		    4.0f,      4.5f,         5.0f,        5.5f,         6.0f,       6.5f,        7.0f,        7.5f,
		    8.0f,      9.0f,         10.0f,       11.0f,        12.0f,      13.0f,       14.0f,       15.0f,
		    16.0f,     18.0f,        20.0f,       22.0f,        24.0f,      26.0f,       28.0f,       30.0f,
		    32.0f,     36.0f,        40.0f,       44.0f,        48.0f,      52.0f,       56.0f,       60.0f,
		    64.0f,     72.0f,        80.0f,       88.0f,        96.0f,      104.0f,      112.0f,      120.0f,
		    128.0f,    144.0f,       160.0f,      176.0f,       192.0f,     208.0f,      224.0f,      240.0f,
		    256.0f,    288.0f,       320.0f,      352.0f,       384.0f,     416.0f,      448.0f,
		    std::numeric_limits<float>::quiet_NaN()};
		const uint32_t magnitude_mask = (1u << (format::exponent_bits + format::mantissa_bits)) - 1;
		const uint32_t sign_shift = detail::layout::f32::exponent_bits + detail::layout::f32::mantissa_bits -
		                            (format::exponent_bits + format::mantissa_bits);
		uint32_t bits = detail::f32_to_u32_bits(values[value_ & magnitude_mask]);
		bits |= static_cast<uint32_t>(value_ & (1u << (format::exponent_bits + format::mantissa_bits))) << sign_shift;
		return detail::u32_bits_to_f32(bits);
	}

private:
	constexpr float8_e4m3(uint8_t bits, bool) : value_(bits) {}

	uint8_t value_;
};

static_assert(sizeof(float8_e4m3) == 1, "float8_e4m3 must occupy one byte");
}  // namespace lp

namespace std {
template <>
class numeric_limits<lp::float8_e4m3>
	: public lp::detail::limits<lp::float8_e4m3, 0x08, 0x7e, 0xfe, 0x01, 0x30, 0, 0x7f, 0x01> {
public:
	static constexpr int digits = 4;
	static constexpr int digits10 = 0;
	static constexpr int min_exponent = -5;
	static constexpr int min_exponent10 = -2;
	static constexpr int max_exponent = 10;
	static constexpr int max_exponent10 = 2;
	static constexpr bool is_signed = true;
	static constexpr bool has_infinity = false;
	static constexpr bool has_quiet_NaN = true;
	static constexpr bool has_signaling_NaN = false;
	static constexpr float_denorm_style has_denorm = denorm_present;
};
}  // namespace std