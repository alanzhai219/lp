// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
// Contains conversion logic derived from TensorFlow's float16 implementation.
// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0; see
// https://www.apache.org/licenses/LICENSE-2.0
#pragma once

#include "detail.hpp"

namespace lp {
namespace detail {
inline uint16_t f32_to_f16_u16_bits(float value) noexcept {
	using f32 = layout::f32;
	using f16 = layout::f16;
	// Bias comes from the exponent width; pack f16 into the high half of f32 for conversion.
	constexpr uint32_t f16_output_shift = 32 - (f16::sign_bits + f16::exponent_bits + f16::mantissa_bits);
	constexpr uint32_t f16_exponent_shift = 32 - f16::sign_bits - f16::exponent_bits;
	constexpr uint32_t mantissa_alignment = f16_exponent_shift - f32::mantissa_bits;
	constexpr uint32_t f32_exponent_mask = ((1u << f32::exponent_bits) - 1) << f32::mantissa_bits;
	constexpr uint32_t f32_mantissa_mask = (1u << f32::mantissa_bits) - 1;
	constexpr uint32_t f16_exponent_mask = ((1u << f16::exponent_bits) - 1) << f16_exponent_shift;
	constexpr uint32_t f16_mantissa_mask = ((1u << f16::mantissa_bits) - 1) << f16_output_shift;

	const uint32_t input = f32_to_u32_bits(value);
	constexpr uint32_t sign_mask = 1u << (f32::exponent_bits + f32::mantissa_bits);
	// The retained f16 mantissa starts at bit f16_output_shift; inspect its LSB
	// and the discarded bits to implement round-to-nearest, ties-to-even.
	constexpr uint32_t half_round = (1u << (f16_output_shift + 1)) - 1;
	constexpr uint32_t norm_round = (1u << (f16_output_shift - 1)) - 1;
	constexpr uint32_t round_odd = 3u << (f16_output_shift - 1);
	constexpr uint32_t round_even = 1u << (f16_output_shift - 1);

	const uint32_t exponent = input & f32_exponent_mask;
	uint32_t fraction = (input & f32_mantissa_mask) << mantissa_alignment;
	if (exponent == f32_exponent_mask) {
		if (fraction) {
			fraction &= f16_mantissa_mask;
			if (!fraction) {
				fraction = 1u << f16_output_shift;
			}
		}
		return static_cast<uint16_t>(((input & sign_mask) | f16_exponent_mask | fraction) >> f16_output_shift);
	}
	if (!exponent) {
		return static_cast<uint16_t>((input & sign_mask) >> f16_output_shift);
	}

	int half_exponent = static_cast<int>(exponent >> f32::mantissa_bits) -
	                    static_cast<int>(f32::exponent_bias) + static_cast<int>(f16::exponent_bias);
	// Round the aligned mantissa before checking for carry into the exponent.
	if ((fraction & half_round) == round_odd || (fraction & norm_round)) {
		fraction += round_even;
		if (fraction & f16_exponent_mask) {
			fraction &= f16_exponent_mask;
			++half_exponent;
		}
	}

	fraction &= f16_mantissa_mask;
	if (half_exponent > (1 << f16::exponent_bits) - 2) {
		return static_cast<uint16_t>(((input & sign_mask) | f16_exponent_mask) >> f16_output_shift);
	}
	if (half_exponent > 0) {
		return static_cast<uint16_t>(((input & sign_mask) |
		                              (static_cast<uint32_t>(half_exponent) << f16_exponent_shift) | fraction) >>
		                             f16_output_shift);
	}

	// Restore the implicit leading bit before shifting into the subnormal range.
	fraction = (1u << f16_exponent_shift) | ((input & f32_mantissa_mask) << mantissa_alignment);
	const uint32_t shift = half_exponent < -((1 << f16::exponent_bits) - 2) ? 0u : (1u << (1 - half_exponent));
	const uint32_t sticky = (fraction & (shift - 1)) ? 1u : 0u;
	fraction = 1 - half_exponent > (1 << f16::exponent_bits) - 1 ? 0 : fraction >> (1 - half_exponent);
	fraction |= sticky;
	if ((fraction & half_round) == round_odd || (fraction & norm_round)) {
		fraction += round_even;
	}
	return static_cast<uint16_t>(((input & sign_mask) | fraction) >> f16_output_shift);
}

inline float f16_u16_bits_to_f32(uint16_t input) noexcept {
	// Unpack f16 and rebias its exponent for f32.
	using f32 = layout::f32;
	using f16 = layout::f16;
	constexpr uint32_t f16_output_shift = 32 - (f16::sign_bits + f16::exponent_bits + f16::mantissa_bits);
	constexpr uint32_t f16_exponent_mask = (1u << f16::exponent_bits) - 1;
	constexpr uint32_t f16_mantissa_mask = (1u << f16::mantissa_bits) - 1;
	const uint32_t exponent = (input >> f16::mantissa_bits) & f16_exponent_mask;
	uint32_t fraction = input & f16_mantissa_mask;
	uint32_t float_exponent = exponent + f32::exponent_bias - f16::exponent_bias;
	if (!exponent) {
		if (!fraction) {
			float_exponent = 0;
		} else {
			// Normalize a subnormal mantissa until its leading bit becomes implicit.
			++float_exponent;
			while (!(fraction & (1u << f16::mantissa_bits))) {
				--float_exponent;
				fraction <<= 1;
			}
			fraction &= f16_mantissa_mask;
		}
	} else if (exponent == f16_exponent_mask) {
		float_exponent = (1u << f32::exponent_bits) - 1;
	}
	return u32_bits_to_f32((static_cast<uint32_t>(input & (1u << (f16::mantissa_bits + f16::exponent_bits)))
	                       << f16_output_shift) |
	                       (float_exponent << f32::mantissa_bits) |
	                       (fraction << (f32::mantissa_bits - f16::mantissa_bits)));
}
}  // namespace detail

class float16 : public detail::arithmetic<float16> {
public:
	float16() = default;

	// Pack sign(1) | exponent(5) | mantissa(10) using the shared layout.
	float16(uint32_t sign, uint32_t exponent, uint32_t fraction)
		: value_(static_cast<uint16_t>(((sign & 1u) << (detail::layout::f16::exponent_bits +
		                                             detail::layout::f16::mantissa_bits)) |
		                               ((exponent & ((1u << detail::layout::f16::exponent_bits) - 1)) <<
		                                detail::layout::f16::mantissa_bits) |
		                               (fraction & ((1u << detail::layout::f16::mantissa_bits) - 1)))) {}

	float16(float value) : value_(detail::f32_to_f16_u16_bits(value)) {}

	template <class T>
	explicit float16(T value) : float16(static_cast<float>(value)) {}

	static constexpr float16 from_bits(uint16_t bits) noexcept {
		return float16(bits, true);
	}

	uint16_t to_bits() const noexcept {
		return value_;
	}

	operator float() const noexcept {
		return detail::f16_u16_bits_to_f32(value_);
	}

private:
	constexpr float16(uint16_t bits, bool) : value_(bits) {}

	uint16_t value_;
};

static_assert(sizeof(float16) == 2, "float16 must occupy two bytes");
}  // namespace lp

namespace std {
template <>
class numeric_limits<lp::float16>
	: public lp::detail::limits<lp::float16, 0x0200, 0x7bff, 0xfbff, 0x1200, 0x3c00, 0x7c00, 0x7fff, 0> {
public:
	static constexpr int digits = 11;
	static constexpr int digits10 = 3;
	static constexpr int min_exponent = -13;
	static constexpr int min_exponent10 = -4;
	static constexpr int max_exponent = 16;
	static constexpr int max_exponent10 = 4;
	static constexpr bool is_signed = true;
	static constexpr bool has_infinity = true;
	static constexpr bool has_quiet_NaN = true;
	static constexpr bool has_signaling_NaN = true;
	static constexpr float_denorm_style has_denorm = denorm_absent;

	static constexpr lp::float16 signaling_NaN() noexcept {
		return lp::float16::from_bits(0x7dff);
	}
};
}  // namespace std