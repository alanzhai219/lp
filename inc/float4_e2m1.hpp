// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "detail.hpp"

namespace lp {
namespace detail {
inline uint8_t f32_to_e2m1_u8_bits(float value) noexcept {
	using format = layout::f4e2m1;
	// These thresholds select the nearest exponent/mantissa encoding.
	// Midpoints round to even; large values saturate at +/-6.
	const float magnitude = std::abs(value);
	uint8_t result = 0;
	if (magnitude > 0.25f) {
		result = 1;
	}
	if (magnitude >= 0.75f) {
		result = 2;
	}
	if (magnitude > 1.25f) {
		result = 3;
	}
	if (magnitude >= 1.75f) {
		result = 4;
	}
	if (magnitude > 2.5f) {
		result = 5;
	}
	if (magnitude >= 3.5f) {
		result = 6;
	}
	if (magnitude > 5.0f) {
		result = 7;
	}
	// Split the selected nonnegative encoding into its e and m fields.
	const uint8_t exponent = result >> format::mantissa_bits;
	const uint8_t mantissa = result & ((1u << format::mantissa_bits) - 1);
	const uint8_t sign = std::signbit(value) ? 1u : 0u;
	return static_cast<uint8_t>((sign << (format::exponent_bits + format::mantissa_bits)) |
	                            (exponent << format::mantissa_bits) | mantissa);
}
}  // namespace detail

class float4_e2m1 : public detail::arithmetic<float4_e2m1> {
public:
	float4_e2m1() = default;

	float4_e2m1(float value) : value_(detail::f32_to_e2m1_u8_bits(value)) {}

	template <class T>
	explicit float4_e2m1(T value) : float4_e2m1(static_cast<float>(value)) {}

	static constexpr float4_e2m1 from_bits(uint8_t bits) noexcept {
		return float4_e2m1(bits, true);
	}

	uint8_t to_bits() const noexcept {
		return value_;
	}

	operator float() const noexcept {
		using format = detail::layout::f4e2m1;
		// Index by all four encoded bits, including the sign (and negative zero).
		static constexpr float values[] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
		                                   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
		return values[value_ & ((1u << (format::sign_bits + format::exponent_bits + format::mantissa_bits)) - 1)];
	}

private:
	constexpr float4_e2m1(uint8_t bits, bool) : value_(bits) {}

	uint8_t value_;
};

static_assert(sizeof(float4_e2m1) == 1, "float4_e2m1 must occupy one byte");
}  // namespace lp

namespace std {
template <>
class numeric_limits<lp::float4_e2m1>
	: public lp::detail::limits<lp::float4_e2m1, 0x02, 0x07, 0x0f, 0x01, 0x01, 0, 0, 0x01> {
public:
	static constexpr int digits = 2;
	static constexpr int digits10 = 0;
	static constexpr int min_exponent = 1;
	static constexpr int min_exponent10 = 0;
	static constexpr int max_exponent = 3;
	static constexpr int max_exponent10 = 0;
	static constexpr bool is_signed = true;
	static constexpr bool has_infinity = false;
	static constexpr bool has_quiet_NaN = false;
	static constexpr bool has_signaling_NaN = false;
	static constexpr float_denorm_style has_denorm = denorm_present;
};
}  // namespace std