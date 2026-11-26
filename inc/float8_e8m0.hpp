#pragma once

#include "detail.hpp"

namespace lp {
namespace detail {
inline uint8_t f32_to_e8m0_u8_bits(float value) noexcept {
	// Round the discarded f32 mantissa into the retained exponent.
	using f32 = layout::f32;
	constexpr uint32_t f32_exponent_mask = (1u << f32::exponent_bits) - 1;
	constexpr uint32_t f32_mantissa_mask = (1u << f32::mantissa_bits) - 1;
	constexpr uint32_t halfway_mantissa = 1u << (f32::mantissa_bits - 1);
	// An f32 subnormal must exceed 3/4 of its mantissa range to round to e=1.
	constexpr uint32_t subnormal_boundary = 3u << (f32::mantissa_bits - 2);

	const uint32_t input = f32_to_u32_bits(value);
	const uint8_t exponent = static_cast<uint8_t>((input >> f32::mantissa_bits) & f32_exponent_mask);
	if (std::signbit(value)) {
		return 0;
	}
	if (exponent >= f32_exponent_mask - 1) {
		return static_cast<uint8_t>(exponent - std::isinf(value));
	}

	const uint32_t mantissa = input & f32_mantissa_mask;
	if (!exponent) {
		return mantissa <= subnormal_boundary ? 0 : 1;
	}
	return static_cast<uint8_t>(exponent +
	                            (mantissa > halfway_mantissa ||
	                             (mantissa == halfway_mantissa && (exponent & 1))));
}
}  // namespace detail

class float8_e8m0 : public detail::arithmetic<float8_e8m0> {
public:
	float8_e8m0() = default;

	float8_e8m0(float value) : value_(detail::f32_to_e8m0_u8_bits(value)) {}

	template <class T>
	explicit float8_e8m0(T value) : float8_e8m0(static_cast<float>(value)) {}

	static constexpr float8_e8m0 from_bits(uint8_t bits) noexcept {
		return float8_e8m0(bits, true);
	}

	uint8_t to_bits() const noexcept {
		return value_;
	}

	operator float() const noexcept {
		// With m=0, the all-ones exponent denotes NaN; e=0 is the half-minimum value.
		if (value_ == (1u << detail::layout::f8e8m0::exponent_bits) - 1) {
			return std::numeric_limits<float>::quiet_NaN();
		}
		if (value_ == 0) {
			return std::numeric_limits<float>::min() / 2;
		}
		// The eight encoded e bits become the f32 exponent; its m bits are zero.
		return detail::u32_bits_to_f32(static_cast<uint32_t>(value_) << detail::layout::f32::mantissa_bits);
	}

private:
	constexpr float8_e8m0(uint8_t bits, bool) : value_(bits) {}

	uint8_t value_{};
};

static_assert(sizeof(float8_e8m0) == 1, "float8_e8m0 must occupy one byte");
}  // namespace lp

namespace std {
template <>
class numeric_limits<lp::float8_e8m0>
	: public lp::detail::limits<lp::float8_e8m0, 0, 0xfe, 0, 0x01, 0x7e, 0, 0xff, 0> {
public:
	static constexpr int digits = 1;
	static constexpr int digits10 = 0;
	static constexpr int min_exponent = -126;
	static constexpr int min_exponent10 = -38;
	static constexpr int max_exponent = 128;
	static constexpr int max_exponent10 = 38;
	static constexpr bool is_signed = false;
	static constexpr bool has_infinity = false;
	static constexpr bool has_quiet_NaN = true;
	static constexpr bool has_signaling_NaN = false;
	static constexpr float_denorm_style has_denorm = denorm_absent;
};
}  // namespace std