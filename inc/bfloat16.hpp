#pragma once

#include "detail.hpp"

#include <vector>

namespace lp {
class bfloat16 : public detail::arithmetic<bfloat16> {
public:
	bfloat16() = default;

	bfloat16(float value) : value_(round_to_nearest_even(value)) {}

	template <class T>
	explicit bfloat16(T value) : bfloat16(static_cast<float>(value)) {}

	static uint16_t round_to_nearest_even(float value) noexcept {
		const uint32_t bit_pattern = detail::f32_to_u32_bits(value);
		const uint32_t retained_mantissa_lsb = (bit_pattern >> discarded_mantissa_bits) & 1u;
		const uint32_t rounding_bias = retained_mantissa_lsb << (discarded_mantissa_bits - 1);
		return static_cast<uint16_t>((bit_pattern + rounding_bias) >> discarded_mantissa_bits);
	}

	static uint16_t round_to_nearest(float value) noexcept {
		const uint32_t rounding_bias = 1u << (discarded_mantissa_bits - 1);
		return static_cast<uint16_t>((detail::f32_to_u32_bits(value) + rounding_bias) >> discarded_mantissa_bits);
	}

	static uint16_t truncate(float value) noexcept {
		return static_cast<uint16_t>(detail::f32_to_u32_bits(value) >> discarded_mantissa_bits);
	}

	static constexpr bfloat16 from_bits(uint16_t bits) noexcept {
		return bfloat16(bits, true);
	}

	uint16_t to_bits() const noexcept {
		return value_;
	}

	operator float() const noexcept {
		return detail::u32_bits_to_f32(static_cast<uint32_t>(value_) << discarded_mantissa_bits);
	}

	static std::vector<float> to_float_vector(const std::vector<bfloat16>& values) {
		return {values.begin(), values.end()};
	}

	static std::vector<bfloat16> from_float_vector(const std::vector<float>& values) {
		std::vector<bfloat16> result;
		result.reserve(values.size());
		for (float value : values) {
			result.emplace_back(value);
		}
		return result;
	}

private:
	// bf16 keeps the high mantissa bits of f32; the low bits are discarded.
	static constexpr uint32_t discarded_mantissa_bits =
	    detail::layout::f32::mantissa_bits - detail::layout::bf16::mantissa_bits;

	constexpr bfloat16(uint16_t bits, bool) : value_(bits) {}

	uint16_t value_;
};

static_assert(sizeof(bfloat16) == 2, "bfloat16 must occupy two bytes");
}  // namespace lp

namespace std {
template <>
class numeric_limits<lp::bfloat16>
	: public lp::detail::limits<lp::bfloat16, 0x007f, 0x7f7f, 0xff7f, 0x3c00, 0x3f00, 0x7f80, 0x7fc0, 0> {
public:
	static constexpr int digits = 7;
	static constexpr int digits10 = 2;
	static constexpr int min_exponent = -125;
	static constexpr int min_exponent10 = -37;
	static constexpr int max_exponent = 128;
	static constexpr int max_exponent10 = 38;
	static constexpr bool is_signed = true;
	static constexpr bool has_infinity = true;
	static constexpr bool has_quiet_NaN = true;
	static constexpr bool has_signaling_NaN = true;
	static constexpr float_denorm_style has_denorm = denorm_absent;
};
}  // namespace std