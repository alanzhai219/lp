// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace lp {
namespace detail {
// Select a shared power-of-two E8M0 scale, or NaN for non-finite inputs.
// max_exponent is the largest exponent of the element format (2, 8, or 15).
template <std::size_t N>
uint8_t mx_scale_bits(const std::array<float, N>& values, int max_exponent) noexcept {
	float max_magnitude = 0.0f;
	for (float value : values) {
		if (!std::isfinite(value)) {
			return 0xff;  // The E8M0 NaN encoding marks the entire block as NaN.
		}
		max_magnitude = std::max(max_magnitude, std::abs(value));
	}

	if (max_magnitude == 0.0f) {
		return 127;  // An all-zero block uses a scale of one.
	}

	// The smallest E8M0 scale is 2^-127; larger values use the block
	// maximum's binary exponent minus the element format's maximum exponent.
	const int exponent = std::max(std::ilogb(max_magnitude) - max_exponent, -127);
	return static_cast<uint8_t>(exponent + 127);
}
}  // namespace detail
}  // namespace lp