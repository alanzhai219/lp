#pragma once

#include "float16.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#ifndef QK_K
#define QK_K 256
#endif

#ifndef K_SCALE_SIZE
#define K_SCALE_SIZE 12
#endif

static_assert(QK_K == 256 && K_SCALE_SIZE == 12, "Q4_K requires GGML's 256-value block and 12 scale bytes");

namespace lp {
namespace detail {
// Match ggml's software binary32 -> binary16 scale conversion, including
// subnormal rounding. Using the project's generic fp16 conversion here
// would not guarantee byte-for-byte reference output at rounding ties.
inline uint16_t q4_k_half_bits(float value) noexcept {
	float base = (std::abs(value) * 0x1.0p+112f) * 0x1.0p-110f;
	const uint32_t bits = f32_to_u32_bits(value);
	const uint32_t doubled = bits + bits;
	const uint32_t sign = bits & 0x80000000u;
	const uint32_t bias = std::max(doubled & 0xff000000u, 0x71000000u);
	base = u32_bits_to_f32((bias >> 1) + 0x07800000u) + base;
	const uint32_t rounded = f32_to_u32_bits(base);
	const uint32_t nonsign = ((rounded >> 13) & 0x7c00u) + (rounded & 0x0fffu);
	return static_cast<uint16_t>((sign >> 16) | (doubled > 0xff000000u ? 0x7e00u : nonsign));
}

// ggml's round-to-nearest-even helper (assuming IEEE-754 binary32).
inline int q4_k_nearest_int(float value) noexcept {
	assert(std::abs(value) <= 4194303.0f);
	const float rounded = value + 12582912.0f;
	int bits;
	std::memcpy(&bits, &rounded, sizeof(bits));
	return (bits & 0x007fffff) - 0x00400000;
}

// Port of ggml's make_qkx2_quants, specialized for Q4_K's 32 weights,
// 15 quantization levels and weighted squared-error search.
inline float q4_k_group_scale(const float* x, const float* weights, uint8_t* levels,
                              float& minimum, uint8_t* candidate) noexcept {
	float low = x[0];
	float high = x[0];
	float sum_w = weights[0];
	float sum_x = sum_w * x[0];
	for (int i = 1; i < 32; ++i) {
		if (x[i] < low) low = x[i];
		if (x[i] > high) high = x[i];
		const float weight = weights[i];
		sum_w += weight;
		sum_x += weight * x[i];
	}
	if (low > 0) low = 0;
	if (high == low) {
		std::fill_n(levels, 32, uint8_t{0});
		minimum = -low;
		return 0.0f;
	}

	float inv_scale = 15.0f / (high - low);
	float scale = 1.0f / inv_scale;
	float best_error = 0.0f;
	for (int i = 0; i < 32; ++i) {
		const int level = std::clamp(q4_k_nearest_int(inv_scale * (x[i] - low)), 0, 15);
		levels[i] = static_cast<uint8_t>(level);
		const float diff = scale * level + low - x[i];
		best_error += weights[i] * (diff * diff);
	}

	for (int step = 0; step <= 20; ++step) {
		inv_scale = (-1.0f + 0.1f * step + 15.0f) / (high - low);
		float sum_l = 0.0f;
		float sum_l2 = 0.0f;
		float sum_xl = 0.0f;
		for (int i = 0; i < 32; ++i) {
			const int level = std::clamp(q4_k_nearest_int(inv_scale * (x[i] - low)), 0, 15);
			candidate[i] = static_cast<uint8_t>(level);
			const float weight = weights[i];
			sum_l += weight * level;
			sum_l2 += weight * level * level;
			sum_xl += weight * level * x[i];
		}
		const float determinant = sum_w * sum_l2 - sum_l * sum_l;
		if (determinant > 0) {
			float trial_scale = (sum_w * sum_xl - sum_x * sum_l) / determinant;
			float trial_min = (sum_l2 * sum_x - sum_l * sum_xl) / determinant;
			if (trial_min > 0) {
				trial_min = 0;
				trial_scale = sum_xl / sum_l2;
			}
			float error = 0.0f;
			for (int i = 0; i < 32; ++i) {
				const float diff = trial_scale * candidate[i] + trial_min - x[i];
				error += weights[i] * (diff * diff);
			}
			if (error < best_error) {
				std::copy_n(candidate, 32, levels);
				best_error = error;
				scale = trial_scale;
				low = trial_min;
			}
		}
	}
	minimum = -low;
	return scale;
}
}  // namespace detail

// GGML-compatible block layout without the unused d/dmin/dm union.
// d: fp16 编码的全局比例系数，每个块的所有权重共享其量化尺度倍数。
// dmin: fp16 编码的全局最小值系数，用于还原权重时作为块级偏移量乘数。
// scales: 12 字节的紧凑表，记录 8 个 32 权重组的 6-bit scale 和 min。
//         每个组在解码时使用 step=d*scale、offset=dmin*min。
// qs: 128 字节的量化数据，每字节存放相邻两个权重的 4-bit 量化值。
// d 和 dmin 以 fp16 位模式保存；序列化假定为小端字节序。
struct block_q4_K {
	uint16_t d;
	uint16_t dmin;
	uint8_t scales[K_SCALE_SIZE];
	uint8_t qs[QK_K / 2];
};
static_assert(sizeof(block_q4_K) == 2 * sizeof(uint16_t) + K_SCALE_SIZE + QK_K / 2,
	          "GGML Q4_K blocks must not contain padding");
static_assert(offsetof(block_q4_K, scales) == 4, "GGML Q4_K scales must follow d and dmin");
static_assert(offsetof(block_q4_K, qs) == 2 * sizeof(uint16_t) + K_SCALE_SIZE,
	          "GGML Q4_K quants must follow the scales");

namespace detail {
// Six-bit scale/min metadata for one 32-weight group.
inline void q4_k_get_scale_min(int group, const uint8_t* packed, uint8_t& scale, uint8_t& min) noexcept {
	if (group < 4) {
		scale = packed[group] & 0x3f;
		min = packed[group + 4] & 0x3f;
	} else {
		scale = static_cast<uint8_t>((packed[group + 4] & 0x0f) | ((packed[group - 4] >> 6) << 4));
		min = static_cast<uint8_t>((packed[group + 4] >> 4) | ((packed[group] >> 6) << 4));
	}
}

inline void q4_k_set_scale_min(int group, uint8_t scale, uint8_t min, uint8_t* packed) noexcept {
	if (group < 4) {
		packed[group] = scale;
		packed[group + 4] = min;
	} else {
		packed[group + 4] = static_cast<uint8_t>((scale & 0x0f) | ((min & 0x0f) << 4));
		packed[group - 4] |= static_cast<uint8_t>((scale >> 4) << 6);
		packed[group] |= static_cast<uint8_t>((min >> 4) << 6);
	}
}

inline uint16_t q4_k_checked_half(float value) {
	const uint16_t bits = q4_k_half_bits(value);
	if (!std::isfinite(static_cast<float>(float16::from_bits(bits)))) {
		throw std::overflow_error("Q4_K scale exceeds fp16 range");
	}
	return bits;
}
}  // namespace detail

// Quantize k float values into k/256 GGML-compatible blocks.
// 过程：
// 1. 每 256 个 float 分成 8 个 32 权重组，分别通过加权误差搜索得到 scale 和 minimum。
// 2. 将 8 个 scale 和 minimum 归一化为 6-bit，再压成 12 字节的 scales 字段。
// 3. 全局 scale 与 minimum 分别保存为 d 和 dmin 的 fp16 位模式。
// 4. 对每个原始权重计算 normalized=(x+dmin*min)/(d*scale)，取整并限制到 0..15。
// 5. 相邻两个 4-bit 量化值合并到同一个 qs 字节；低 4 位和高 4 位分别对应前后半组。
inline void quantize_row_q4_K_ref(const float* x, block_q4_K* y, int64_t k) {
	assert(k >= 0 && k % QK_K == 0);
	for (int64_t block = 0; block < k / QK_K; ++block) {
		block_q4_K& output = y[block];
		output = {};
		uint8_t levels[QK_K]{};
		uint8_t candidate[32]{};
		float weights[32]{};
		float scales[QK_K / 32]{};
		float minima[QK_K / 32]{};
		float max_scale = 0.0f;
		float max_minimum = 0.0f;
		const float* values = x + block * QK_K;

		for (int group = 0; group < QK_K / 32; ++group) {
			float sum_x2 = 0.0f;
			for (int i = 0; i < 32; ++i) {
				const float value = values[group * 32 + i];
				if (!std::isfinite(value)) {
					throw std::invalid_argument("Q4_K cannot encode non-finite values");
				}
				sum_x2 += value * value;
			}
			const float average = std::sqrt(sum_x2 / 32.0f);
			for (int i = 0; i < 32; ++i) {
				weights[i] = average + std::abs(values[group * 32 + i]);
			}
			scales[group] = detail::q4_k_group_scale(values + group * 32, weights,
			                                          levels + group * 32, minima[group], candidate);
			max_scale = std::max(max_scale, scales[group]);
			max_minimum = std::max(max_minimum, minima[group]);
		}

		const float inv_scale = max_scale > 0 ? 63.0f / max_scale : 0.0f;
		const float inv_min = max_minimum > 0 ? 63.0f / max_minimum : 0.0f;
		for (int group = 0; group < QK_K / 32; ++group) {
			const uint8_t scale = static_cast<uint8_t>(std::min(63, detail::q4_k_nearest_int(inv_scale * scales[group])));
			const uint8_t min = static_cast<uint8_t>(std::min(63, detail::q4_k_nearest_int(inv_min * minima[group])));
			detail::q4_k_set_scale_min(group, scale, min, output.scales);
		}
		output.d = detail::q4_k_checked_half(max_scale / 63.0f);
		output.dmin = detail::q4_k_checked_half(max_minimum / 63.0f);
		const float d = static_cast<float>(float16::from_bits(output.d));
		const float dmin = static_cast<float>(float16::from_bits(output.dmin));
		for (int group = 0; group < QK_K / 32; ++group) {
			uint8_t scale;
			uint8_t min;
			detail::q4_k_get_scale_min(group, output.scales, scale, min);
			const float step = d * scale;
			if (step == 0.0f) {
				continue;
			}
			const float offset = dmin * min;
			for (int i = 0; i < 32; ++i) {
				const int index = group * 32 + i;
				const float normalized = (values[index] + offset) / step;
				levels[index] = static_cast<uint8_t>(std::clamp(detail::q4_k_nearest_int(normalized), 0, 15));
			}
		}
		for (int pair = 0; pair < QK_K / 64; ++pair) {
			for (int i = 0; i < 32; ++i) {
				const int index = pair * 64 + i;
				output.qs[pair * 32 + i] = static_cast<uint8_t>(levels[index] | (levels[index + 32] << 4));
			}
		}
	}
}

// Dequantize k/256 GGML-compatible blocks into k float values.
// 过程：
// 1. 将 d 和 dmin 从 fp16 位模式恢复为 float。
// 2. 从 scales 索引并解压每个 32 权重组的 6-bit scale 和 min。
// 3. 分别计算 step=d*scale 和 offset=dmin*min。
// 4. 从 qs 的低 4 位或高 4 位取出 4-bit 量化值。
// 5. 使用 value=step*quant-offset 还原每个 float 权重。
inline void dequantize_row_q4_K(const block_q4_K* x, float* y, int64_t k) noexcept {
	assert(k >= 0 && k % QK_K == 0);
	for (int64_t block = 0; block < k / QK_K; ++block) {
		const block_q4_K& input = x[block];
		const float d = static_cast<float>(float16::from_bits(input.d));
		const float dmin = static_cast<float>(float16::from_bits(input.dmin));
		float* values = y + block * QK_K;
		for (int pair = 0; pair < QK_K / 64; ++pair) {
			for (int half = 0; half < 2; ++half) {
				uint8_t scale;
				uint8_t min;
				detail::q4_k_get_scale_min(pair * 2 + half, input.scales, scale, min);
				const float step = d * scale;
				const float offset = dmin * min;
				for (int i = 0; i < 32; ++i) {
					const uint8_t packed = input.qs[pair * 32 + i];
					const uint8_t quant = half == 0 ? packed & 0x0f : packed >> 4;
					values[pair * 64 + half * 32 + i] = step * quant - offset;
				}
			}
		}
	}
}
}  // namespace lp