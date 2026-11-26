#pragma once

#include "float4_e2m1.hpp"
#include "float8_e8m0.hpp"
#include "mx_detail.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace lp {
// One MXFP4 block: a shared E8M0 scale followed by 32 packed E2M1 values.
// The low nibble of each byte holds the even-indexed element.
class mxfp4 {
public:
	static constexpr std::size_t block_size = 32;
	static constexpr std::size_t packed_size = block_size / 2;
	using values_type = std::array<float, block_size>;
	using packed_type = std::array<uint8_t, packed_size>;

	mxfp4() = default;

	explicit mxfp4(const values_type& values) {
		encode(values);
	}

	static constexpr mxfp4 from_bits(uint8_t scale_bits, const packed_type& packed) noexcept {
		return mxfp4(scale_bits, packed);
	}

	const packed_type& to_bits() const noexcept {
		return packed_;
	}

	float scale() const noexcept {
		return static_cast<float>(float8_e8m0::from_bits(scale_bits_));
	}

	uint8_t scale_bits() const noexcept {
		return scale_bits_;
	}

	float operator[](std::size_t index) const noexcept {
		return decode_idx(index, scale());
	}

	values_type to_array() const noexcept {
		values_type values{};
		const float block_scale = scale();
		decode(values, block_scale);
		return values;
	}

private:
	void encode(const values_type& values) {
		scale_bits_ = detail::mx_scale_bits(values, 2);
		if (scale_bits_ == 0xff) {
			return;
		}

		const float block_scale = scale();
		for (std::size_t index = 0; index < block_size; ++index) {
			const uint8_t bits = float4_e2m1(values[index] / block_scale).to_bits();
			packed_[index / 2] |= static_cast<uint8_t>(bits << (4 * (index % 2)));
		}
	}

    void decode(values_type& values, float block_scale) const noexcept {
        for (std::size_t index = 0; index < block_size; ++index) {
            values[index] = decode_idx(index, block_scale);
        }
    }

	float decode_idx(std::size_t index, float block_scale) const noexcept {
		const uint8_t bits = static_cast<uint8_t>((packed_[index / 2] >> (4 * (index % 2))) & 0x0f);
		return block_scale * detail::e2m1_values[bits];
	}

	constexpr mxfp4(uint8_t scale_bits, const packed_type& packed) noexcept
		: scale_bits_(scale_bits), packed_(packed) {}

	uint8_t scale_bits_{127};
	packed_type packed_{};
};

static_assert(sizeof(mxfp4) == 17, "mxfp4 must occupy 17 bytes");
}  // namespace lp
