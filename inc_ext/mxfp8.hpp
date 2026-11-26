#pragma once

#include "float8_e4m3.hpp"
#include "float8_e5m2.hpp"
#include "float8_e8m0.hpp"
#include "mx_detail.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace lp {
// One MXFP8 block: a shared E8M0 scale followed by 32 FP8 elements.
// FP8 must be float8_e4m3 or float8_e5m2.
template <class FP8>
class mxfp8 {
    static_assert(std::is_same_v<FP8, float8_e4m3> || std::is_same_v<FP8, float8_e5m2>,
                  "MXFP8 requires float8_e4m3 or float8_e5m2");

public:
    static constexpr std::size_t block_size = 32;
    static constexpr std::size_t packed_size = block_size;
    using values_type = std::array<float, block_size>;
    using packed_type = std::array<uint8_t, packed_size>;

    mxfp8() = default;

    explicit mxfp8(const values_type& values) {
        encode(values);
    }

    static constexpr mxfp8 from_bits(uint8_t scale_bits, const packed_type& packed) noexcept {
        return mxfp8(scale_bits, packed);
    }

    uint8_t scale_bits() const noexcept {
        return scale_bits_;
    }

    const packed_type& to_bits() const noexcept {
        return packed_;
    }

    float operator[](std::size_t index) const noexcept {
        return decode(index, scale());
    }

    values_type to_array() const noexcept {
        values_type values{};
        const float block_scale = scale();
        for (std::size_t index = 0; index < block_size; ++index) {
            values[index] = decode(index, block_scale);
        }
        return values;
    }

private:
    float scale() const noexcept {
        return static_cast<float>(float8_e8m0::from_bits(scale_bits_));
    }

    void encode(const values_type& values) {
        // The largest finite FP8 exponent is 8 for E4M3 or 15 for E5M2.
        constexpr int max_exponent = std::is_same_v<FP8, float8_e4m3> ? 8 : 15;
        scale_bits_ = detail::mx_scale_bits(values, max_exponent);
        if (scale_bits_ == 0xff) {
            return;
        }

        const float block_scale = scale();
        const float max_value = static_cast<float>(std::numeric_limits<FP8>::max());
        for (std::size_t index = 0; index < block_size; ++index) {
            // Avoid E4M3 NaN and E5M2 infinity when rounding above max.
            const float normalized = values[index] / block_scale;
            const float bounded = std::copysign(std::min(std::abs(normalized), max_value), normalized);
            packed_[index] = FP8(bounded).to_bits();
        }
    }

    float decode(std::size_t index, float block_scale) const noexcept {
        return block_scale * static_cast<float>(FP8::from_bits(packed_[index]));
    }

    constexpr mxfp8(uint8_t scale_bits, const packed_type& packed) noexcept
        : scale_bits_(scale_bits), packed_(packed) {}

    uint8_t scale_bits_{127};
    packed_type packed_{};
};

using mxfp8_e4m3 = mxfp8<float8_e4m3>;
using mxfp8_e5m2 = mxfp8<float8_e5m2>;

static_assert(sizeof(mxfp8_e4m3) == 33, "MXFP8 E4M3 must occupy 33 bytes");
static_assert(sizeof(mxfp8_e5m2) == 33, "MXFP8 E5M2 must occupy 33 bytes");
}  // namespace lp