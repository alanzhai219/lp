// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include "mxfp4.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>

int main() {
    static_assert(lp::mxfp4::block_size == 32);
    static_assert(lp::mxfp4::packed_size == 16);
    static_assert(sizeof(lp::mxfp4) == 17);

    const lp::mxfp4 zero;
    assert(zero.scale_bits() == 127);
    assert(zero[0] == 0.0f);

    lp::mxfp4::values_type values{};
    values[0] = 6.0f;
    values[1] = -3.0f;
    values[2] = 0.25f;  // Ties to even (zero).
    values[3] = 0.75f;  // Ties to even (one).
    values[4] = -0.0f;
    values[5] = 5.0f;   // Ties to even (four).
    for (std::size_t index = 6; index < values.size(); ++index) {
        values[index] = index % 2 == 0 ? 1.0f : -0.5f;
    }
    const lp::mxfp4 mxfp4_block(values);
    assert(mxfp4_block.scale_bits() == 127);
    assert(mxfp4_block.to_bits()[0] == 0xd7);
    assert(mxfp4_block.to_bits()[1] == 0x20);
    assert(mxfp4_block.to_bits()[15] == 0x92);
    assert(mxfp4_block[0] == 6.0f && mxfp4_block[1] == -3.0f);
    assert(mxfp4_block[2] == 0.0f && mxfp4_block[3] == 1.0f);
    assert(std::signbit(mxfp4_block[4]));
    assert(mxfp4_block[5] == 4.0f);
    for (std::size_t index = 6; index < values.size(); ++index) {
        assert(mxfp4_block[index] == values[index]);
        assert(mxfp4_block.to_array()[index] == values[index]);
    }

    const auto restored = lp::mxfp4::from_bits(mxfp4_block.scale_bits(), mxfp4_block.to_bits());
    assert(restored.to_bits() == mxfp4_block.to_bits());
    assert(restored.to_array() == mxfp4_block.to_array());

    values = {};
    values[0] = 12.0f;
    values[1] = -8.0f;
    values[2] = 14.0f;
    const lp::mxfp4 scaled(values);
    assert(scaled.scale_bits() == 128);
    assert(scaled[0] == 12.0f && scaled[1] == -8.0f);
    assert(scaled[2] == 12.0f);  // Saturates to 6 * 2.
    assert(scaled.to_array()[0] == scaled[0]);
    assert(scaled.to_array()[1] == scaled[1]);

    values = {};
    values[0] = std::ldexp(1.0f, -126);
    values[1] = std::ldexp(1.0f, -149);
    const lp::mxfp4 tiny(values);
    assert(tiny.scale_bits() == 0);
    assert(tiny[0] == values[0]);
    assert(tiny[1] == 0.0f);

    values = {};
    values[0] = std::numeric_limits<float>::quiet_NaN();
    const lp::mxfp4 nan(values);
    assert(nan.scale_bits() == 255);
    assert(std::isnan(nan[0]) && std::isnan(nan[31]));
    assert(std::isnan(nan.to_array()[0]) && std::isnan(nan.to_array()[31]));
    values[0] = std::numeric_limits<float>::infinity();
    assert(lp::mxfp4(values).scale_bits() == 255);

    lp::mxfp4::packed_type packed{};
    packed[0] = 0x8f;
    const auto raw = lp::mxfp4::from_bits(127, packed);
    assert(raw[0] == -6.0f);
    assert(std::signbit(raw[1]));

    // Verify every nibble against the standalone E2M1 decoder at two scales.
    for (uint8_t scale_bits : {uint8_t{127}, uint8_t{128}}) {
        const float scale = static_cast<float>(lp::float8_e8m0::from_bits(scale_bits));
        for (unsigned bits = 0; bits < 16; ++bits) {
            packed[0] = static_cast<uint8_t>(bits | (bits << 4));
            const auto decoded = lp::mxfp4::from_bits(scale_bits, packed);
            const float expected = scale * static_cast<float>(lp::float4_e2m1::from_bits(static_cast<uint8_t>(bits)));
            assert(decoded[0] == expected && decoded[1] == expected);
            assert(std::signbit(decoded[0]) == std::signbit(expected));
            assert(std::signbit(decoded.to_array()[1]) == std::signbit(expected));
        }
    }
}