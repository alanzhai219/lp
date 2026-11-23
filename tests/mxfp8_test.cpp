// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include "mxfp8.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

template <class Block, class FP8>
void test_format(float max_value, int max_exponent) {
    static_assert(Block::block_size == 32);
    static_assert(Block::packed_size == 32);
    static_assert(sizeof(Block) == 33);

    const Block zero;
    assert(zero.scale_bits() == 127);
    assert(zero[0] == 0.0f);

    typename Block::values_type values{};
    values[0] = max_value;
    values[1] = -max_value / 2;
    values[2] = 0.0f;
    values[3] = -0.0f;
    values[31] = max_value / 4;
    const Block block(values);
    assert(block.scale_bits() == static_cast<uint8_t>(127 - max_exponent + std::ilogb(max_value)));
    const float scale = static_cast<float>(lp::float8_e8m0::from_bits(block.scale_bits()));
    for (std::size_t index = 0; index < Block::block_size; ++index) {
        assert(block.to_bits()[index] == FP8(values[index] / scale).to_bits());
        assert(block[index] == scale * static_cast<float>(FP8::from_bits(block.to_bits()[index])));
    }
    assert(std::signbit(block[3]));
    assert(block.to_array() == values);

    const auto restored = Block::from_bits(block.scale_bits(), block.to_bits());
    assert(restored.to_bits() == block.to_bits());
    assert(restored.to_array() == block.to_array());

    values = {};
    values[0] = max_value * 4;
    const Block scaled(values);
    assert(scaled.scale_bits() == static_cast<uint8_t>(129 - max_exponent + std::ilogb(max_value)));
    assert(scaled[0] == values[0]);

    // At the top of an exponent bin, the scaled input can exceed FP8 max.
    values = {};
    values[0] = std::ldexp(1.0f, max_exponent + 1) - 1.0f;
    const Block saturated(values);
    assert(saturated.scale_bits() == 127);
    assert(saturated.to_bits()[0] == std::numeric_limits<FP8>::max().to_bits());
    assert(saturated[0] == max_value);

    values[0] = std::numeric_limits<float>::max();
    const Block largest(values);
    assert(largest.scale_bits() != 255);
    assert(std::isfinite(largest[0]));
    assert(largest[0] > 0.0f);

    values = {};
    values[0] = std::ldexp(1.0f, -149);
    const Block tiny(values);
    assert(tiny.scale_bits() == 0);
    assert(tiny[0] == 0.0f);

    values = {};
    values[0] = std::numeric_limits<float>::quiet_NaN();
    const Block nan(values);
    assert(nan.scale_bits() == 255);
    assert(std::isnan(nan[0]) && std::isnan(nan.to_array()[31]));
    values[0] = std::numeric_limits<float>::infinity();
    assert(Block(values).scale_bits() == 255);

    typename Block::packed_type packed{};
    packed[0] = FP8(-1.0f).to_bits();
    packed[1] = 0x80;
    const auto raw = Block::from_bits(128, packed);
    assert(raw[0] == -2.0f);
    assert(std::signbit(raw[1]));
    assert(raw.to_array()[0] == raw[0]);

    packed[0] = std::numeric_limits<FP8>::quiet_NaN().to_bits();
    assert(std::isnan(Block::from_bits(127, packed)[0]));
}

int main() {
    test_format<lp::mxfp8_e4m3, lp::float8_e4m3>(448.0f, 8);
    test_format<lp::mxfp8_e5m2, lp::float8_e5m2>(57344.0f, 15);
}