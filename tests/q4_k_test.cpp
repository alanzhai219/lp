#include "q4_k.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

int main() {
    static_assert(QK_K == 256 && K_SCALE_SIZE == 12);
    static_assert(sizeof(lp::block_q4_K) == 144);

    std::array<float, QK_K> values{};
    lp::block_q4_K zero{};
    lp::quantize_row_q4_K_ref(values.data(), &zero, values.size());
    std::array<float, QK_K> decoded{};
    lp::dequantize_row_q4_K(&zero, decoded.data(), decoded.size());
    assert(decoded == values);
    const lp::block_q4_K empty{};
    assert(std::memcmp(&zero, &empty, sizeof(zero)) == 0);

    // Independent GGML layout fixture: d=0.5, dmin=0.25, scales/mins
    // 0..3 in the low six bits, 4..7 split between high bits and bytes 8..11.
    lp::block_q4_K raw{};
    raw.d = 0x3800;     // fp16 0.5.
    raw.dmin = 0x3400;  // fp16 0.25.
    const std::array<uint8_t, 8> scales{1, 2, 15, 63, 16, 31, 32, 63};
    const std::array<uint8_t, 8> minima{0, 1, 32, 63, 16, 31, 47, 63};
    for (std::size_t group = 0; group < 4; ++group) {
        raw.scales[group] = static_cast<uint8_t>((scales[group] & 63) | ((scales[group + 4] >> 4) << 6));
        raw.scales[4 + group] = static_cast<uint8_t>((minima[group] & 63) | ((minima[group + 4] >> 4) << 6));
        raw.scales[8 + group] = static_cast<uint8_t>((scales[group + 4] & 15) | ((minima[group + 4] & 15) << 4));
    }
    for (std::size_t pair = 0; pair < 4; ++pair) {
        for (std::size_t i = 0; i < 32; ++i) {
            raw.qs[pair * 32 + i] = static_cast<uint8_t>((i % 16) | ((15 - i % 16) << 4));
        }
    }
    lp::dequantize_row_q4_K(&raw, decoded.data(), decoded.size());
    for (std::size_t group = 0; group < 8; ++group) {
        for (std::size_t i = 0; i < 32; ++i) {
            const float quant = static_cast<float>(group % 2 == 0 ? i % 16 : 15 - i % 16);
            const float expected = 0.5f * scales[group] * quant - 0.25f * minima[group];
            assert(decoded[group * 32 + i] == expected);
        }
    }

    values = {};
    for (std::size_t group = 0; group < 8; ++group) {
        for (std::size_t i = 0; i < 32; ++i) {
            values[group * 32 + i] = (static_cast<float>(i % 16) - 7.0f) * static_cast<float>(group + 1);
        }
    }
    lp::block_q4_K block{};
    lp::quantize_row_q4_K_ref(values.data(), &block, values.size());
    lp::dequantize_row_q4_K(&block, decoded.data(), decoded.size());
    const auto* bytes = reinterpret_cast<const uint8_t*>(&block);
    assert(block.d == static_cast<uint16_t>(bytes[0] | (bytes[1] << 8)));
    assert(block.dmin == static_cast<uint16_t>(bytes[2] | (bytes[3] << 8)));
    assert(block.scales[0] == bytes[4]);
    assert(block.qs[0] == bytes[16]);

    std::array<float, 512> row_values{};
    std::copy(values.begin(), values.end(), row_values.begin());
    std::copy(values.begin(), values.end(), row_values.begin() + 256);
    std::array<lp::block_q4_K, 2> row_blocks{};
    lp::quantize_row_q4_K_ref(row_values.data(), row_blocks.data(), row_values.size());
    assert(std::memcmp(&row_blocks[0], &block, sizeof(block)) == 0);
    assert(std::memcmp(&row_blocks[1], &block, sizeof(block)) == 0);
    std::array<float, 512> row_decoded{};
    lp::dequantize_row_q4_K(row_blocks.data(), row_decoded.data(), row_decoded.size());
    for (std::size_t index = 0; index < row_decoded.size(); ++index) {
        assert(row_decoded[index] == decoded[index % 256]);
    }
    for (std::size_t index = 0; index < values.size(); ++index) {
        assert(std::abs(decoded[index] - values[index]) <= 0.6f * static_cast<float>(index / 32 + 1));
    }

    values.fill(4.0f);
    lp::block_q4_K constant{};
    lp::quantize_row_q4_K_ref(values.data(), &constant, values.size());
    for (std::size_t group = 0; group < 4; ++group) {
        assert(constant.scales[group] == 0xff);
        assert(constant.scales[4 + group] == 0x00);
        assert(constant.scales[8 + group] == 0x0f);
    }
    for (uint8_t quant : constant.qs) {
        assert(quant == 0xff);
    }
    lp::dequantize_row_q4_K(&constant, decoded.data(), decoded.size());
    for (float value : decoded) {
        assert(std::abs(value - 4.0f) < 0.01f);
    }

    values.fill(-4.0f);
    lp::block_q4_K negative{};
    lp::quantize_row_q4_K_ref(values.data(), &negative, values.size());
    for (std::size_t group = 0; group < 4; ++group) {
        assert(negative.scales[group] == 0x00);
        assert(negative.scales[4 + group] == 0xff);
        assert(negative.scales[8 + group] == 0xf0);
    }
    for (uint8_t quant : negative.qs) {
        assert(quant == 0x00);
    }
    lp::dequantize_row_q4_K(&negative, decoded.data(), decoded.size());
    for (float value : decoded) {
        assert(std::abs(value + 4.0f) < 0.01f);
    }

    values.fill(0.0f);
    values[0] = std::numeric_limits<float>::quiet_NaN();
    bool rejected = false;
    try {
        lp::quantize_row_q4_K_ref(values.data(), &block, values.size());
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);
    values[0] = std::numeric_limits<float>::infinity();
    rejected = false;
    try {
        lp::quantize_row_q4_K_ref(values.data(), &block, values.size());
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);

    values[0] = 100000000.0f;
    rejected = false;
    try {
        lp::quantize_row_q4_K_ref(values.data(), &block, values.size());
    } catch (const std::overflow_error&) {
        rejected = true;
    }
    assert(rejected);
}