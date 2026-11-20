// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include "bfloat16.hpp"
#include "element_type_traits.hpp"
#include "float16.hpp"
#include "float4_e2m1.hpp"
#include "float8_e4m3.hpp"
#include "float8_e5m2.hpp"
#include "float8_e8m0.hpp"

#include <cassert>
#include <cmath>
#include <limits>
#include <sstream>
#include <type_traits>

int main() {
    static_assert(lp::detail::layout::f32::mantissa_bits == 23);
    static_assert(lp::detail::layout::bf16::mantissa_bits == 7);
    static_assert(lp::detail::layout::f32::mantissa_bits - lp::detail::layout::bf16::mantissa_bits == 16);
    static_assert(lp::detail::layout::f16::mantissa_bits == 10);
    static_assert(lp::detail::layout::f8e4m3::exponent_bits == 4);
    static_assert(lp::detail::layout::f8e5m2::mantissa_bits == 2);
    static_assert(std::is_same_v<lp::fundamental_type_for<lp::element::Type_t::f16>, lp::float16>);
    static_assert(lp::element::from<lp::float8_e8m0>() == lp::element::f8e8m0);
    static_assert(sizeof(lp::float16) == 2 && sizeof(lp::float4_e2m1) == 1);
    assert(lp::float16(1.0f).to_bits() == 0x3c00);
    assert(lp::float16(-2.0f).to_bits() == 0xc000);
    assert(lp::float16(0.5f).to_bits() == 0x3800);
    assert(lp::float16(0, 15, 0).to_bits() == lp::float16(1.0f).to_bits());
    assert(lp::float16(1.0f + 1.0f / 2048).to_bits() == 0x3c00);
    assert(lp::float16(1.0f + 3.0f / 2048).to_bits() == 0x3c02);
    assert(lp::float16(std::ldexp(1.0f, -24)).to_bits() == 0x0001);
    assert(std::isnan(static_cast<float>(std::numeric_limits<lp::float16>::quiet_NaN())));
    assert(lp::bfloat16(1.0f).to_bits() == 0x3f80);
    const float halfway_bf16 = lp::detail::u32_bits_to_f32(0x3f808000);
    assert(lp::bfloat16::round_to_nearest(halfway_bf16) == 0x3f81);
    assert(lp::bfloat16::round_to_nearest_even(halfway_bf16) == 0x3f80);
    assert(lp::bfloat16::truncate(halfway_bf16) == 0x3f80);
    assert(lp::float8_e4m3(1.0f).to_bits() == 0x38);
    assert(lp::float8_e4m3(0, 7, 0).to_bits() == lp::float8_e4m3(1.0f).to_bits());
    assert(lp::float8_e4m3(1.0625f).to_bits() == 0x38);
    assert(lp::float8_e4m3(1.1875f).to_bits() == 0x3a);
    assert(lp::float8_e4m3(std::ldexp(1.0f, -9)).to_bits() == 0x01);
    assert(std::signbit(static_cast<float>(lp::float8_e4m3::from_bits(0x80))));
    assert(std::isnan(static_cast<float>(lp::float8_e4m3::from_bits(0xff))));
    assert(std::signbit(static_cast<float>(lp::float8_e4m3::from_bits(0xff))));
    assert(lp::float8_e5m2(1.0f).to_bits() == 0x3c);
    assert(lp::float8_e5m2(0, 15, 0).to_bits() == lp::float8_e5m2(1.0f).to_bits());
    assert(lp::float8_e5m2(1.125f).to_bits() == 0x3c);
    assert(lp::float8_e5m2(1.375f).to_bits() == 0x3e);
    assert(lp::float4_e2m1(1.0f).to_bits() == 2);
    assert(static_cast<float>(lp::float4_e2m1::from_bits(0b0101)) == 3.0f);
    assert(std::signbit(static_cast<float>(lp::float4_e2m1::from_bits(0b1000))));
    assert(static_cast<float>(lp::float4_e2m1::from_bits(0b1111)) == -6.0f);
    assert(lp::float8_e8m0(1.0f).to_bits() == 127);
    assert(lp::float8_e8m0(lp::detail::u32_bits_to_f32(0x00600000)).to_bits() == 0);
    assert(lp::float8_e8m0(lp::detail::u32_bits_to_f32(0x00600001)).to_bits() == 1);
    assert(lp::float4_e2m1(0.25f).to_bits() == 0);
    assert(lp::float4_e2m1(0.75f).to_bits() == 2);
    assert(lp::float4_e2m1(5.0f).to_bits() == 6);
    assert(lp::float8_e5m2(std::numeric_limits<float>::infinity()).to_bits() == 0x7c);
    assert(std::isnan(static_cast<float>(lp::float8_e8m0::from_bits(255))));
    assert((lp::float16(1.0f) + 1.0f).to_bits() == 0x4000);
    auto sum = lp::float16(1.0f);
    sum += 1.0f;
    assert(sum.to_bits() == 0x4000);

    // Every encoding must decode and re-encode consistently (except NaN payloads).
    for (unsigned bits = 0; bits < 65536; ++bits) {
        const auto half = lp::float16::from_bits(static_cast<uint16_t>(bits));
        if ((bits & 0x7c00) != 0x7c00 || !(bits & 0x03ff)) {
            assert(lp::float16(static_cast<float>(half)).to_bits() == bits);
        }
    }
    for (unsigned bits = 0; bits < 256; ++bits) {
        const auto e4 = lp::float8_e4m3::from_bits(static_cast<uint8_t>(bits));
        const unsigned exponent = (bits & 0x7f) >> 3;
        const unsigned fraction = bits & 7;
        const float decoded = static_cast<float>(e4);
        if ((bits & 0x7f) != 0x7f) {
            const float expected = exponent ? std::ldexp(static_cast<float>(8 + fraction), static_cast<int>(exponent) - 10)
                                            : std::ldexp(static_cast<float>(fraction), -9);
            assert(std::abs(decoded) == expected);
            assert(std::signbit(decoded) == static_cast<bool>(bits & 0x80));
            assert(lp::float8_e4m3(decoded).to_bits() == bits);
        }
        const auto e5 = lp::float8_e5m2::from_bits(static_cast<uint8_t>(bits));
        if ((bits & 0x7c) != 0x7c) {
            assert(lp::float8_e5m2(static_cast<float>(e5)).to_bits() == bits);
        }
        const auto e8 = lp::float8_e8m0::from_bits(static_cast<uint8_t>(bits));
        if (bits != 255) {
            assert(lp::float8_e8m0(static_cast<float>(e8)).to_bits() == bits);
        }
    }
    for (unsigned bits = 0; bits < 16; ++bits) {
        const auto f4 = lp::float4_e2m1::from_bits(static_cast<uint8_t>(bits));
        assert(lp::float4_e2m1(static_cast<float>(f4)).to_bits() == bits);
    }

    const lp::element::Type type("FP16");
    assert(type.bitwidth() == 16 && type.size() == 2 && type.is_real());
    assert(lp::element::Type("BIN").to_string() == "u1");
    assert(lp::element::Type::get_known_types().size() == 25);
    lp::element::Type merged;
    assert(lp::element::Type::merge(merged, lp::element::dynamic, type));
    assert(merged.to_string() == "f16");
    std::stringstream stream;
    stream << merged;
    stream >> merged;
    assert(merged.to_string() == "f16");
}