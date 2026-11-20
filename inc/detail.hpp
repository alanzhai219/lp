// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "float_layout.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <ostream>

namespace lp {
namespace detail {
inline uint32_t f32_to_u32_bits(float value) noexcept {
    uint32_t bit_pattern;
    static_assert(sizeof(bit_pattern) == sizeof(value), "float must be 32 bits");
    std::memcpy(&bit_pattern, &value, sizeof(bit_pattern));
    return bit_pattern;
}

inline float u32_bits_to_f32(uint32_t bit_pattern) noexcept {
    float value;
    std::memcpy(&value, &bit_pattern, sizeof(value));
    return value;
}

// Reuse the arithmetic operators without putting a multi-line macro in each type.
template <class Type>
class arithmetic {
public:
    template <class T>
    bool operator==(const T& value) const {
        return static_cast<float>(self()) == static_cast<float>(value);
    }

    template <class T>
    bool operator!=(const T& value) const {
        return !(*this == value);
    }

    template <class T>
    bool operator<(const T& value) const {
        return static_cast<float>(self()) < static_cast<float>(value);
    }

    template <class T>
    bool operator<=(const T& value) const {
        return static_cast<float>(self()) <= static_cast<float>(value);
    }

    template <class T>
    bool operator>(const T& value) const {
        return static_cast<float>(self()) > static_cast<float>(value);
    }

    template <class T>
    bool operator>=(const T& value) const {
        return static_cast<float>(self()) >= static_cast<float>(value);
    }

    template <class T>
    Type operator+(const T& value) const {
        return Type(static_cast<float>(self()) + static_cast<float>(value));
    }

    template <class T>
    Type operator-(const T& value) const {
        return Type(static_cast<float>(self()) - static_cast<float>(value));
    }

    template <class T>
    Type operator*(const T& value) const {
        return Type(static_cast<float>(self()) * static_cast<float>(value));
    }

    template <class T>
    Type operator/(const T& value) const {
        return Type(static_cast<float>(self()) / static_cast<float>(value));
    }

    template <class T>
    Type operator+=(const T& value) {
        return self() = self() + value;
    }

    template <class T>
    Type operator-=(const T& value) {
        return self() = self() - value;
    }

    template <class T>
    Type operator*=(const T& value) {
        return self() = self() * value;
    }

    template <class T>
    Type operator/=(const T& value) {
        return self() = self() / value;
    }

    friend std::ostream& operator<<(std::ostream& out, const Type& value) {
        return out << static_cast<float>(value);
    }

private:
    Type& self() {
        return static_cast<Type&>(*this);
    }

    const Type& self() const {
        return static_cast<const Type&>(*this);
    }
};

// Shared numeric_limits constants. Each public type declares its own specialization.
template <class Type,
          unsigned Min,
          unsigned Max,
          unsigned Lowest,
          unsigned Epsilon,
          unsigned Round,
          unsigned Inf,
          unsigned NaN,
          unsigned Denorm>
struct limits {
    static constexpr bool is_specialized = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr bool is_iec559 = false;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = false;
    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false;
    static constexpr bool has_denorm_loss = false;
    static constexpr int radix = 2;
    static constexpr std::float_round_style round_style = std::round_to_nearest;

    static constexpr Type min() noexcept {
        return Type::from_bits(Min);
    }

    static constexpr Type max() noexcept {
        return Type::from_bits(Max);
    }

    static constexpr Type lowest() noexcept {
        return Type::from_bits(Lowest);
    }

    static constexpr Type epsilon() noexcept {
        return Type::from_bits(Epsilon);
    }

    static constexpr Type round_error() noexcept {
        return Type::from_bits(Round);
    }

    static constexpr Type infinity() noexcept {
        return Type::from_bits(Inf);
    }

    static constexpr Type quiet_NaN() noexcept {
        return Type::from_bits(NaN);
    }

    static constexpr Type signaling_NaN() noexcept {
        return Type::from_bits(NaN);
    }

    static constexpr Type denorm_min() noexcept {
        return Type::from_bits(Denorm);
    }
};
}  // namespace detail
}  // namespace lp
