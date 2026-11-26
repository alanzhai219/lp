#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "bfloat16.hpp"
#include "float16.hpp"
#include "float4_e2m1.hpp"
#include "float8_e4m3.hpp"
#include "float8_e5m2.hpp"
#include "float8_e8m0.hpp"

namespace lp {
namespace element {
enum class Type_t {
    dynamic,
    boolean,
    bf16,
    f16,
    f32,
    f64,
    i4,
    i8,
    i16,
    i32,
    i64,
    u1,
    u2,
    u3,
    u4,
    u6,
    u8,
    u16,
    u32,
    u64,
    nf4,
    f8e4m3,
    f8e5m2,
    string,
    f4e2m1,
    f8e8m0
};

namespace detail {
struct Info {
    std::size_t bits;
    bool real;
    bool signed_value;
    bool quantized;
    const char* c_name;
    const char* name;
};

inline constexpr std::array<Info, 26> info{{
    {0, false, false, false, "dynamic", "dynamic"},
    {8, false, true, false, "char", "boolean"},
    {16, true, true, false, "bfloat16", "bf16"},
    {16, true, true, false, "float16", "f16"},
    {32, true, true, false, "float", "f32"},
    {64, true, true, false, "double", "f64"},
    {4, false, true, true, "int4_t", "i4"},
    {8, false, true, true, "int8_t", "i8"},
    {16, false, true, false, "int16_t", "i16"},
    {32, false, true, true, "int32_t", "i32"},
    {64, false, true, false, "int64_t", "i64"},
    {1, false, false, false, "uint1_t", "u1"},
    {2, false, false, false, "uint2_t", "u2"},
    {3, false, false, false, "uint3_t", "u3"},
    {4, false, false, false, "uint4_t", "u4"},
    {6, false, false, false, "uint6_t", "u6"},
    {8, false, false, true, "uint8_t", "u8"},
    {16, false, false, false, "uint16_t", "u16"},
    {32, false, false, false, "uint32_t", "u32"},
    {64, false, false, false, "uint64_t", "u64"},
    {4, false, false, true, "nfloat4", "nf4"},
    {8, true, true, true, "f8e4m3", "f8e4m3"},
    {8, true, true, true, "f8e5m2", "f8e5m2"},
    {8 * sizeof(std::string), false, false, false, "string", "string"},
    {4, true, true, true, "f4e2m1", "f4e2m1"},
    {8, true, true, true, "f8e8m0", "f8e8m0"}
}};

inline const Info& get(Type_t type) {
    const auto index = static_cast<std::size_t>(type);
    if (index >= info.size()) throw std::out_of_range("Unsupported element type");
    return info[index];
}
}  // namespace detail

class Type {
public:
    constexpr Type() = default;

    constexpr Type(Type_t type) : type_(type) {}

    explicit Type(const std::string& name) {
        if (name == "UNSPECIFIED" || name == "undefined") {
            type_ = Type_t::dynamic;
            return;
        }
        if (name == "bin" || name == "BIN") {
            type_ = Type_t::u1;
            return;
        }

        for (std::size_t i = 0; i < detail::info.size(); ++i) {
            const auto candidate = detail::info[i].name;
            if (name == candidate || (name == "BOOL" && i == 1) ||
                (name == "FP16" && i == 3) || (name == "FP32" && i == 4) ||
                (name == "FP64" && i == 5)) {
                type_ = static_cast<Type_t>(i);
                return;
            }

            std::string upper(candidate);
            for (auto& ch : upper) {
                if (ch >= 'a' && ch <= 'z') {
                    ch = static_cast<char>(ch - 'a' + 'A');
                }
            }
            if (name == upper) {
                type_ = static_cast<Type_t>(i);
                return;
            }
        }
        throw std::invalid_argument("Unsupported element type: " + name);
    }

    constexpr operator Type_t() const noexcept {
        return type_;
    }

    std::string c_type_string() const {
        return detail::get(type_).c_name;
    }

    std::size_t bitwidth() const {
        return detail::get(type_).bits;
    }

    std::size_t size() const {
        return (bitwidth() + 7) >> 3;
    }

    std::size_t hash() const noexcept {
        return static_cast<std::size_t>(type_);
    }

    bool is_static() const noexcept {
        return type_ != Type_t::dynamic;
    }

    bool is_dynamic() const noexcept {
        return !is_static();
    }

    bool is_real() const {
        return detail::get(type_).real;
    }

    bool is_integral() const {
        return !is_real();
    }

    bool is_integral_number() const {
        return is_integral() && type_ != Type_t::boolean;
    }

    bool is_signed() const {
        return detail::get(type_).signed_value;
    }

    bool is_quantized() const {
        return detail::get(type_).quantized;
    }

    std::string to_string() const {
        return detail::get(type_).name;
    }

    std::string get_type_name() const {
        return to_string();
    }

    bool compatible(const Type& other) const {
        return is_dynamic() || other.is_dynamic() || type_ == other.type_;
    }

    static bool merge(Type& dst, const Type& lhs, const Type& rhs) {
        if (lhs.is_dynamic()) {
            dst = rhs;
            return true;
        }
        if (rhs.is_dynamic() || lhs.type_ == rhs.type_) {
            dst = lhs;
            return true;
        }
        return false;
    }

    static std::vector<const Type*> get_known_types() {
        static const auto types = [] {
            std::array<Type, detail::info.size() - 1> values{};
            for (std::size_t i = 0; i < values.size(); ++i) {
                values[i] = Type(static_cast<Type_t>(i + 1));
            }
            return values;
        }();
        std::vector<const Type*> result;
        result.reserve(types.size());
        for (const auto& type : types) {
            result.push_back(&type);
        }
        return result;
    }

private:
    Type_t type_{Type_t::dynamic};
};

using TypeVector = std::vector<Type>;
inline constexpr Type dynamic(Type_t::dynamic);
inline constexpr Type boolean(Type_t::boolean);
inline constexpr Type bf16(Type_t::bf16);
inline constexpr Type f16(Type_t::f16);
inline constexpr Type f32(Type_t::f32);
inline constexpr Type f64(Type_t::f64);
inline constexpr Type i4(Type_t::i4);
inline constexpr Type i8(Type_t::i8);
inline constexpr Type i16(Type_t::i16);
inline constexpr Type i32(Type_t::i32);
inline constexpr Type i64(Type_t::i64);
inline constexpr Type u1(Type_t::u1);
inline constexpr Type u2(Type_t::u2);
inline constexpr Type u3(Type_t::u3);
inline constexpr Type u4(Type_t::u4);
inline constexpr Type u6(Type_t::u6);
inline constexpr Type u8(Type_t::u8);
inline constexpr Type u16(Type_t::u16);
inline constexpr Type u32(Type_t::u32);
inline constexpr Type u64(Type_t::u64);
inline constexpr Type nf4(Type_t::nf4);
inline constexpr Type f8e4m3(Type_t::f8e4m3);
inline constexpr Type f8e5m2(Type_t::f8e5m2);
inline constexpr Type string(Type_t::string);
inline constexpr Type f4e2m1(Type_t::f4e2m1);
inline constexpr Type f8e8m0(Type_t::f8e8m0);

template <class T>
constexpr Type from() {
    if constexpr (std::is_same_v<T, char> || std::is_same_v<T, bool>) {
        return boolean;
    } else if constexpr (std::is_same_v<T, lp::bfloat16>) {
        return bf16;
    } else if constexpr (std::is_same_v<T, lp::float16>) {
        return f16;
    } else if constexpr (std::is_same_v<T, lp::float8_e4m3>) {
        return f8e4m3;
    } else if constexpr (std::is_same_v<T, lp::float8_e5m2>) {
        return f8e5m2;
    } else if constexpr (std::is_same_v<T, lp::float4_e2m1>) {
        return f4e2m1;
    } else if constexpr (std::is_same_v<T, lp::float8_e8m0>) {
        return f8e8m0;
    } else if constexpr (std::is_same_v<T, float>) {
        return f32;
    } else if constexpr (std::is_same_v<T, double>) {
        return f64;
    } else if constexpr (std::is_same_v<T, std::int8_t>) {
        return i8;
    } else if constexpr (std::is_same_v<T, std::int16_t>) {
        return i16;
    } else if constexpr (std::is_same_v<T, std::int32_t>) {
        return i32;
    } else if constexpr (std::is_same_v<T, std::int64_t>) {
        return i64;
    } else if constexpr (std::is_same_v<T, std::uint8_t>) {
        return u8;
    } else if constexpr (std::is_same_v<T, std::uint16_t>) {
        return u16;
    } else if constexpr (std::is_same_v<T, std::uint32_t>) {
        return u32;
    } else if constexpr (std::is_same_v<T, std::uint64_t>) {
        return u64;
    } else if constexpr (std::is_same_v<T, std::string>) {
        return string;
    } else {
        static_assert(!std::is_same_v<T, T>, "Unknown element type");
    }
}

inline std::ostream& operator<<(std::ostream& out, const Type& type) {
    return out << type.to_string();
}

inline std::istream& operator>>(std::istream& in, Type& type) {
    std::string name;
    if (in >> name) {
        type = Type(name);
    }
    return in;
}
}  // namespace element
}  // namespace lp