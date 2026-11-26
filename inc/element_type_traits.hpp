#pragma once

#include "element_type.hpp"

namespace lp {
template <element::Type_t>
struct element_type_traits {};

template <element::Type_t Type>
using fundamental_type_for = typename element_type_traits<Type>::value_type;

#define LP_TYPE_TRAIT(Element, Cpp) \
    template <> \
    struct element_type_traits<element::Type_t::Element> { \
        using value_type = Cpp; \
    };
LP_TYPE_TRAIT(boolean, char)
LP_TYPE_TRAIT(bf16, bfloat16)
LP_TYPE_TRAIT(f16, float16)
LP_TYPE_TRAIT(f32, float)
LP_TYPE_TRAIT(f64, double)
LP_TYPE_TRAIT(i4, std::int8_t)
LP_TYPE_TRAIT(i8, std::int8_t)
LP_TYPE_TRAIT(i16, std::int16_t)
LP_TYPE_TRAIT(i32, std::int32_t)
LP_TYPE_TRAIT(i64, std::int64_t)
LP_TYPE_TRAIT(u1, std::int8_t)
LP_TYPE_TRAIT(u2, std::int8_t)
LP_TYPE_TRAIT(u3, std::int8_t)
LP_TYPE_TRAIT(u4, std::int8_t)
LP_TYPE_TRAIT(u6, std::int8_t)
LP_TYPE_TRAIT(u8, std::uint8_t)
LP_TYPE_TRAIT(u16, std::uint16_t)
LP_TYPE_TRAIT(u32, std::uint32_t)
LP_TYPE_TRAIT(u64, std::uint64_t)
LP_TYPE_TRAIT(nf4, std::int8_t)
LP_TYPE_TRAIT(f8e4m3, float8_e4m3)
LP_TYPE_TRAIT(f8e5m2, float8_e5m2)
LP_TYPE_TRAIT(string, std::string)
LP_TYPE_TRAIT(f4e2m1, float4_e2m1)
LP_TYPE_TRAIT(f8e8m0, float8_e8m0)
#undef LP_TYPE_TRAIT
}  // namespace lp