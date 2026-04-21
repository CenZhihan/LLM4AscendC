/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef HALF_HPP_INCLUDED
#define HALF_HPP_INCLUDED

#include <cstdint>
#include <cmath>
#include <iostream>
#include <limits>
#include <type_traits>
#include <algorithm>
#include <cstring>

namespace fp16 {

    class half {
    public:
        uint16_t data;

        // 构造函数
        constexpr half() noexcept : data(0) {}
        half(float f) noexcept { data = float_to_half(f); }
        half(double d) noexcept { data = float_to_half(static_cast<float>(d)); }
        explicit half(int i) noexcept { data = float_to_half(static_cast<float>(i)); }

        static constexpr half from_bits(uint16_t bits) noexcept {
            half h;
            h.data = bits;
            return h;
        }

        // 类型转换
        operator float() const noexcept { return half_to_float(data); }
        explicit operator double() const noexcept { return static_cast<double>(half_to_float(data)); }
        explicit operator int() const noexcept { return static_cast<int>(half_to_float(data)); }
        explicit operator bool() const noexcept { return (data & 0x7FFF) != 0; }

        // 基础算术运算符 (+=, -=, *=, /=)
        half& operator+=(const half& rhs) { return *this = half(float(*this) + float(rhs)); }
        half& operator-=(const half& rhs) { return *this = half(float(*this) - float(rhs)); }
        half& operator*=(const half& rhs) { return *this = half(float(*this) * float(rhs)); }
        half& operator/=(const half& rhs) { return *this = half(float(*this) / float(rhs)); }

        half& operator+=(float rhs) { return *this = half(float(*this) + rhs); }
        half& operator-=(float rhs) { return *this = half(float(*this) - rhs); }
        half& operator*=(float rhs) { return *this = half(float(*this) * rhs); }
        half& operator/=(float rhs) { return *this = half(float(*this) / rhs); }

        half operator+() const { return *this; }
        half operator-() const { return from_bits(data ^ 0x8000); }

        // 自增自减
        half& operator++() { return *this += half(1.0f); }
        half& operator--() { return *this -= half(1.0f); }
        half operator++(int) { half temp = *this; ++(*this); return temp; }
        half operator--(int) { half temp = *this; --(*this); return temp; }

        // 注意：比较运算符已移至类外，以支持混合类型比较 (如 h < 0)

    private:
        // Float32 -> Float16
        static uint16_t float_to_half(float f) noexcept {
            uint32_t x;
            std::memcpy(&x, &f, sizeof(float));
            const uint32_t sign = (x >> 31) & 1;
            uint32_t exponent = (x >> 23) & 0xFF;
            uint32_t mantissa = x & 0x7FFFFF;
            uint16_t h_sign = static_cast<uint16_t>(sign << 15);

            if (exponent == 255) return h_sign | 0x7C00 | (mantissa ? 0x200 : 0);
            if (exponent == 0) return h_sign;

            int new_exp = static_cast<int>(exponent) - 127 + 15;
            if (new_exp >= 31) return h_sign | 0x7C00;
            if (new_exp <= 0) {
                if (new_exp < -10) return h_sign;
                mantissa |= 0x800000;
                int shift = 14 - new_exp;
                uint32_t result_mantissa = (mantissa >> shift);
                if ((mantissa & ((1 << shift) - 1)) > (1 << (shift - 1))) result_mantissa++;
                return h_sign | static_cast<uint16_t>(result_mantissa);
            }
            uint32_t result_mantissa = (mantissa >> 13);
            if (mantissa & 0x1000) result_mantissa++;
            if (result_mantissa & 0x400) {
                result_mantissa = 0;
                new_exp++;
                if (new_exp >= 31) return h_sign | 0x7C00;
            }
            return h_sign | (static_cast<uint16_t>(new_exp) << 10) | static_cast<uint16_t>(result_mantissa);
        }

        // Float16 -> Float32
        static float half_to_float(uint16_t h) noexcept {
            uint32_t sign = (h >> 15) & 1;
            uint32_t exponent = (h >> 10) & 0x1F;
            uint32_t mantissa = h & 0x3FF;
            uint32_t f_sign = sign << 31;
            uint32_t f_exp, f_man;

            if (exponent == 0) {
                if (mantissa == 0) { f_exp = 0; f_man = 0; }
                else {
                    f_exp = 0;
                    while ((mantissa & 0x400) == 0) { mantissa <<= 1; f_exp++; }
                    mantissa &= 0x3FF;
                    f_exp = 127 - 15 - f_exp;
                    f_man = mantissa << 13;
                }
            } else if (exponent == 31) {
                f_exp = 255; f_man = mantissa ? 0x400000 : 0;
            } else {
                f_exp = exponent + 127 - 15; f_man = mantissa << 13;
            }
            uint32_t result_bits = f_sign | (f_exp << 23) | f_man;
            float result;
            std::memcpy(&result, &result_bits, sizeof(float));
            return result;
        }
    };

    // =========================================================
    // 算术运算符 (Global)
    // =========================================================
    inline half operator+(half lhs, half rhs) { return lhs += rhs; }
    inline half operator-(half lhs, half rhs) { return lhs -= rhs; }
    inline half operator*(half lhs, half rhs) { return lhs *= rhs; }
    inline half operator/(half lhs, half rhs) { return lhs /= rhs; }

    inline half operator+(half lhs, float rhs) { return lhs += rhs; }
    inline half operator-(half lhs, float rhs) { return lhs -= rhs; }
    inline half operator*(half lhs, float rhs) { return lhs *= rhs; }
    inline half operator/(half lhs, float rhs) { return lhs /= rhs; }
    inline half operator+(float lhs, half rhs) { return half(lhs) += rhs; }
    inline half operator-(float lhs, half rhs) { return half(lhs) -= rhs; }
    inline half operator*(float lhs, half rhs) { return half(lhs) *= rhs; }
    inline half operator/(float lhs, half rhs) { return half(lhs) /= rhs; }

    // =========================================================
    // 比较运算符 (使用模板解决歧义)
    // =========================================================

    // 1. half vs half (保持不变)
    inline bool operator==(half lhs, half rhs) { return float(lhs) == float(rhs); }
    inline bool operator!=(half lhs, half rhs) { return float(lhs) != float(rhs); }
    inline bool operator<(half lhs, half rhs)  { return float(lhs) < float(rhs); }
    inline bool operator>(half lhs, half rhs)  { return float(lhs) > float(rhs); }
    inline bool operator<=(half lhs, half rhs) { return float(lhs) <= float(rhs); }
    inline bool operator>=(half lhs, half rhs) { return float(lhs) >= float(rhs); }

    // 辅助模板：仅允许算术类型 (int, float, double, unsigned...) 参与比较
    // 这避免了与非数值类型比较时产生奇怪的错误
    template <typename T>
    using EnableIfArithmetic = typename std::enable_if<std::is_arithmetic<T>::value, bool>::type;

    // 2. half vs T (T 可以是 int, unsigned, float, double...)
    template <typename T>
    inline EnableIfArithmetic<T> operator==(half lhs, T rhs) { return float(lhs) == static_cast<float>(rhs); }
    template <typename T>
    inline EnableIfArithmetic<T> operator!=(half lhs, T rhs) { return float(lhs) != static_cast<float>(rhs); }
    template <typename T>
    inline EnableIfArithmetic<T> operator<(half lhs, T rhs)  { return float(lhs) < static_cast<float>(rhs); }
    template <typename T>
    inline EnableIfArithmetic<T> operator>(half lhs, T rhs)  { return float(lhs) > static_cast<float>(rhs); }
    template <typename T>
    inline EnableIfArithmetic<T> operator<=(half lhs, T rhs) { return float(lhs) <= static_cast<float>(rhs); }
    template <typename T>
    inline EnableIfArithmetic<T> operator>=(half lhs, T rhs) { return float(lhs) >= static_cast<float>(rhs); }

    // 3. T vs half
    template <typename T>
    inline EnableIfArithmetic<T> operator==(T lhs, half rhs) { return static_cast<float>(lhs) == float(rhs); }
    template <typename T>
    inline EnableIfArithmetic<T> operator!=(T lhs, half rhs) { return static_cast<float>(lhs) != float(rhs); }
    template <typename T>
    inline EnableIfArithmetic<T> operator<(T lhs, half rhs)  { return static_cast<float>(lhs) < float(rhs); }
    template <typename T>
    inline EnableIfArithmetic<T> operator>(T lhs, half rhs)  { return static_cast<float>(lhs) > float(rhs); }
    template <typename T>
    inline EnableIfArithmetic<T> operator<=(T lhs, half rhs) { return static_cast<float>(lhs) <= float(rhs); }
    template <typename T>
    inline EnableIfArithmetic<T> operator>=(T lhs, half rhs) { return static_cast<float>(lhs) >= float(rhs); }

    // =========================================================
    // IO & Literal
    // =========================================================
    inline std::ostream& operator<<(std::ostream& os, const half& h) { return os << static_cast<float>(h); }
    inline std::istream& operator>>(std::istream& is, half& h) { float f; is >> f; h = half(f); return is; }
    inline half operator"" _h(long double val) { return half(static_cast<float>(val)); }

} // namespace fp16

// ==========================================
// STL Traits 特化
// ==========================================
namespace std {
    template<> class numeric_limits<fp16::half> {
    public:
        static constexpr bool is_specialized = true;
        static constexpr bool is_signed = true;
        static constexpr bool is_integer = false;
        static constexpr bool is_exact = false;
        static constexpr bool has_infinity = true;
        static constexpr bool has_quiet_NaN = true;
        static constexpr bool has_signaling_NaN = true;
        static constexpr float_denorm_style has_denorm = denorm_present;
        static constexpr bool has_denorm_loss = false;
        static constexpr bool is_iec559 = true;
        static constexpr bool is_bounded = true;
        static constexpr bool is_modulo = false;
        static constexpr int digits = 11;
        static constexpr int digits10 = 3;
        static constexpr int max_digits10 = 5;
        static constexpr int radix = 2;
        static constexpr int min_exponent = -13;
        static constexpr int min_exponent10 = -4;
        static constexpr int max_exponent = 16;
        static constexpr int max_exponent10 = 4;
        static constexpr bool traps = false;
        static constexpr bool tinyness_before = false;
        static constexpr float_round_style round_style = round_to_nearest;

        static constexpr fp16::half min() noexcept { return fp16::half::from_bits(0x0400); }
        static constexpr fp16::half max() noexcept { return fp16::half::from_bits(0x7BFF); }
        static constexpr fp16::half lowest() noexcept { return fp16::half::from_bits(0xFBFF); }
        static constexpr fp16::half epsilon() noexcept { return fp16::half::from_bits(0x1400); }
        static constexpr fp16::half round_error() noexcept { return fp16::half::from_bits(0x1000); }
        static constexpr fp16::half infinity() noexcept { return fp16::half::from_bits(0x7C00); }
        static constexpr fp16::half quiet_NaN() noexcept { return fp16::half::from_bits(0x7FFF); }
        static constexpr fp16::half signaling_NaN() noexcept { return fp16::half::from_bits(0x7DFF); }
        static constexpr fp16::half denorm_min() noexcept { return fp16::half::from_bits(0x0001); }
    };

    template<> struct hash<fp16::half> {
        size_t operator()(const fp16::half& h) const noexcept {
            if ((h.data & 0x7FFF) == 0) return hash<uint16_t>{}(0);
            return hash<uint16_t>{}(h.data);
        }
    };

    inline fp16::half abs(fp16::half h) { return fp16::half(std::abs(float(h))); }
    inline fp16::half sqrt(fp16::half h) { return fp16::half(std::sqrt(float(h))); }
    inline fp16::half exp(fp16::half h) { return fp16::half(std::exp(float(h))); }
    inline fp16::half log(fp16::half h) { return fp16::half(std::log(float(h))); }
    inline fp16::half log10(fp16::half h) { return fp16::half(std::log10(float(h))); }
    inline fp16::half sin(fp16::half h) { return fp16::half(std::sin(float(h))); }
    inline fp16::half cos(fp16::half h) { return fp16::half(std::cos(float(h))); }
    inline fp16::half tan(fp16::half h) { return fp16::half(std::tan(float(h))); }
    inline fp16::half floor(fp16::half h) { return fp16::half(std::floor(float(h))); }
    inline fp16::half ceil(fp16::half h) { return fp16::half(std::ceil(float(h))); }
    inline bool isnan(fp16::half h) { return std::isnan(float(h)); }
    inline bool isinf(fp16::half h) { return std::isinf(float(h)); }
    inline bool isfinite(fp16::half h) { return std::isfinite(float(h)); }
}

#endif