// Windows x64 host checks compiled with the baseline x86-64 ISA.
#pragma once
#define NOMINMAX
#include <windows.h>
#include <cpuid.h>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>

namespace qlh_isa {
inline std::string quote(const std::string & text) {
    std::string out = "\"";
    constexpr const char * hex = "0123456789abcdef";
    for (unsigned char ch : text) {
        if (ch == '"' || ch == '\\') { out += '\\'; out += ch; }
        else if (ch < 32) { out += "\\u00"; out += hex[ch >> 4]; out += hex[ch & 15]; }
        else { out += ch; }
    }
    return out + '"';
}
inline std::string utf8(const wchar_t * text, int length) {
    int size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, text, length, nullptr, 0, nullptr, nullptr);
    if (size <= 0) throw std::runtime_error("utf8_conversion_failed");
    std::string value(size, '\0');
    if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, text, length, value.data(), size, nullptr, nullptr) != size)
        throw std::runtime_error("utf8_conversion_failed");
    return value;
}
inline std::string module_path(const wchar_t * name) {
    HMODULE module = GetModuleHandleW(name);
    if (!module) throw std::runtime_error("expected_native_module_not_loaded");
    wchar_t path[32768];
    DWORD size = GetModuleFileNameW(module, path, 32768);
    if (!size || size >= 32768) throw std::runtime_error("native_module_path_failed");
    return utf8(path, static_cast<int>(size));
}
inline DWORD_PTR affinity() {
    DWORD_PTR process = 0, system = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &process, &system))
        throw std::runtime_error("get_affinity_failed");
    return process;
}
inline std::string host_json() {
    unsigned int a=0,b=0,c=0,d=0;
    const unsigned int maximum = __get_cpuid_max(0, nullptr);
    unsigned int basic_c = 0, ext_b = 0, sub_a = 0, leaf7_max = 0;
    if (maximum >= 1) { __cpuid_count(1,0,a,b,c,d); basic_c = c; }
    if (maximum >= 7) {
        __cpuid_count(7,0,a,b,c,d); ext_b = b; leaf7_max = a;
        if (a >= 1) { __cpuid_count(7,1,a,b,c,d); sub_a = a; }
    }
    const bool osxsave = (basic_c & (1u << 27)) != 0;
    unsigned int xlo = 0, xhi = 0;
    if (osxsave) __asm__ volatile("xgetbv" : "=a"(xlo), "=d"(xhi) : "c"(0));
    const bool avx_os = osxsave && (xlo & 6u) == 6u;
    char brand[49] = {};
    if (__get_cpuid_max(0x80000000u, nullptr) >= 0x80000004u) {
        for (unsigned int i = 0; i < 3; ++i) {
            __cpuid_count(0x80000002u + i, 0, a,b,c,d);
            unsigned int words[4] = {a,b,c,d};
            for (int j=0; j<16; ++j) brand[i*16+j] = reinterpret_cast<char *>(words)[j];
        }
    }
    DWORD_PTR process=0, system=0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &process, &system))
        throw std::runtime_error("get_affinity_failed");
    std::ostringstream out;
    out << std::boolalpha << "{\"schema\":\"qlh.isa_host.v1\",\"ok\":true,\"brand\":" << quote(brand)
        << ",\"process_affinity_mask\":" << quote(std::to_string(process))
        << ",\"system_affinity_mask\":" << quote(std::to_string(system))
        << ",\"cpuid_leaf1_ecx\":" << basic_c << ",\"cpuid_leaf7_ebx\":" << ext_b
        << ",\"cpuid_leaf7_max_subleaf\":" << leaf7_max << ",\"cpuid_leaf7_1_eax\":" << sub_a
        << ",\"xcr0_low\":" << xlo << ",\"xcr0_high\":" << xhi
        << ",\"os_avx_state\":" << avx_os << ",\"features\":{"
        << "\"sse42\":" << bool(basic_c & (1u<<20))
        << ",\"avx\":" << (avx_os && bool(basic_c & (1u<<28)))
        << ",\"avx2\":" << (avx_os && bool(ext_b & (1u<<5)))
        << ",\"bmi2\":" << bool(ext_b & (1u<<8))
        << ",\"fma\":" << (avx_os && bool(basic_c & (1u<<12)))
        << ",\"f16c\":" << (avx_os && bool(basic_c & (1u<<29)))
        << ",\"avx_vnni\":" << (avx_os && bool(sub_a & (1u<<4))) << "}}";
    return out.str();
}
}
