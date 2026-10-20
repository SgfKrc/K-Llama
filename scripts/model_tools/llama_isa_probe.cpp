// Reuse the fixed-input numerical probe, always with repack ON. Add explicit
// affinity and the actually loaded DLL paths; do not change production code.
#include "llama_isa_host.h"
#define main qlh_repack_probe_unused_main
#include "llama_kernel_probe.cpp"
#undef main
int main(int argc, char ** argv) {
    try {
        require(argc == 11, "isa_probe_requires_affinity_mask");
        std::string mask_text(argv[10]);
        require(!mask_text.empty() && mask_text.find_first_not_of("0123456789") == std::string::npos,
                "invalid_affinity_mask");
        std::size_t parsed = 0;
        const auto mask = std::stoull(mask_text, &parsed);
        require(mask > 0 && parsed == mask_text.size(), "invalid_affinity_mask");
        require(SetProcessAffinityMask(GetCurrentProcess(), static_cast<DWORD_PTR>(mask)) != 0,
                "set_affinity_failed");
        require(qlh_isa::affinity() == mask, "affinity_readback_failed");
        const auto opt = arguments(10, argv);
        require(opt.repack, "isa_probe_requires_repack_on");
        auto value = success_json(run(opt));
        value.resize(value.size() - 2); // closing brace and newline
        value += ",\"isa_host\":" + qlh_isa::host_json() + ",\"loaded_modules\":{";
        const wchar_t * names[] = {L"libllama.dll", L"ggml.dll", L"ggml-base.dll", L"ggml-cpu.dll",
            L"libgcc_s_seh-1.dll", L"libstdc++-6.dll", L"libwinpthread-1.dll", L"libgomp-1.dll"};
        bool first = true;
        for (const auto * name : names) {
            if (!first) value += ',';
            first = false;
            value += qlh_isa::quote(qlh_isa::utf8(name, static_cast<int>(wcslen(name)))) + ':'
                   + qlh_isa::quote(qlh_isa::module_path(name));
        }
        value += "}}\n";
        require(std::fwrite(value.data(), 1, value.size(), stdout) == value.size(), "stdout_failed");
        return std::fflush(stdout) == 0 ? 0 : 1;
    } catch (const ProbeError & error) { return failure(error.code); }
      catch (const std::exception & error) { std::fprintf(stderr, "%s\n", error.what()); return failure("isa_probe_failed"); }
}
