// No model or candidate DLL is loaded by the host preflight.
#include "llama_isa_host.h"
#include <iostream>
int main() {
    try { std::cout << qlh_isa::host_json() << '\n'; return 0; }
    catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
