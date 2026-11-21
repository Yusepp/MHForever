#pragma once

#include "mhf/psp/abi.hpp"

namespace mhf::psp {

// True skips the callee of a patched direct call after the delay slot has run.
using CallFilter = bool (*)(Runtime &, Context &);

namespace abi {
[[nodiscard]] CallFilter find_call_filter(Runtime &rt, const char *name);
}

} // namespace mhf::psp
