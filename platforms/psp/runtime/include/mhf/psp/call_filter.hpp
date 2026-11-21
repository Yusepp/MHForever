#pragma once

#include "mhf/psp/abi.hpp"

namespace mhf::psp {

// True skips a patched callee or, for a `filter` instruction patch, that instruction.
namespace abi {
[[nodiscard]] CallFilter find_call_filter(Runtime &rt, const char *name);
}

} // namespace mhf::psp
