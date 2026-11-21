#include "mhf/psp/runtime.hpp"

#include "check.hpp"

int main() {
    using namespace mhf::psp;
    constexpr std::uint32_t entry = 0x08804000, target = entry + 0x20, ret = entry + 0x100;
    const CallPatchEntry patch{entry + 4, "test_draw"};
    const Program program{"test", entry, 0, {}, {}, entry, target + 8, -1, {&patch, 1}};
    Runtime runtime;
    runtime.add_program(program);
    auto &mem = runtime.memory();
    mem.store32(entry, 0x03E04021u); // move t0, ra
    mem.store32(entry + 4, 0x0C000000u | (target >> 2u)); // jal target
    mem.store32(entry + 8, 0x24040007u); // addiu a0, zero, 7 (delay slot)
    mem.store32(entry + 12, 0x01000008u); // jr t0
    mem.store32(entry + 16, 0u); // nop
    mem.store32(target, 0x03E00008u); // jr ra
    mem.store32(target + 4, 0x24020009u); // addiu v0, zero, 9

    const auto run = [&](bool filter_enabled) {
        Context ctx{};
        ctx.set_gpr(31, ret);
        ctx.set_gpr(5, filter_enabled ? 1u : 0u);
        abi::interpret(runtime, ctx, entry);
        MHF_CHECK(ctx.gpr[4] == 7 && ctx.gpr[31] == entry + 12);
        return ctx.gpr[2];
    };
    MHF_CHECK(run(true) == 9);
    runtime.register_call_filter("test_draw", [](Runtime &, Context &ctx) {
        return ctx.gpr[4] == 7 && ctx.gpr[5] != 0;
    });
    MHF_CHECK(run(false) == 9);
    MHF_CHECK(run(true) == 0);
    return mhf::test::finish("test_call_filter");
}
