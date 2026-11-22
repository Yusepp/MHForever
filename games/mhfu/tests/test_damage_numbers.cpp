#include "damage_numbers.hpp"
#include "test_support.hpp"

#include <bit>
#include <cstdint>

namespace {

void test_record_render_and_expire() {
    psprecomp::GuestMemory mem;
    constexpr std::uint32_t kEnemy = 0x09050000u;
    constexpr std::uint32_t kSp = 0x09080000u;
    mem.store32(kEnemy + 0x200u, std::bit_cast<std::uint32_t>(0.0f));
    mem.store32(kEnemy + 0x204u, std::bit_cast<std::uint32_t>(0.0f));
    mem.store32(kEnemy + 0x208u, std::bit_cast<std::uint32_t>(300.0f));
    mem.store16(kSp + 62u, 95u); // heavy weakspot hit

    mhf::mhfu::DamageNumbers dmg;
    dmg.on_enemy_damage_hook(mem, kEnemy, kSp, 1'000'000u, true);
    MHF_CHECK(dmg.active_count() == 1u);

    std::vector<std::uint32_t> rgba(480u * 272u, 0u);
    mhf::mhfu::CameraProjection cam{{0.0f, 0.0f, 0.0f}, 0u, 480.0f / 272.0f, true};
    MHF_CHECK(dmg.render(rgba, cam, 1'100'000u));

    bool any_non_zero = false;
    for (std::uint32_t px : rgba) {
        if (px != 0u) {
            any_non_zero = true;
            break;
        }
    }
    MHF_CHECK(any_non_zero);

    // After lifetime expires, popup is pruned.
    MHF_CHECK(!dmg.render(rgba, cam, 2'500'000u));
    MHF_CHECK(dmg.active_count() == 0u);
}

} // namespace

int main() {
    test_record_render_and_expire();
    return 0;
}
