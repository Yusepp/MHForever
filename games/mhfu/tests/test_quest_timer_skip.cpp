#include "quest_timer_skip.hpp"
#include "check.hpp"

#include <cstdint>

namespace {

constexpr std::uint32_t kQuestManagerPtrAddr = 0x08A5DD4Cu;
constexpr std::uint32_t kGameSysPtrAddr = 0x089C7508u;
constexpr std::uint32_t kQuestManagerAddr = 0x09010000u;
constexpr std::uint32_t kGameSysAddr = 0x09020000u;

void setup_post_quest(psprecomp::GuestMemory &mem, std::int8_t state, std::uint8_t phase, std::uint32_t countdown) {
    mem.store32(kQuestManagerPtrAddr, kQuestManagerAddr);
    mem.store32(kGameSysPtrAddr, kGameSysAddr);
    mem.store8(kQuestManagerAddr + 0x00u, static_cast<std::uint8_t>(state));
    mem.store32(kQuestManagerAddr + 0x710u, countdown);
    mem.store8(kGameSysAddr + 0x2Eu, phase);
}

void test_skips_after_hold() {
    psprecomp::GuestMemory mem;
    setup_post_quest(mem, 1, 3, 1800u);

    mhf::mhfu::QuestTimerSkip skip({kQuestManagerPtrAddr, kGameSysPtrAddr});
    MHF_CHECK(!skip.poll(mem, 0u, 1'000'000u, true));
    MHF_CHECK(skip.can_skip());
    MHF_CHECK(skip.label(1) == "End Quest Immediately");
    MHF_CHECK(skip.label(3) == "Finalizar misión ahora");

    MHF_CHECK(!skip.poll(mem, mhf::mhfu::kQuestTimerSkipButton, 1'100'000u, true));
    MHF_CHECK(skip.label(1) == "Returning...");
    MHF_CHECK(!skip.poll(mem, mhf::mhfu::kQuestTimerSkipButton, 1'700'000u, true));
    MHF_CHECK(skip.hold_progress() > 0.5f);
    MHF_CHECK(mem.load32(kQuestManagerAddr + 0x710u) == 1800u);

    MHF_CHECK(skip.poll(mem, mhf::mhfu::kQuestTimerSkipButton, 1'950'000u, true));
    MHF_CHECK(mem.load32(kQuestManagerAddr + 0x710u) == mhf::mhfu::kQuestTimerSkipTargetFrames);
    MHF_CHECK(!skip.can_skip());
}

void test_ignores_active_quest_and_disabled() {
    psprecomp::GuestMemory mem;
    setup_post_quest(mem, 0, 2, 1800u);

    mhf::mhfu::QuestTimerSkip skip({kQuestManagerPtrAddr, kGameSysPtrAddr});
    MHF_CHECK(!skip.poll(mem, mhf::mhfu::kQuestTimerSkipButton, 1'000'000u, true));
    MHF_CHECK(!skip.can_skip());

    setup_post_quest(mem, 1, 3, 1800u);
    MHF_CHECK(!skip.poll(mem, mhf::mhfu::kQuestTimerSkipButton, 2'000'000u, false));
    MHF_CHECK(!skip.can_skip());
    MHF_CHECK(mem.load32(kQuestManagerAddr + 0x710u) == 1800u);
}

void test_ignores_initial_3s_fanfare() {
    psprecomp::GuestMemory mem;
    // Initial 3-second (90-frame) Quest Clear fanfare before the real 60s/20s return countdown begins.
    setup_post_quest(mem, 1, 3, 90u);
    mhf::mhfu::QuestTimerSkip skip({kQuestManagerPtrAddr, kGameSysPtrAddr});
    MHF_CHECK(!skip.poll(mem, 0u, 1'000'000u, true));
    MHF_CHECK(!skip.can_skip());

    // Once the real 60s (1800-frame) return countdown begins, the widget activates and stays visible down to 3s.
    setup_post_quest(mem, 1, 3, 1800u);
    MHF_CHECK(!skip.poll(mem, 0u, 2'000'000u, true));
    MHF_CHECK(skip.can_skip());
    MHF_CHECK(skip.remaining_seconds() == 60u);

    setup_post_quest(mem, 1, 3, 90u);
    MHF_CHECK(!skip.poll(mem, 0u, 59'000'000u, true));
    MHF_CHECK(skip.can_skip());
    MHF_CHECK(skip.remaining_seconds() == 3u);
}

} // namespace

int main() {
    test_skips_after_hold();
    test_ignores_active_quest_and_disabled();
    test_ignores_initial_3s_fanfare();
    return 0;
}
