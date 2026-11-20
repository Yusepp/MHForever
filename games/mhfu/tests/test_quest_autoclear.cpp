// Tests the quest auto-clear trigger logic (host/quest_autoclear.cpp) against synthetic guest RAM — no game data.

#include "quest_autoclear.hpp"

#include "check.hpp"

#include <cstdint>

namespace {

using mhf::mhfu::QuestAutoclear;
using mhf::mhfu::QuestAutoclearSymbols;
using mhf::mhfu::kQuestAutoclearCombo;
using mhf::mhfu::kQuestAutoclearHoldUs;
namespace buttons = mhf::psp::buttons;

constexpr std::uint32_t kQuestMgrPtrAddr = 0x08A00000u;
constexpr std::uint32_t kGameSysPtrAddr = 0x08A00004u;
constexpr std::uint32_t kQuestMgrAddr = 0x09000000u;
constexpr std::uint32_t kGameSysAddr = 0x09010000u;

constexpr std::uint32_t kQuestState = 0x00u;
constexpr std::uint32_t kQuestId = 0x10u;
constexpr std::uint32_t kQuestRequired = 0x67Cu;
constexpr std::uint32_t kQuestPostTimer = 0x710u;
constexpr std::uint32_t kQuestObj0 = 0x728u;
constexpr std::uint32_t kQuestObj1 = 0x730u;
constexpr std::uint32_t kGameSysPhase = 0x2Eu;

struct TestHarness {
    psprecomp::GuestMemory memory;
    QuestAutoclear autoclear{{kQuestMgrPtrAddr, kGameSysPtrAddr}};

    TestHarness() {
        memory.store32(kQuestMgrPtrAddr, kQuestMgrAddr);
        memory.store32(kGameSysPtrAddr, kGameSysAddr);
        setup_quest(60001u, 1u, 0x12u, 0x0001u, 1u, 0u, 0u, 0u);
    }

    void setup_quest(std::uint32_t qid, std::uint8_t required, std::uint32_t flags0, std::uint16_t target0,
                     std::uint16_t rem0, std::uint32_t flags1, std::uint16_t target1, std::uint16_t rem1) {
        memory.store8(kQuestMgrAddr + kQuestState, 0u);
        memory.store32(kQuestMgrAddr + kQuestId, qid);
        memory.store8(kQuestMgrAddr + kQuestRequired, required);
        memory.store32(kQuestMgrAddr + kQuestPostTimer, 1800u); // 60s countdown (must remain untouched)
        memory.store32(kQuestMgrAddr + kQuestObj0, flags0);
        memory.store16(kQuestMgrAddr + kQuestObj0 + 4u, target0);
        memory.store16(kQuestMgrAddr + kQuestObj0 + 6u, rem0);
        memory.store32(kQuestMgrAddr + kQuestObj1, flags1);
        memory.store16(kQuestMgrAddr + kQuestObj1 + 4u, target1);
        memory.store16(kQuestMgrAddr + kQuestObj1 + 6u, rem1);
        memory.store8(kGameSysAddr + kGameSysPhase, 2u);
    }
};

void test_single_objective_hold_and_latch() {
    TestHarness h;
    // Partial combo does not start the timer.
    MHF_CHECK(!h.autoclear.poll(h.memory, buttons::kLTrigger | buttons::kRTrigger, 0u, true));
    MHF_CHECK(!h.autoclear.poll(h.memory, buttons::kLTrigger | buttons::kRTrigger, kQuestAutoclearHoldUs + 500'000u, true));
    MHF_CHECK(h.memory.load32(kQuestMgrAddr + kQuestObj0) == 0x12u);

    // Releasing before 2s resets the timer.
    MHF_CHECK(!h.autoclear.poll(h.memory, kQuestAutoclearCombo, 10'000'000u, true));
    MHF_CHECK(!h.autoclear.poll(h.memory, kQuestAutoclearCombo, 11'999'999u, true));
    MHF_CHECK(h.memory.load32(kQuestMgrAddr + kQuestObj0) == 0x12u);
    MHF_CHECK(!h.autoclear.poll(h.memory, buttons::kLTrigger, 12'000'000u, true));
    MHF_CHECK(!h.autoclear.poll(h.memory, kQuestAutoclearCombo, 12'100'000u, true));
    MHF_CHECK(!h.autoclear.poll(h.memory, kQuestAutoclearCombo, 14'099'999u, true));

    // Continuous 2s hold triggers once (extra buttons held simultaneously still count).
    MHF_CHECK(h.autoclear.poll(h.memory, kQuestAutoclearCombo | buttons::kCross, 14'100'000u, true));
    MHF_CHECK(h.memory.load32(kQuestMgrAddr + kQuestObj0) == 0x32u);
    MHF_CHECK(h.memory.load16(kQuestMgrAddr + kQuestObj0 + 4u) == 0x0001u); // target id preserved
    MHF_CHECK(h.memory.load16(kQuestMgrAddr + kQuestObj0 + 6u) == 0u);      // remaining zeroed
    MHF_CHECK(h.memory.load32(kQuestMgrAddr + kQuestObj1) == 0u);           // unused slot untouched
    MHF_CHECK(h.memory.load32(kQuestMgrAddr + kQuestPostTimer) == 1800u);   // post-clear timer untouched

    // Latched for the remainder of the quest, even if released and held again before state flips.
    h.memory.store32(kQuestMgrAddr + kQuestObj0, 0x12u);
    MHF_CHECK(!h.autoclear.poll(h.memory, kQuestAutoclearCombo, 17'000'000u, true));
    MHF_CHECK(!h.autoclear.poll(h.memory, 0u, 17'100'000u, true));
    MHF_CHECK(!h.autoclear.poll(h.memory, kQuestAutoclearCombo, 17'200'000u, true));
    MHF_CHECK(!h.autoclear.poll(h.memory, kQuestAutoclearCombo, 20'000'000u, true));
    MHF_CHECK(h.memory.load32(kQuestMgrAddr + kQuestObj0) == 0x12u);

    // Once the quest transitions to clear (state = 1) and a new quest begins (even same qid), latch resets.
    h.memory.store8(kQuestMgrAddr + kQuestState, 1u);
    MHF_CHECK(!h.autoclear.poll(h.memory, kQuestAutoclearCombo, 21'000'000u, true));
    h.setup_quest(60001u, 1u, 0x14u, 0x0022u, 3u, 0u, 0u, 0u);
    MHF_CHECK(!h.autoclear.poll(h.memory, kQuestAutoclearCombo, 22'000'000u, true));
    MHF_CHECK(h.autoclear.poll(h.memory, kQuestAutoclearCombo, 22'000'000u + kQuestAutoclearHoldUs, true));
    MHF_CHECK(h.memory.load32(kQuestMgrAddr + kQuestObj0) == 0x34u);
    MHF_CHECK(h.memory.load16(kQuestMgrAddr + kQuestObj0 + 6u) == 0u);
}

void test_dual_objective_quest() {
    TestHarness h;
    // Objective 0 already cleared (0x32), objective 1 still active (0x12, 2 remaining).
    h.setup_quest(60205u, 2u, 0x32u, 0x0010u, 0u, 0x12u, 0x0011u, 2u);
    MHF_CHECK(!h.autoclear.poll(h.memory, kQuestAutoclearCombo, 0u, true));
    MHF_CHECK(h.autoclear.poll(h.memory, kQuestAutoclearCombo, kQuestAutoclearHoldUs, true));
    MHF_CHECK(h.memory.load32(kQuestMgrAddr + kQuestObj0) == 0x32u);
    MHF_CHECK(h.memory.load16(kQuestMgrAddr + kQuestObj0 + 6u) == 0u);
    MHF_CHECK(h.memory.load32(kQuestMgrAddr + kQuestObj1) == 0x32u);
    MHF_CHECK(h.memory.load16(kQuestMgrAddr + kQuestObj1 + 4u) == 0x0011u);
    MHF_CHECK(h.memory.load16(kQuestMgrAddr + kQuestObj1 + 6u) == 0u);
}

void test_guards() {
    // Disabled option never triggers and clears any in-progress hold.
    {
        TestHarness h;
        MHF_CHECK(!h.autoclear.poll(h.memory, kQuestAutoclearCombo, 1'000'000u, true));
        MHF_CHECK(!h.autoclear.poll(h.memory, kQuestAutoclearCombo, 2'000'000u, false));
        MHF_CHECK(!h.autoclear.poll(h.memory, kQuestAutoclearCombo, 3'500'000u, true));
        MHF_CHECK(h.memory.load32(kQuestMgrAddr + kQuestObj0) == 0x12u);
    }

    // Quest not active: state != 0, qid == 0, game_sys phase != 2, required == 0 or > 2.
    {
        TestHarness h;
        h.memory.store8(kQuestMgrAddr + kQuestState, 2u); // failed
        MHF_CHECK(!h.autoclear.poll(h.memory, kQuestAutoclearCombo, 0u, true));
        MHF_CHECK(!h.autoclear.poll(h.memory, kQuestAutoclearCombo, kQuestAutoclearHoldUs, true));
        MHF_CHECK(h.memory.load32(kQuestMgrAddr + kQuestObj0) == 0x12u);

        h.setup_quest(0u, 1u, 0x12u, 1u, 1u, 0u, 0u, 0u);
        MHF_CHECK(!h.autoclear.poll(h.memory, kQuestAutoclearCombo, 5'000'000u, true));
        MHF_CHECK(!h.autoclear.poll(h.memory, kQuestAutoclearCombo, 5'000'000u + kQuestAutoclearHoldUs, true));

        h.setup_quest(60001u, 1u, 0x12u, 1u, 1u, 0u, 0u, 0u);
        h.memory.store8(kGameSysAddr + kGameSysPhase, 3u);
        MHF_CHECK(!h.autoclear.poll(h.memory, kQuestAutoclearCombo, 10'000'000u, true));
        MHF_CHECK(!h.autoclear.poll(h.memory, kQuestAutoclearCombo, 10'000'000u + kQuestAutoclearHoldUs, true));

        h.setup_quest(60001u, 0u, 0x12u, 1u, 1u, 0u, 0u, 0u);
        MHF_CHECK(!h.autoclear.poll(h.memory, kQuestAutoclearCombo, 15'000'000u, true));
        MHF_CHECK(!h.autoclear.poll(h.memory, kQuestAutoclearCombo, 15'000'000u + kQuestAutoclearHoldUs, true));

        h.setup_quest(60001u, 3u, 0x12u, 1u, 1u, 0x12u, 2u, 1u);
        MHF_CHECK(!h.autoclear.poll(h.memory, kQuestAutoclearCombo, 20'000'000u, true));
        MHF_CHECK(!h.autoclear.poll(h.memory, kQuestAutoclearCombo, 20'000'000u + kQuestAutoclearHoldUs, true));
        MHF_CHECK(h.memory.load32(kQuestMgrAddr + kQuestObj0) == 0x12u);
    }

    // Failed or inactive objective blocks writing any slot (no partial dual-objective mutation).
    {
        TestHarness h;
        h.setup_quest(60002u, 2u, 0x12u, 1u, 1u, 0x52u, 2u, 1u); // slot 1 has 0x40 failed bit
        MHF_CHECK(!h.autoclear.poll(h.memory, kQuestAutoclearCombo, 0u, true));
        MHF_CHECK(!h.autoclear.poll(h.memory, kQuestAutoclearCombo, kQuestAutoclearHoldUs, true));
        MHF_CHECK(h.memory.load32(kQuestMgrAddr + kQuestObj0) == 0x12u);
        MHF_CHECK(h.memory.load32(kQuestMgrAddr + kQuestObj1) == 0x52u);

        h.setup_quest(60003u, 2u, 0x12u, 1u, 1u, 0x02u, 2u, 1u); // slot 1 missing 0x10 active bit
        MHF_CHECK(!h.autoclear.poll(h.memory, kQuestAutoclearCombo, 5'000'000u, true));
        MHF_CHECK(!h.autoclear.poll(h.memory, kQuestAutoclearCombo, 5'000'000u + kQuestAutoclearHoldUs, true));
        MHF_CHECK(h.memory.load32(kQuestMgrAddr + kQuestObj0) == 0x12u);
        MHF_CHECK(h.memory.load32(kQuestMgrAddr + kQuestObj1) == 0x02u);
    }

    // Missing symbols, unaligned pointers, or out-of-bounds guest pointers fail safely.
    {
        psprecomp::GuestMemory memory;
        QuestAutoclear unmapped{QuestAutoclearSymbols{}};
        MHF_CHECK(!unmapped.poll(memory, kQuestAutoclearCombo, 0u, true));
        MHF_CHECK(!unmapped.poll(memory, kQuestAutoclearCombo, kQuestAutoclearHoldUs, true));

        TestHarness h;
        h.memory.store32(kQuestMgrPtrAddr, kQuestMgrAddr + 2u); // unaligned
        MHF_CHECK(!h.autoclear.poll(h.memory, kQuestAutoclearCombo, 0u, true));
        MHF_CHECK(!h.autoclear.poll(h.memory, kQuestAutoclearCombo, kQuestAutoclearHoldUs, true));

        h.memory.store32(kQuestMgrPtrAddr, 0x0BFFFF00u); // out of guest RAM
        MHF_CHECK(!h.autoclear.poll(h.memory, kQuestAutoclearCombo, 5'000'000u, true));
        MHF_CHECK(!h.autoclear.poll(h.memory, kQuestAutoclearCombo, 5'000'000u + kQuestAutoclearHoldUs, true));

        h.memory.store32(kQuestMgrPtrAddr, kQuestMgrAddr);
        h.memory.store32(kGameSysPtrAddr, 0u);
        MHF_CHECK(!h.autoclear.poll(h.memory, kQuestAutoclearCombo, 10'000'000u, true));
        MHF_CHECK(!h.autoclear.poll(h.memory, kQuestAutoclearCombo, 10'000'000u + kQuestAutoclearHoldUs, true));
    }

    // Explicit clear() resets hold state when leaving the quest overlay.
    {
        TestHarness h;
        MHF_CHECK(!h.autoclear.poll(h.memory, kQuestAutoclearCombo, 1'000'000u, true));
        h.autoclear.clear();
        MHF_CHECK(!h.autoclear.poll(h.memory, kQuestAutoclearCombo, 3'500'000u, true));
        MHF_CHECK(h.memory.load32(kQuestMgrAddr + kQuestObj0) == 0x12u);
        MHF_CHECK(h.autoclear.poll(h.memory, kQuestAutoclearCombo, 3'500'000u + kQuestAutoclearHoldUs, true));
        MHF_CHECK(h.memory.load32(kQuestMgrAddr + kQuestObj0) == 0x32u);
    }
}

} // namespace

int main() {
    test_single_objective_hold_and_latch();
    test_dual_objective_quest();
    test_guards();
    return mhf::test::finish("mhfu_quest_autoclear");
}
