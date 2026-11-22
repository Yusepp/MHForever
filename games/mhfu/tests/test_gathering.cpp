#include "check.hpp"
#include "gathering_logic.hpp"
#include <array>
using namespace mhf::mhfu;
int main() {
    std::array<GatheringItem, 5> bag{{{0, 0}, {0x90, 0}, {0x91, 2}, {0x94, 1}, {0x123, 4}}};
    GatheringInput in{0, 3, bag};
    auto out = decide_gathering(in);
    MHF_CHECK(out.kind == GatheringKind::mining);
    MHF_CHECK(out.tool_slot == 2);
    in.node_type = 4;
    out = decide_gathering(in);
    MHF_CHECK(out.kind == GatheringKind::bugs && out.tool_slot == 3);
    in.inventory = {};
    out = decide_gathering(in);
    MHF_CHECK(out.kind == GatheringKind::bugs && !out.tool_slot);
    in.node_type = 1;
    out = decide_gathering(in);
    MHF_CHECK(out.kind == GatheringKind::gather && !out.tool_slot);
    in.node_index = 0xffff;
    MHF_CHECK(decide_gathering(in).kind == GatheringKind::none);
    in.node_index = 0;
    in.node_index = 0x10000;
    MHF_CHECK(decide_gathering(in).kind == GatheringKind::none);
    in.node_index = 0;
    in.node_type = 3;
    in.inventory = bag;
    bag[2] = {0x4c9, 1}; // Supply Pickaxe
    MHF_CHECK(decide_gathering(in).tool_slot == 2);
    in.node_type = 4;
    bag[3] = {0x4ca, 1}; // Supply Bug Net
    MHF_CHECK(decide_gathering(in).tool_slot == 3);
    bag[3].count = 0;
    MHF_CHECK(!decide_gathering(in).tool_slot);
    in.node_type = 3;
    in.exhausted = true;
    auto empty = decide_gathering(in);
    MHF_CHECK(empty.kind == GatheringKind::mining && !empty.available);
    in.exhausted = false;
    MHF_CHECK(decide_gathering(in).available);
    in.inventory = {};
    MHF_CHECK(!decide_gathering(in).available);
    // Infinite tools: provides high-tier virtual pickaxe, bug net, and Mega Fishing Fly even with empty bag.
    in.infinite_tools = true;
    auto inf_mine = decide_gathering(in);
    MHF_CHECK(inf_mine.kind == GatheringKind::mining && inf_mine.available && inf_mine.uses_virtual_tool &&
              inf_mine.virtual_tool_id == kMegaPickaxeId);
    in.node_type = 4;
    auto inf_bug = decide_gathering(in);
    MHF_CHECK(inf_bug.kind == GatheringKind::bugs && inf_bug.available && inf_bug.uses_virtual_tool &&
              inf_bug.virtual_tool_id == kMegaBugNetId);
    in.fishing = true;
    in.node_index = 0xffff;
    auto inf_fish = decide_gathering(in);
    MHF_CHECK(inf_fish.kind == GatheringKind::fishing && inf_fish.available && inf_fish.uses_virtual_tool &&
              inf_fish.virtual_tool_id == kMegaFishingFlyId);
    // Fishing without infinite tools requires bait in inventory.
    in.infinite_tools = false;
    MHF_CHECK(!decide_gathering(in).available);
    bag[4] = {0x8f, 3};
    in.inventory = bag;
    auto real_fish = decide_gathering(in);
    MHF_CHECK(real_fish.kind == GatheringKind::fishing && real_fish.available && real_fish.tool_slot == 4 &&
              !real_fish.uses_virtual_tool);
    in.fishing = false;
    in.node_index = 0;
    in.inventory = {};
    in.carving = true;
    auto carve = decide_gathering(in);
    MHF_CHECK(carve.kind == GatheringKind::carve && carve.available && !carve.tool_slot);
    in.exhausted = true;
    MHF_CHECK(!decide_gathering(in).available);
    return mhf::test::finish("test_gathering");
}
