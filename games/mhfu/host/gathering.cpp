#include "gathering.hpp"
#include "gathering_logic.hpp"
#include "mhf/psp/dialog.hpp"
#include "mhf/psp/host.hpp"
#include "mhfu_region_data.hpp"
#include <array>
#include <bit>
#include <cmath>
#include <string_view>

namespace mhf::mhfu {
namespace region = psp::generated::mhfu_region_data;
namespace {

constexpr std::uint32_t kGameSysPlayerSlot = 0x28, kGameSysMenu = 0x10;
constexpr std::uint32_t kPlayerManagerPlayers = 0x0C, kPlayerSlots = 6;
constexpr std::uint32_t kCockpitMenu = 0x604, kCockpitWindow = 0x52F;
constexpr std::uint32_t kPlayerPosition = 0x200, kPlayerArea = 0x29A, kPlayerReady = 0x325;
constexpr std::uint32_t kPlayerAction = 0x299, kPlayerTransportFlags = 0x614, kPlayerSelectedItem = 0x648;
constexpr std::uint32_t kQuestObjectives = 0x728, kObjectiveStride = 8, kMaxObjectives = 2;
constexpr std::uint32_t kObjectiveDelivery = 0x02, kObjectiveActive = 0x10, kObjectiveCleared = 0x20;
constexpr std::uint32_t kNodeId = 16, kNodeRemaining = 18, kNodeType = 20;
constexpr std::uint32_t kDynamicNodes = 20, kDynamicNodeSize = 32, kStaticNodeSize = 24;
constexpr std::uint32_t kInventorySlots = 25, kCircle = 0x2000;
constexpr std::uint8_t kGatherAction = 0x47, kCarveAction = 0x4a, kFishingAction = 0x4f, kMiningAction = 0x51, kBugAction = 0x52;
constexpr std::uint16_t kFishingSpotKind = 23, kSupplyBoxKind = 24, kDeliveryBoxKind = 33;
constexpr std::uint64_t kHintLifetimeUs = 150'000;

struct HintText {
    const char *mine, *bugs, *fish, *gather, *carve, *supply_box, *delivery_box, *depleted, *no_tool, *no_deliverable;
};
constexpr std::array<HintText, 6> kHints{{
    {"採掘", "虫捕り", "釣り", "採集", "剥ぎ取り", "支給品BOX", "納品BOX", "採集済み", "道具なし", "納品物なし"},
    {"Mine", "Catch bugs", "Fish", "Gather", "Carve", "Supply Box", "Delivery Box", "Depleted", "No tool", "Nothing to deliver"},
    {"Miner", "Attraper des insectes", "Pêcher", "Ramasser", "Dépecer", "Boîte de fournitures", "Boîte de livraison", "Épuisé", "Sans outil", "Rien à livrer"},
    {"Minar", "Atrapar insectos", "Pescar", "Recoger", "Tallar", "Caja de suministros", "Caja de entregas", "Agotado", "Sin herramienta", "Nada que entregar"},
    {"Abbauen", "Käfer fangen", "Angeln", "Sammeln", "Ausweiden", "Vorratsbox", "Lieferbox", "Erschöpft", "Kein Werkzeug", "Nichts abzugeben"},
    {"Estrai", "Cattura insetti", "Pesca", "Raccogli", "Scuoia", "Cassa scorte", "Cassa consegne", "Esaurito", "Nessun attrezzo", "Nulla da consegnare"},
}};

struct Target {
    std::uint32_t index;
    std::uint16_t type, remaining;
    bool carving;
    bool fishing;
    bool supply_box;
    bool delivery_box;
};

bool has_deliverable_item(const psprecomp::GuestMemory &m, std::uint32_t player,
                          std::span<const GatheringItem> items) noexcept {
    // Carrying a transport item (egg, powderstone, etc.) can always be delivered at the Red Box.
    if ((m.load32(player + kPlayerTransportFlags) & 1u) != 0u) return true;
    if constexpr (region::mh_quest_manager) {
        const std::uint32_t qm = m.load32(*region::mh_quest_manager);
        if (qm != 0u && (qm & 3u) == 0u && m.contains(qm, kQuestObjectives + kMaxObjectives * kObjectiveStride)) {
            for (std::uint32_t i = 0; i < kMaxObjectives; ++i) {
                const std::uint32_t slot = qm + kQuestObjectives + i * kObjectiveStride;
                const std::uint32_t flags = m.load32(slot);
                const std::uint16_t target_id = m.load16(slot + 4u);
                const std::uint16_t remaining = m.load16(slot + 6u);
                if ((flags & kObjectiveActive) != 0u && (flags & kObjectiveCleared) == 0u &&
                    (flags & kObjectiveDelivery) != 0u && remaining > 0u && target_id != 0u) {
                    for (const auto &item : items) {
                        if (item.id == target_id && item.count > 0u) return true;
                    }
                }
            }
        }
    }
    return false;
}

[[maybe_unused]] std::uint16_t pick_virtual_slot(std::span<const GatheringItem> items) noexcept {
    for (std::size_t i = items.size(); i-- > 0;) {
        if (items[i].id == 0 || items[i].count == 0) return static_cast<std::uint16_t>(i);
    }
    return static_cast<std::uint16_t>(items.size() - 1u);
}

// Read-only: gathering checks and inventory access must preserve the interrupted guest context.
std::optional<Target> read_target(psp::Runtime &rt, const psp::Context &ctx, std::uint32_t player,
                                  std::uint32_t scratch) {
    auto &m = rt.memory();
    auto query = ctx;
    query.gpr[4] = player;
    query.gpr[5] = scratch + 12;
    query.gpr[6] = scratch;
    m.store16(scratch + 12, std::uint16_t{0xffff});
    rt.call(query, *region::mh_gathering_probe);
    auto index = query.gpr[2] & 0xffffu;
    auto type = m.load16(scratch + 12);
    std::uint16_t remaining = 0;
    bool dynamic = false;
    if (index == 0xffffu) {
        // The stock carving query scans these dynamic corpse/tail targets.
        // Read only, with bounds/finite checks; never invoke reward/consumption
        // routines.
        const auto targets = m.load32(*region::mh_gathering_dynamic_nodes);
        if (!m.contains(targets, 4 + kDynamicNodes * kDynamicNodeSize)) return {};
        const auto number = [](std::uint32_t bits) { return std::bit_cast<float>(bits); };
        const float px = number(m.load32(player + kPlayerPosition));
        const float py = number(m.load32(player + kPlayerPosition + 4));
        const float pz = number(m.load32(player + kPlayerPosition + 8));
        if (!std::isfinite(px) || !std::isfinite(py) || !std::isfinite(pz)) return {};
        for (unsigned i = 0; i < kDynamicNodes; ++i) {
            const auto node = targets + 4 + i * kDynamicNodeSize;
            if (m.load16(node + kNodeId) == 0xffff || m.load16(node + 24) != m.load16(player + kPlayerArea)) continue;
            const float x = number(m.load32(node)), y = number(m.load32(node + 4));
            const float z = number(m.load32(node + 8)), radius = number(m.load32(node + 12));
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) || !std::isfinite(radius) || radius < 0 ||
                py < y - 200 || py >= y + 100 || (px - x) * (px - x) + (pz - z) * (pz - z) > radius * radius)
                continue;
            index = m.load16(node + kNodeId);
            type = m.load16(node + kNodeType);
            remaining = m.load16(node + kNodeRemaining);
            dynamic = true;
            break;
        }
        if (!dynamic) {
            if constexpr (region::mh_fishing_probe) {
                auto fish_call = ctx;
                fish_call.gpr[4] = player;
                fish_call.gpr[5] = scratch;
                fish_call.gpr[6] = scratch + 16;
                fish_call.gpr[7] = scratch + 20;
                rt.call(fish_call, *region::mh_fishing_probe);
                const std::uint16_t stage_kind = static_cast<std::uint16_t>(fish_call.gpr[2] & 0xffffu);
                if (stage_kind == kFishingSpotKind) {
                    return Target{0, 0, 1, false, true, false, false};
                }
                if (stage_kind == kSupplyBoxKind) {
                    const bool carrying = (m.load32(player + kPlayerTransportFlags) & 1u) != 0u;
                    return Target{0, 0, static_cast<std::uint16_t>(carrying ? 0u : 1u), false, false, true, false};
                }
                if (stage_kind == kDeliveryBoxKind) {
                    return Target{0, 0, 1, false, false, false, true};
                }
            }
            return {};
        }
    } else {
        // The static probe ignores the remaining-use field; inspect the matched
        // node.
        const auto stage = m.load32(*region::mh_stage_manager);
        if (!m.contains(stage, 0x64)) return {};
        auto list_call = ctx;
        list_call.gpr[4] = stage;
        list_call.gpr[5] = m.load16(player + kPlayerArea);
        rt.call(list_call, *region::mh_stage_gathering_nodes);
        auto node = list_call.gpr[2];
        bool found = false;
        for (unsigned i = 0; i < 512 && m.contains(node, kStaticNodeSize); ++i, node += kStaticNodeSize) {
            if (m.load32(node) == 0xbf800000u) break; // x=-1 sentinel
            if (m.load16(node + kNodeId) != index || m.load16(node + kNodeType) != type) continue;
            if (m.load32(node) != m.load32(scratch) || m.load32(node + 4) != m.load32(scratch + 4) ||
                m.load32(node + 8) != m.load32(scratch + 8))
                continue;
            remaining = m.load16(node + kNodeRemaining);
            found = true;
            break;
        }
        if (!found) return {};
    }
    return Target{index, type, remaining, dynamic && type == 2, false, false, false};
}

} // namespace

void Gathering::update(psp::Runtime &rt, psp::Context &ctx) {
    if (busy_) return;
    visual_active_ = false;
    struct Fade {
        std::uint8_t &alpha;
        const bool &active;
        ~Fade() { alpha = gathering_alpha(alpha, active); }
    } fade{alpha_, visual_active_};
    const bool circle = (psp::controller_state(rt).buttons & kCircle) != 0;
    const bool pressed = circle && !held_;
    held_ = circle;
    hint_.clear();
    updated_ = rt.now_us();
    if constexpr (region::mh_gathering_probe && region::mh_gathering_interact && region::mh_inventory_slot &&
                  region::mh_game_sys && region::mh_player_manager && region::mh_cockpit &&
                  region::mh_gathering_dynamic_nodes && region::mh_stage_manager && region::mh_stage_gathering_nodes) {
        auto &m = rt.memory();
        auto cleanup_virtual = [&](const std::optional<VirtualTool> &vt) {
            if (!vt || !m.contains(vt->bag + 4u * vt->slot, 4u)) return;
            const std::uint32_t addr = vt->bag + 4u * vt->slot;
            const std::uint16_t cur_id = m.load16(addr);
            if (vt->saved.id == 0) {
                if (cur_id == vt->tool_id) {
                    m.store16(addr, std::uint16_t{0});
                    m.store16(addr + 2u, std::uint16_t{0});
                }
            } else {
                m.store16(addr, vt->saved.id);
                m.store16(addr + 2u, vt->saved.count);
            }
        };
        auto clear_selection = [&]() {
            if (selection_) {
                cleanup_virtual(selection_->virtual_tool);
                selection_.reset();
            }
        };
        const auto *overlay = rt.active_overlay_at(*region::mh_gathering_probe);
        if (!overlay || !std::string_view(overlay->name).ends_with("_ovl_game_task")) {
            clear_selection();
            repeat_ = {};
            tool_repeat_ = {};
            return;
        }
        const auto sys = m.load32(*region::mh_game_sys), manager = m.load32(*region::mh_player_manager);
        if (!m.contains(sys, 0x29) || !m.contains(manager, 0x24)) {
            clear_selection();
            repeat_ = {};
            tool_repeat_ = {};
            return;
        }
        const auto slot = m.load8(sys + kGameSysPlayerSlot);
        if (slot >= kPlayerSlots) {
            clear_selection();
            repeat_ = {};
            tool_repeat_ = {};
            return;
        }
        const auto player = m.load32(manager + kPlayerManagerPlayers + slot * 4u);
        if (!m.contains(player, 0x696)) {
            clear_selection();
            repeat_ = {};
            tool_repeat_ = {};
            return;
        }
        // Keep the tool selected (and any temporary virtual tool present) until the action finishes.
        // Never undo a later selection or write to a player that has left.
        if (selection_) {
            if (selection_->player != player || m.load8(player + kPlayerReady) > 1 ||
                m.load16(player + kPlayerSelectedItem) != selection_->tool) {
                clear_selection();
            } else if (m.load8(player + kPlayerAction) != selection_->action) {
                m.store16(player + kPlayerSelectedItem, selection_->previous);
                clear_selection();
            }
        }
        if (!modern_carving || !circle) {
            repeat_ = {};
            tool_repeat_ = {};
        }
        if (!automatic_tools && !infinite_tools) tool_repeat_ = {};
        if (!indicators && !quest_box_icons && !automatic_tools && !modern_carving && !infinite_tools) return;
        if ((input_blocked && input_blocked()) || psp::current_dialog(rt).kind != psp::DialogView::Kind::None) {
            repeat_ = {};
            tool_repeat_ = {};
            return;
        }
        const auto cockpit = m.load32(*region::mh_cockpit);
        if (!m.contains(cockpit, 0x605) || m.load8(sys + kGameSysMenu) != 0 || m.load8(cockpit + kCockpitMenu) != 0 ||
            m.load8(cockpit + kCockpitWindow) != 0) {
            repeat_ = {};
            tool_repeat_ = {};
            return;
        }
        if (m.load8(player + kPlayerReady) > 1) {
            repeat_ = {};
            tool_repeat_ = {};
            return;
        }
        if (!scratch_) {
            const auto block = rt.allocate_kernel_block("gathering-query", 32);
            if (!block) return;
            scratch_ = *block;
        }
        busy_ = true;
        struct Guard {
            bool &b;
            ~Guard() { b = false; }
        } guard{busy_};
        // Every guest call gets a copy: probe/accessor/interaction must not clobber
        // the interrupted game's context.
        const auto target = read_target(rt, ctx, player, scratch_);
        const auto action = m.load8(player + kPlayerAction);
        const auto area = m.load16(player + kPlayerArea);
        std::uint8_t repeat_action = 0;
        if (target && target->remaining != 0 && !target->fishing && !target->supply_box && !target->delivery_box &&
            (target->carving || (target->type != 3 && target->type != 4)))
            repeat_action = target->carving ? kCarveAction : kGatherAction;
        const bool repeatable = modern_carving && target && target->remaining != 0 && repeat_action != 0;
        const bool same_target = target && repeat_.player == player && repeat_.area == area &&
                                 repeat_.target == target->index && repeat_.type == target->type;
        if (repeat_.armed || repeat_.working) {
            if (repeat_.player != player || repeat_.area != area || (target && !same_target)) {
                repeat_ = {};
            } else if (repeat_.working && action == repeat_.action) {
                // Keep the target latched through the game's normal gathering or carving animation.
            } else if (repeat_.armed && action == repeat_.action && m.load8(player + kPlayerReady) == 1) {
                repeat_.armed = false;
                repeat_.working = true;
            } else if (repeat_.armed && action == 0 && repeatable && same_target) {
                // Wait for the first action to start; the game's own Circle handling starts it.
            } else {
                repeat_ = {};
            }
        } else if (repeatable && circle && action == repeat_action && m.load8(player + kPlayerReady) == 1) {
            // Circle may have started the action just before this hook first observed the target.
            repeat_ = {player, target->index, target->type, area, repeat_action, false, true};
        } else if (repeatable && circle && action == 0 && m.load8(player + kPlayerReady) == 0) {
            // Arm on the same target before the game's first action starts.
            repeat_ = {player, target->index, target->type, area, repeat_action, true, false};
        }
        if (!target) {
            tool_repeat_ = {};
            return;
        }
        const auto [index, type, remaining, carving, fishing, supply_box, delivery_box] = *target;
        if (repeat_.working && modern_carving) {
            if (repeat_target_ != index) { repeat_target_ = index; repeat_remaining_ = remaining; reward_received_ = false; }
            else if (remaining < repeat_remaining_) {
                reward_received_ = true; repeat_remaining_ = remaining;
                hold_crouched_pose_ = remaining != 0; pose_next_cycle_ = false;
            }
            else if (remaining > repeat_remaining_) { repeat_remaining_ = remaining; reward_received_ = false; }
        }
        auto bag_call = ctx;
        bag_call.gpr[4] = sys;
        bag_call.gpr[5] = 0;
        rt.call(bag_call, *region::mh_inventory_slot);
        const auto bag = bag_call.gpr[2];
        if (!m.contains(bag, kInventorySlots * 4)) return;
        std::array<GatheringItem, kInventorySlots> items{};
        for (std::uint32_t i = 0; i < kInventorySlots; ++i)
            items[i] = {m.load16(bag + 4u * i), m.load16(bag + 4u * i + 2)};
        const bool deliverable = delivery_box && has_deliverable_item(m, player, items);
        const auto decision = decide_gathering(
            {index, type, items, remaining == 0, carving, fishing, infinite_tools, supply_box, delivery_box, deliverable});
        const bool box_kind = decision.kind == GatheringKind::supply_box || decision.kind == GatheringKind::delivery_box;
        const bool tool_kind = decision.kind == GatheringKind::mining || decision.kind == GatheringKind::bugs ||
                               decision.kind == GatheringKind::fishing;
        const bool allow_tool_circle = automatic_tools || infinite_tools;
        const bool has_usable_tool = decision.tool_slot.has_value() || decision.uses_virtual_tool;
        const bool show_prompt = box_kind ? quest_box_icons : indicators;
        // Hide the fishing prompt while actively fishing so the screen stays clear for the bobber.
        visual_active_ = show_prompt && decision.kind != GatheringKind::none && !(fishing && action == kFishingAction);
        kind_ = decision.kind;
        unavailable_ = !decision.available;
        button_ = decision.available &&
                  (box_kind || decision.kind == GatheringKind::gather || decision.kind == GatheringKind::carve ||
                   (tool_kind && allow_tool_circle && has_usable_tool));
        if (visual_active_) {
            const auto &text = kHints[language >= 0 && static_cast<std::size_t>(language) < kHints.size()
                                         ? static_cast<std::size_t>(language) : 1u];
            if (decision.kind == GatheringKind::mining)
                hint_ = text.mine;
            else if (decision.kind == GatheringKind::bugs)
                hint_ = text.bugs;
            else if (decision.kind == GatheringKind::fishing)
                hint_ = text.fish;
            else if (decision.kind == GatheringKind::gather)
                hint_ = text.gather;
            else if (decision.kind == GatheringKind::carve)
                hint_ = text.carve;
            else if (decision.kind == GatheringKind::supply_box)
                hint_ = text.supply_box;
            else if (decision.kind == GatheringKind::delivery_box)
                hint_ = text.delivery_box;
            if (decision.kind == GatheringKind::delivery_box) {
                if (!decision.available)
                    hint_.append(" - ").append(text.no_deliverable);
                else
                    hint_ = "○  " + hint_;
            } else if (remaining == 0) {
                hint_.append(" - ").append(text.depleted);
            } else if (tool_kind) {
                if (!has_usable_tool)
                    hint_.append(" - ").append(text.no_tool);
                else if (allow_tool_circle)
                    hint_ = "○  " + hint_;
            } else if (!hint_.empty())
                hint_ = "○  " + hint_;
        }
        if (!tool_kind) {
            tool_repeat_ = {};
            return;
        }
        const auto expected_action = decision.kind == GatheringKind::mining ? kMiningAction
                                     : decision.kind == GatheringKind::bugs ? kBugAction
                                                                            : kFishingAction;
        const bool repeatable_tool = decision.kind == GatheringKind::mining || decision.kind == GatheringKind::bugs;
        if (tool_repeat_.active) {
            const bool same_tool_target = tool_repeat_.player == player && tool_repeat_.area == area &&
                                          tool_repeat_.target == index && tool_repeat_.type == type;
            if (!repeatable_tool || !same_tool_target || !decision.available || !has_usable_tool ||
                (action != tool_repeat_.action && action != 0)) {
                tool_repeat_ = {};
            }
        } else if (modern_carving && allow_tool_circle && repeatable_tool && circle && decision.available &&
                   has_usable_tool && action == expected_action) {
            tool_repeat_ = ToolRepeat{player, index, type, area, expected_action, true};
        }
        const bool want_tool = pressed || (tool_repeat_.active && circle && action == 0);
        if (!allow_tool_circle || !want_tool || !has_usable_tool || !decision.available) return;
        if (action == kMiningAction || action == kBugAction || action == kFishingAction) return;

        std::uint16_t use_slot = 0;
        std::optional<VirtualTool> vt;
        if (decision.uses_virtual_tool) {
            use_slot = pick_virtual_slot(items);
            vt = VirtualTool{bag, use_slot, decision.virtual_tool_id, items[use_slot]};
            m.store16(bag + 4u * use_slot, decision.virtual_tool_id);
            m.store16(bag + 4u * use_slot + 2u, std::uint16_t{1});
        } else {
            use_slot = *decision.tool_slot;
        }

        const auto previous_slot = m.load16(player + kPlayerSelectedItem);
        m.store16(player + kPlayerSelectedItem, use_slot);
        try {
            auto interact = ctx;
            interact.gpr[4] = player;
            if (decision.kind == GatheringKind::fishing) {
                if constexpr (region::mh_item_use) {
                    interact.gpr[5] = 0;
                    rt.call(interact, *region::mh_item_use);
                }
            } else {
                interact.gpr[5] = 2;
                rt.call(interact, *region::mh_gathering_interact);
            }
        } catch (...) {
            m.store16(player + kPlayerSelectedItem, previous_slot);
            cleanup_virtual(vt);
            throw;
        }
        if (m.load8(player + kPlayerAction) != expected_action) {
            m.store16(player + kPlayerSelectedItem, previous_slot);
            cleanup_virtual(vt);
        } else {
            selection_ = Selection{player, previous_slot, use_slot, expected_action, vt};
            if (modern_carving && repeatable_tool)
                tool_repeat_ = ToolRepeat{player, index, type, area, expected_action, true};
        }
    }
}

bool Gathering::repeat_available(psp::Runtime &rt, psp::Context &ctx, std::uint32_t player) {
    if (!modern_carving || !repeat_.working || (psp::controller_state(rt).buttons & kCircle) == 0 ||
        (input_blocked && input_blocked()) || psp::current_dialog(rt).kind != psp::DialogView::Kind::None ||
        !scratch_ || busy_)
        return false;
    auto &m = rt.memory();
    if (player != repeat_.player || !m.contains(player, kPlayerAction + 1) ||
        m.load8(player + kPlayerAction) != repeat_.action ||
        m.load16(player + kPlayerArea) != repeat_.area)
        return false;
    busy_ = true;
    struct Guard {
        bool &busy;
        ~Guard() { busy = false; }
    } guard{busy_};
    const auto target = read_target(rt, ctx, player, scratch_);
    const bool available = target && target->index == repeat_.target && target->type == repeat_.type &&
                           target->remaining != 0 &&
                           (target->carving || (target->type != 3 && target->type != 4));
    return available;
}

bool Gathering::preserve_action(psp::Runtime &rt, psp::Context &ctx) {
    return repeat_available(rt, ctx, ctx.gpr[20]);
}

void Gathering::choose_crouched_pose(psp::Runtime &rt, psp::Context &ctx) {
    if (ctx.gpr[4] == repeat_.player && repeat_.action == kCarveAction && repeat_available(rt, ctx, ctx.gpr[4]))
        ctx.gpr[6] = 0x60u;
}

void Gathering::animation_sample(psp::Runtime &rt, psp::Context &ctx) {
    if (modern_carving && hold_crouched_pose_ && repeat_.working &&
        (psp::controller_state(rt).buttons & kCircle) && ctx.gpr[4] >= repeat_.player + 0x1200u &&
        ctx.gpr[4] < repeat_.player + 0x4100u && std::isfinite(ctx.fpr[12]))
        ctx.fpr[12] = 100.0f;
}

void Gathering::animation_update(psp::Runtime &rt, psp::Context &ctx) {
    if (hold_crouched_pose_ && repeat_.working && ctx.gpr[4] == repeat_.player + 0x80u &&
        rt.memory().contains(ctx.gpr[4] + 16u, 4u)) {
        const auto clock = std::bit_cast<float>(rt.memory().load32(ctx.gpr[4] + 16u));
        if (clock < 20.0f) pose_next_cycle_ = true;
        if (pose_next_cycle_ && clock >= 100.0f) hold_crouched_pose_ = false;
    }
    if (!(psp::controller_state(rt).buttons & kCircle) || !repeat_.working) hold_crouched_pose_ = false;
    if (!modern_carving || !repeat_.working || ctx.gpr[4] != repeat_.player + 0x80u ||
        !(psp::controller_state(rt).buttons & kCircle)) return;
    auto &memory = rt.memory();
    const auto object = ctx.gpr[4];
    if (!memory.contains(object, 32u)) return;
    const auto clock = std::bit_cast<float>(memory.load32(object + 16u));
    const auto step = std::bit_cast<float>(memory.load32(object + 20u));
    const auto end = std::bit_cast<float>(memory.load32(object + 28u));
    if (clock < 20.0f) reward_received_ = false;
    const float extra = reward_received_ ? step : step * 0.25f;
    if (std::isfinite(clock) && std::isfinite(step) && std::isfinite(end) && step > 0.0f &&
        clock >= 0.0f && clock + step + extra < end)
        memory.store32(object + 16u, std::bit_cast<std::uint32_t>(clock + extra));
}

std::string Gathering::hint(psp::Runtime &rt) const {
    const bool is_box = kind_ == GatheringKind::supply_box || kind_ == GatheringKind::delivery_box;
    const bool enabled = is_box ? quest_box_icons : indicators;
    return enabled && rt.now_us() - updated_ < kHintLifetimeUs ? hint_ : std::string{};
}
std::vector<overlay::HintSprite> Gathering::sprites(psp::Runtime &rt) const {
    const bool is_box = kind_ == GatheringKind::supply_box || kind_ == GatheringKind::delivery_box;
    const bool enabled = is_box ? quest_box_icons : indicators;
    if (!enabled || !visual_active_ || !art_ || rt.now_us() - updated_ >= kHintLifetimeUs)
        return {};
    return gathering_sprites(*art_, kind_, alpha_, button_, unavailable_);
}
} // namespace mhf::mhfu
