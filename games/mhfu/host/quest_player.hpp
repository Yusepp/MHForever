#pragma once

#include "psprecomp/guest_memory.hpp"

namespace mhf::mhfu {

// The Player* of the hunter on this machine, or zero during transitions.
inline std::uint32_t local_player(const psprecomp::GuestMemory &memory, std::uint32_t game_sys,
                                  std::uint32_t player_manager, std::size_t player_size) {
    constexpr std::uint32_t kPlayerSlot = 0x28, kPlayers = 0x0C, kSlots = 6;
    if (!memory.contains(game_sys, kPlayerSlot + 1) || !memory.contains(player_manager, 4)) return 0;
    const std::uint32_t slot = memory.load8(game_sys + kPlayerSlot);
    const std::uint32_t manager = memory.load32(player_manager);
    if (slot >= kSlots || (manager & 3u) != 0 || !memory.contains(manager, kPlayers + kSlots * 4)) return 0;
    const std::uint32_t player = memory.load32(manager + kPlayers + slot * 4);
    return (player & 3u) == 0 && memory.contains(player, player_size) ? player : 0;
}

// The game's own Cockpit_menu_chk and the cockpit's open-window flag.
inline bool quest_menu_open(const psprecomp::GuestMemory &memory, std::uint32_t game_sys, std::uint32_t cockpit) {
    constexpr std::uint32_t kGameSysMenu = 0x10, kCockpitMenu = 0x604, kCockpitWindow = 0x52F;
    return (memory.contains(game_sys, kGameSysMenu + 1) && memory.load8(game_sys + kGameSysMenu) != 0) ||
           (memory.contains(cockpit, kCockpitMenu + 1) &&
            (memory.load8(cockpit + kCockpitMenu) != 0 || memory.load8(cockpit + kCockpitWindow) != 0));
}

} // namespace mhf::mhfu
