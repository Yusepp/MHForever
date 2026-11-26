#pragma once

// Game-agnostic, sceCtrl-poll-synced input recorder and deterministic replayer (developer/testing aid).
// Synced strictly to the emulation thread's exact sceCtrl poll counter so replay timing is 100% deterministic
// with zero host-vs-emulation thread drift across save loads, area transitions, and 30/60/120 Hz pacing.
//
// Environment variables:
//   MHF_RECORD_INPUT=<path>       Record ControllerState keyframes + F10 SNAP bookmarks to <path>
//   MHF_REPLAY_INPUT=<path>       Replay ControllerState keyframes from <path>
//   MHF_REPLAY_SNAP_EVERY=<N>     Optional: trigger Vulkan PNG snapshot every N ctrl polls during replay
//   MHF_REPLAY_SNAP_FRAMES=<list> Optional: comma-separated poll numbers to snapshot (e.g. "1200,2600")
//   MHF_REPLAY_EXIT=1             Exit cleanly 120 polls after the final recorded poll

#include "mhf/psp/host.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace mhf::psp {

class InputTape {
public:
    ~InputTape() { flush_record(); }

    void init_from_env() {
        if (initialized_) return;
        initialized_ = true;

        if (const char *rep = std::getenv("MHF_REPLAY_INPUT"); rep && rep[0] != '\0') {
            load_tape(rep);
            if (const char *ex = std::getenv("MHF_REPLAY_EXIT"); ex && ex[0] == '1') {
                exit_on_end_ = true;
            }
            if (const char *ev = std::getenv("MHF_REPLAY_SNAP_EVERY"); ev && ev[0] != '\0') {
                snap_every_ = static_cast<std::uint64_t>((std::max)(0, std::atoi(ev)));
            }
            if (const char *sf = std::getenv("MHF_REPLAY_SNAP_FRAMES"); sf && sf[0] != '\0') {
                std::stringstream ss(sf);
                std::string tok;
                while (std::getline(ss, tok, ',')) {
                    if (!tok.empty()) {
                        snap_polls_.push_back(static_cast<std::uint64_t>(std::strtoull(tok.c_str(), nullptr, 10)));
                    }
                }
            }
            std::sort(snap_polls_.begin(), snap_polls_.end());
            snap_polls_.erase(std::unique(snap_polls_.begin(), snap_polls_.end()), snap_polls_.end());
            return;
        }

        if (const char *rec = std::getenv("MHF_RECORD_INPUT"); rec && rec[0] != '\0') {
            std::filesystem::path p(rec);
            if (p.has_parent_path()) {
                std::error_code ec;
                std::filesystem::create_directories(p.parent_path(), ec);
            }
            out_.open(p, std::ios::out | std::ios::trunc);
            if (out_.is_open()) {
                recording_ = true;
                out_ << "# MHF_INPUT_TAPE v2 (sceCtrl-poll-synced)\n";
                out_ << "# poll buttons_hex dpad_hex lx ly rx ry mx my\n";
                out_.flush();
                std::fprintf(stderr, "[input-tape] recording sceCtrl-poll-synced inputs to %s (press F10 for SNAP)\n",
                             p.string().c_str());
            } else {
                std::fprintf(stderr, "[input-tape] failed to open %s for recording\n", p.string().c_str());
            }
        }
    }

    [[nodiscard]] bool recording() const noexcept { return recording_; }
    [[nodiscard]] bool replaying() const noexcept { return replaying_; }

    // Called inside sceCtrl write_samples() on the emulation thread. Returns true if a SNAP bookmark was recorded.
    bool on_record_poll(std::uint64_t poll_idx, const ControllerState &state, bool snap_pressed) {
        if (!recording_) return false;
        last_recorded_poll_ = poll_idx;
        emit_if_changed(poll_idx, state);
        if (snap_pressed && poll_idx != last_snap_recorded_) {
            last_snap_recorded_ = poll_idx;
            out_ << "SNAP " << poll_idx << "\n";
            out_.flush();
            std::fprintf(stderr, "[input-tape] SNAP bookmark at ctrl_poll=%llu\n",
                         static_cast<unsigned long long>(poll_idx));
            return true;
        }
        return false;
    }

    void flush_record() {
        if (!recording_) return;
        out_ << "END " << last_recorded_poll_ << "\n";
        out_.flush();
        out_.close();
        recording_ = false;
    }

    // Called inside sceCtrl write_samples() on the emulation thread.
    void on_replay_poll(std::uint64_t poll_idx, ControllerState &out_state, bool &out_want_snap, bool &out_want_exit) {
        out_want_snap = false;
        out_want_exit = false;
        if (!replaying_) return;

        while (cursor_ < keys_.size() && keys_[cursor_].poll <= poll_idx) {
            current_state_ = keys_[cursor_].state;
            ++cursor_;
        }
        out_state = current_state_;

        while (snap_cursor_ < snap_polls_.size() && snap_polls_[snap_cursor_] <= poll_idx) {
            out_want_snap = true;
            ++snap_cursor_;
        }
        if (snap_every_ > 0u && poll_idx > 0u && (poll_idx / snap_every_) > last_periodic_snap_bucket_) {
            last_periodic_snap_bucket_ = poll_idx / snap_every_;
            out_want_snap = true;
        }
        if (exit_on_end_ && poll_idx > end_poll_ + 120u) {
            out_want_exit = true;
        }
    }

private:
    struct Keyframe {
        std::uint64_t poll{};
        ControllerState state{};
    };

    static bool states_equal(const ControllerState &a, const ControllerState &b) noexcept {
        return a.buttons == b.buttons && a.dpad_buttons == b.dpad_buttons &&
               a.lx == b.lx && a.ly == b.ly &&
               std::fabs(a.right_stick_x - b.right_stick_x) < 1e-4f &&
               std::fabs(a.right_stick_y - b.right_stick_y) < 1e-4f &&
               std::fabs(a.mouse_x - b.mouse_x) < 0.01 &&
               std::fabs(a.mouse_y - b.mouse_y) < 0.01;
    }

    void emit_if_changed(std::uint64_t poll_idx, const ControllerState &s) {
        if (has_last_written_ && states_equal(last_written_, s)) return;
        char buf[256];
        std::snprintf(buf, sizeof(buf), "%llu %08x %08x %u %u %.5f %.5f %.2f %.2f\n",
                      static_cast<unsigned long long>(poll_idx), s.buttons, s.dpad_buttons,
                      static_cast<unsigned>(s.lx), static_cast<unsigned>(s.ly),
                      s.right_stick_x, s.right_stick_y, s.mouse_x, s.mouse_y);
        out_ << buf;
        out_.flush();
        last_written_ = s;
        has_last_written_ = true;
    }

    void load_tape(const char *path) {
        std::ifstream in(path);
        if (!in.is_open()) {
            std::fprintf(stderr, "[input-tape] failed to open %s for replay\n", path);
            return;
        }
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;
            if (line.rfind("SNAP ", 0) == 0) {
                const auto f = static_cast<std::uint64_t>(std::strtoull(line.c_str() + 5, nullptr, 10));
                snap_polls_.push_back(f);
                end_poll_ = (std::max)(end_poll_, f);
                continue;
            }
            if (line.rfind("END ", 0) == 0) {
                const auto f = static_cast<std::uint64_t>(std::strtoull(line.c_str() + 4, nullptr, 10));
                end_poll_ = (std::max)(end_poll_, f);
                continue;
            }
            unsigned long long poll = 0;
            unsigned buttons = 0, dpad = 0, lx = 128, ly = 128;
            float rx = 0.0f, ry = 0.0f;
            double mx = 0.0, my = 0.0;
            if (std::sscanf(line.c_str(), "%llu %x %x %u %u %f %f %lf %lf",
                            &poll, &buttons, &dpad, &lx, &ly, &rx, &ry, &mx, &my) == 9) {
                Keyframe k{};
                k.poll = static_cast<std::uint64_t>(poll);
                k.state.buttons = buttons;
                k.state.dpad_buttons = dpad;
                k.state.lx = static_cast<std::uint8_t>(std::clamp(lx, 0u, 255u));
                k.state.ly = static_cast<std::uint8_t>(std::clamp(ly, 0u, 255u));
                k.state.right_stick_x = rx;
                k.state.right_stick_y = ry;
                k.state.mouse_x = mx;
                k.state.mouse_y = my;
                keys_.push_back(k);
                end_poll_ = (std::max)(end_poll_, k.poll);
            }
        }
        if (!keys_.empty() || !snap_polls_.empty()) {
            replaying_ = true;
            std::fprintf(stderr, "[input-tape] loaded %zu keyframes, %zu SNAP bookmarks (end_poll=%llu) from %s\n",
                         keys_.size(), snap_polls_.size(), static_cast<unsigned long long>(end_poll_), path);
        }
    }

    bool initialized_{false};
    bool recording_{false};
    bool replaying_{false};
    bool exit_on_end_{false};

    std::ofstream out_;
    std::uint64_t last_recorded_poll_{0};
    bool has_last_written_{false};
    ControllerState last_written_{};
    std::uint64_t last_snap_recorded_{~0ull};

    std::vector<Keyframe> keys_;
    std::size_t cursor_{0};
    ControllerState current_state_{};
    std::vector<std::uint64_t> snap_polls_;
    std::size_t snap_cursor_{0};
    std::uint64_t snap_every_{0};
    std::uint64_t last_periodic_snap_bucket_{0};
    std::uint64_t end_poll_{0};
};

} // namespace mhf::psp
