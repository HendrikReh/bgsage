// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Mark Higgins
#include "bgbot/cube.h"
#include "bgbot/board.h"
#include "bgbot/moves.h"
#include "bgbot/neural_net.h"
#include "bgbot/multipy.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <cstdio>
#include <array>
#include <vector>
#include <unordered_map>
#include <atomic>

namespace bgbot {

// Profiling counters of the old cube.cpp cubeful recursion, which only the
// incl_2ply_details path still used and which was removed on 2026-11-19 (the
// details now run on the engine in cube_eval.cpp). The engine does not count,
// so these stay 0; the API is kept for the profiling tools that print them.
static std::atomic<int64_t> g_leaf_count{0};
static std::atomic<int64_t> g_internal_count{0};
static std::atomic<int64_t> g_cache_hit_count{0};
static std::atomic<int64_t> g_move_gen_count{0};
static std::atomic<int64_t> g_total_candidates{0};

void reset_cubeful_counters() {
    g_leaf_count = 0;
    g_internal_count = 0;
    g_cache_hit_count = 0;
    g_move_gen_count = 0;
    g_total_candidates = 0;
}

void print_cubeful_counters() {
    printf("  Leaf evaluations: %lld\n", (long long)g_leaf_count.load());
    printf("  Internal nodes:   %lld\n", (long long)g_internal_count.load());
    printf("  Cache hits:       %lld\n", (long long)g_cache_hit_count.load());
    printf("  Move gen calls:   %lld\n", (long long)g_move_gen_count.load());
    printf("  Total candidates: %lld (avg %.1f)\n",
           (long long)g_total_candidates.load(),
           g_move_gen_count.load() > 0
               ? (double)g_total_candidates.load() / g_move_gen_count.load()
               : 0.0);
}

CubefulCounters get_cubeful_counters() {
    return CubefulCounters{
        g_leaf_count.load(),
        g_internal_count.load(),
        g_cache_hit_count.load(),
        g_move_gen_count.load(),
        g_total_candidates.load(),
    };
}

void compute_WL(const std::array<float, NUM_OUTPUTS>& probs, float& W, float& L) {
    float p_win = probs[0];
    float p_gw = probs[1];
    float p_bw = probs[2];
    float p_gl = probs[3];
    float p_bl = probs[4];

    // W = average value of wins: 1 (single) + fraction that are gammon/bg
    if (p_win > 1e-7f) {
        W = 1.0f + (p_gw + p_bw) / p_win;
    } else {
        W = 1.0f;  // No wins — W doesn't matter, but avoid division by zero
    }

    // L = average value of losses
    float p_lose = 1.0f - p_win;
    if (p_lose > 1e-7f) {
        L = 1.0f + (p_gl + p_bl) / p_lose;
    } else {
        L = 1.0f;  // No losses
    }
}

float money_live(float W, float L, float p_win, CubeOwner owner,
                 bool jacoby) {
    // Take point and cash point (live cube)
    float TP = (L - 0.5f) / (W + L + 0.5f);
    float CP = (L + 1.0f) / (W + L + 0.5f);

    float p = p_win;

    switch (owner) {
        case CubeOwner::CENTERED: {
            // Interpolate through (0, -L), (TP, -1), (CP, +1), (1, +W)
            //
            // Under Jacoby, the extreme segments are clamped to ±1.0
            // (matching GNUbg's MoneyLive).  Rationale: above the cash
            // point, the player doubles and the opponent passes → +1.0;
            // below the take point, the opponent doubles and the player
            // passes → -1.0.  There is no "too good" / "too weak" premium
            // because gammons are worthless while the cube is centered.
            // The middle segment (doubling window) is unchanged because
            // once the cube is turned, gammons count — the real W/L apply.
            if (p < TP) {
                // Segment: (0, -L) → (TP, -1)
                return jacoby ? -1.0f
                              : -L + (-1.0f + L) * p / TP;
            } else if (p < CP) {
                // Segment: (TP, -1) → (CP, +1)
                return -1.0f + 2.0f * (p - TP) / (CP - TP);
            } else {
                // Segment: (CP, +1) → (1, +W)
                return jacoby ? 1.0f
                              : 1.0f + (W - 1.0f) * (p - CP) / (1.0f - CP);
            }
        }
        case CubeOwner::PLAYER: {
            // Player owns cube: interpolate through (0, -L), (CP, +1), (1, +W)
            if (p < CP) {
                return -L + (1.0f + L) * p / CP;
            } else {
                return 1.0f + (W - 1.0f) * (p - CP) / (1.0f - CP);
            }
        }
        case CubeOwner::OPPONENT: {
            // Opponent owns cube: interpolate through (0, -L), (TP, -1), (1, +W)
            if (p < TP) {
                return -L + (-1.0f + L) * p / TP;
            } else {
                return -1.0f + (W + 1.0f) * (p - TP) / (1.0f - TP);
            }
        }
    }
    return 0.0f;  // unreachable
}

float cl2cf_money(const std::array<float, NUM_OUTPUTS>& probs,
                  CubeOwner owner, float cube_x) {
    return cl2cf_money(probs, owner, cube_x, false);
}

float cl2cf_money(const std::array<float, NUM_OUTPUTS>& probs,
                  CubeOwner owner, float cube_x, bool jacoby_active) {
    float W, L;
    // Always compute real W/L from the full gammon/backgammon rates.
    // The live-cube equity uses real W/L because once the cube is turned,
    // gammons count (Jacoby deactivates).
    compute_WL(probs, W, L);

    // Dead-cube equity: under Jacoby with cube centered, the cube can
    // never be turned again means it stays centered forever, so gammons
    // never count → 2*P(win)-1.  Otherwise (Jacoby off, or cube already
    // turned) full cubeless equity with gammons/backgammons.
    float e_dead = jacoby_active ? (2.0f * probs[0] - 1.0f)
                                 : cubeless_equity(probs);

    // Live-cube equity: uses real W/L but with Jacoby clamping on the
    // extreme segments (p < TP → -1, p > CP → +1).  Matching GNUbg's
    // MoneyLive: the middle segment (doubling window) is unchanged because
    // doubling activates gammons; the extreme segments are clamped because
    // there is no "too good/too weak" premium under Jacoby (you'd just
    // double to cash or get doubled out).
    float e_live = money_live(W, L, probs[0], owner, jacoby_active);

    return e_dead * (1.0f - cube_x) + e_live * cube_x;
}

// ---------------------------------------------------------------------------
// Match play cubeful evaluation (Janowski in MWC space)
// ---------------------------------------------------------------------------

// Check if the cube is "dead" — no useful cube actions possible.
// This happens when the cube value is enough for both players to win the match
// on any win, or during Crawford game.
static bool is_dead_cube(const CubeInfo& cube) {
    if (cube_is_dead(cube)) return true;          // structural cap
    if (cube.is_money()) return false;
    if (cube.match.is_crawford) return true;
    // A cube can only be turned again by a player who BOTH may double it
    // (ownership) AND gains by doing so -- a player whose win at the current
    // cube value already takes the match gains nothing. Ownership matters:
    // with an owned cube only the owner can turn it, so the owner's away score
    // alone decides. Requiring BOTH away scores to be covered is correct only
    // for a CENTERED cube, and left e.g. 3-away/2-away with the cube on 2 owned
    // by the trailer looking live.
    const bool player_can_turn =
        (cube.owner == CubeOwner::CENTERED || cube.owner == CubeOwner::PLAYER)
        && cube.match.away1 > cube.cube_value;
    const bool opponent_can_turn =
        (cube.owner == CubeOwner::CENTERED || cube.owner == CubeOwner::OPPONENT)
        && cube.match.away2 > cube.cube_value;
    return !player_can_turn && !opponent_can_turn;
}

// Compute MWC-space take point and cash point for match play.
// These are the P(win) thresholds analogous to money game TP/CP.
//
// For the player's perspective:
// TP = opponent's take point (P(win) below which opponent should pass)
//    = (MWC_lose_cube - MWC_dp) / (MWC_lose_cube - MWC_win_cube)
// CP = player's cash point (P(win) above which player should play on for gammon)
//    = (MWC_dp - MWC_win_cube_lose) / (MWC_win_cube_win - MWC_win_cube_lose)
//
// But we use the Janowski formulation which works in MWC space directly.
// The take point and cash point in MWC space map to specific P(win) values.

// Compute cubeful MWC for match play — centered cube.
// Piecewise-linear interpolation with 3 regions:
//   p < opponent's TG: opponent too good to double
//   opponent's TG < p < player's TG: in doubling window
//   p > player's TG: player too good to double
// Automatic redouble detection (GNUbg's GetCubePrimeValue).
// When the doubler (away_i) is close enough that losing at cv or 2*cv costs
// the same, they always take a recube — making the effective cube 2*cv.
// Uses 1-indexed away values (GNUbg uses 0-indexed).
static int get_cube_prime_value(int away_i, int away_j, int cv) {
    if ((away_i - 1) < 2 * cv && (away_j - 1) >= 2 * cv)
        return 2 * cv;
    return cv;
}

// Compute live-cube cash points for both players simultaneously using
// GNUbg's recursive GetPoints algorithm.
//
// The algorithm works from the "dead" cube level (highest level where at
// least one player can't survive a recube) DOWN to the current cube level,
// computing cash points recursively.
//
// At dead cube levels: CP = (DTL - DP) / (DTL - DTW)
//   with the doubler's own prime for BOTH DTW and DTL.
// At live cube levels: CP = 1 - opp_CP_above * (DP - DTW) / (rRDP - DTW)
//   where rRDP = MWC after losing at the doubled value (recube-drop payoff).
//
// Parameters use 1-indexed away values. Gammon ratios are from the player's
// perspective: rG0/rBG0 for player's wins, rG1/rBG1 for player's losses.
void get_match_points(
    int away1, int away2, int cv, bool craw,
    float rG0, float rBG0,        // gammon/bg ratios for player's wins
    float rG1, float rBG1,        // gammon/bg ratios for player's losses
    float& player_cp,             // out: player's cash point
    float& opp_cp)                // out: opponent's cash point
{
    // 0-indexed away values (GNUbg convention)
    int idx0 = away1 - 1;
    int idx1 = away2 - 1;

    // Gammon/bg ratios per side (0=player, 1=opponent)
    float arG[2] = {rG0, rG1};
    float arBG[2] = {rBG0, rBG1};

    // Find the dead cube level: highest cube where both can survive a recube
    int nDead = cv;
    int nMax = 0;
    while (idx0 >= 2 * nDead && idx1 >= 2 * nDead) {
        nMax++;
        nDead *= 2;
    }

    // Cash points at each level (k=0: player, k=1: opponent)
    constexpr int MAX_LEVELS = 16;
    float arCPLive[2][MAX_LEVELS] = {};

    for (int nCubeValue = nDead, n = nMax; nCubeValue >= cv;
         nCubeValue >>= 1, n--) {
        // Auto-redouble primes at this cube level (0-indexed comparisons)
        int prime[2];
        prime[0] = (idx0 < 2*nCubeValue && idx1 >= 2*nCubeValue)
                   ? 2*nCubeValue : nCubeValue;
        prime[1] = (idx1 < 2*nCubeValue && idx0 >= 2*nCubeValue)
                   ? 2*nCubeValue : nCubeValue;

        for (int k = 0; k < 2; k++) {
            // k=0: player doubles, k=1: opponent doubles
            int away_d = (k == 0) ? away1 : away2;
            int away_t = (k == 0) ? away2 : away1;
            int id = (k == 0) ? idx0 : idx1;
            int it = (k == 0) ? idx1 : idx0;

            float gW = arG[k], bgW = arBG[k];       // doubler's wins
            float gL = arG[1-k], bgL = arBG[1-k];   // doubler's losses

            if (id < 2*nCubeValue || it < 2*nCubeValue) {
                // Dead cube: at least one player can't survive a recube.
                // Use doubler's own prime for BOTH DTW and DTL.
                // The effective cube after DT (with possible auto-redouble) is
                // 2*prime[k], so outcomes are at 2*p, 4*p, 6*p (not p, 2p, 3p).
                int p = prime[k];
                int dp = 2 * p;  // effective cube level after DT + auto-redouble

                float rDP = get_met_after(away_d, away_t, nCubeValue, true, craw);

                float rDTW = (1.0f - gW - bgW)
                                * get_met_after(away_d, away_t, dp, true, craw)
                           + gW * get_met_after(away_d, away_t, 2*dp, true, craw)
                           + bgW * get_met_after(away_d, away_t, 3*dp, true, craw);

                float rDTL = (1.0f - gL - bgL)
                                * get_met_after(away_d, away_t, dp, false, craw)
                           + gL * get_met_after(away_d, away_t, 2*dp, false, craw)
                           + bgL * get_met_after(away_d, away_t, 3*dp, false, craw);

                float denom = rDTL - rDTW;
                if (std::abs(denom) < 1e-10f)
                    arCPLive[k][n] = 0.5f;
                else
                    arCPLive[k][n] = std::clamp(
                        (rDTL - rDP) / denom, 0.0f, 1.0f);
            } else {
                // Live cube: both players can survive a recube.
                // Use standard doubled value, recursive formula.
                int dcv = 2 * nCubeValue;

                float rDP = get_met_after(away_d, away_t,
                                          nCubeValue, true, craw);
                float rRDP = get_met_after(away_d, away_t,
                                           dcv, false, craw);

                float rDTW = (1.0f - gW - bgW)
                                * get_met_after(away_d, away_t, dcv, true, craw)
                           + gW * get_met_after(away_d, away_t, 2*dcv, true, craw)
                           + bgW * get_met_after(away_d, away_t, 3*dcv, true, craw);

                float denom = rRDP - rDTW;
                if (std::abs(denom) < 1e-10f)
                    arCPLive[k][n] = 0.5f;
                else
                    arCPLive[k][n] = std::clamp(
                        1.0f - arCPLive[1-k][n+1] * (rDP - rDTW) / denom,
                        0.0f, 1.0f);
            }
        }
    }

    player_cp = arCPLive[0][0];
    opp_cp = arCPLive[1][0];
}

static float cl2cf_match_centered(
    const std::array<float, NUM_OUTPUTS>& probs,
    const CubeInfo& cube,
    float cube_x)
{
    int away1 = cube.match.away1;
    int away2 = cube.match.away2;
    int cv = cube.cube_value;
    bool craw = cube.match.is_crawford;

    float p_win = probs[0];

    // Gammon/backgammon ratios
    float rG0, rBG0, rG1, rBG1;
    if (p_win > 1e-7f) {
        rG0 = (probs[1] - probs[2]) / p_win;
        rBG0 = probs[2] / p_win;
    } else {
        rG0 = 0.0f;
        rBG0 = 0.0f;
    }
    if (p_win < 1.0f - 1e-7f) {
        rG1 = (probs[3] - probs[4]) / (1.0f - p_win);
        rBG1 = probs[4] / (1.0f - p_win);
    } else {
        rG1 = 0.0f;
        rBG1 = 0.0f;
    }

    // Dead cube MWC (uses MET-weighted outcome values, not linear approximation)
    float mwc_dead = cubeless_mwc(probs, away1, away2, cv, craw);

    // MET lookups at current cube value (for no-double outcomes)
    float mwc_win_s  = get_met_after(away1, away2, cv, true, craw);
    float mwc_win_g  = get_met_after(away1, away2, 2*cv, true, craw);
    float mwc_win_b  = get_met_after(away1, away2, 3*cv, true, craw);
    float mwc_lose_s = get_met_after(away1, away2, cv, false, craw);
    float mwc_lose_g = get_met_after(away1, away2, 2*cv, false, craw);
    float mwc_lose_b = get_met_after(away1, away2, 3*cv, false, craw);

    // D/P MWC (player cashes cv points)
    float mwc_cash = mwc_win_s;
    // Opponent's D/P MWC (player loses cv points) — from player's perspective
    float mwc_opp_cash = mwc_lose_s;

    // Weighted MWCs for all win/loss types
    float mwc_win_all = (1.0f - rG0 - rBG0) * mwc_win_s
                      + rG0 * mwc_win_g + rBG0 * mwc_win_b;
    float mwc_lose_all = (1.0f - rG1 - rBG1) * mwc_lose_s
                       + rG1 * mwc_lose_g + rBG1 * mwc_lose_b;

    // Compute live-cube cash points using recursive algorithm (GNUbg GetPoints).
    float player_cp, opp_cp;
    get_match_points(away1, away2, cv, craw, rG0, rBG0, rG1, rBG1,
                     player_cp, opp_cp);

    // Convert to the thresholds used in the piecewise interpolation
    float opp_tg = 1.0f - opp_cp;   // Below this P(win), opponent is too good
    float player_tg = player_cp;      // Above this P(win), player is too good

    // Piecewise-linear live cube MWC (3 regions, following GNUbg Cl2CfMatchCentered)
    float mwc_live;

    if (p_win <= opp_tg) {
        // Region 1: Opponent too good to double
        // Linear from (0, mwc_lose_all) to (opp_tg, mwc_opp_cash)
        if (opp_tg > 1e-10f) {
            mwc_live = mwc_lose_all + (mwc_opp_cash - mwc_lose_all) * p_win / opp_tg;
        } else {
            mwc_live = mwc_lose_all;
        }
    } else if (p_win < player_tg) {
        // Region 2: In doubling window
        // Linear from (opp_tg, mwc_opp_cash) to (player_tg, mwc_cash)
        float range = player_tg - opp_tg;
        if (range > 1e-10f) {
            mwc_live = mwc_opp_cash + (mwc_cash - mwc_opp_cash) * (p_win - opp_tg) / range;
        } else {
            mwc_live = mwc_opp_cash;
        }
    } else {
        // Region 3: Player too good to double (play on for gammon)
        // Linear from (player_tg, mwc_cash) to (1, mwc_win_all)
        float range = 1.0f - player_tg;
        if (range > 1e-10f) {
            mwc_live = mwc_cash + (mwc_win_all - mwc_cash) * (p_win - player_tg) / range;
        } else {
            mwc_live = mwc_win_all;
        }
    }

    // Janowski interpolation in MWC space
    return mwc_dead * (1.0f - cube_x) + mwc_live * cube_x;
}

// Compute cubeful MWC for match play — player owns cube.
// Piecewise-linear interpolation with 2 regions:
//   p < player's TG: below cash point (only player can double)
//   p >= player's TG: above cash point (player plays on for gammon)
static float cl2cf_match_owned(
    const std::array<float, NUM_OUTPUTS>& probs,
    const CubeInfo& cube,
    float cube_x)
{
    int away1 = cube.match.away1;
    int away2 = cube.match.away2;
    int cv = cube.cube_value;
    bool craw = cube.match.is_crawford;

    float p_win = probs[0];

    // Gammon/backgammon ratios
    float rG0, rBG0, rG1, rBG1;
    if (p_win > 1e-7f) {
        rG0 = (probs[1] - probs[2]) / p_win;
        rBG0 = probs[2] / p_win;
    } else {
        rG0 = 0.0f;
        rBG0 = 0.0f;
    }
    if (p_win < 1.0f - 1e-7f) {
        rG1 = (probs[3] - probs[4]) / (1.0f - p_win);
        rBG1 = probs[4] / (1.0f - p_win);
    } else {
        rG1 = 0.0f;
        rBG1 = 0.0f;
    }

    // Dead cube MWC (uses MET-weighted outcome values, not linear approximation)
    float mwc_dead = cubeless_mwc(probs, away1, away2, cv, craw);

    // MET lookups at current cube value
    float mwc_win_s  = get_met_after(away1, away2, cv, true, craw);
    float mwc_win_g  = get_met_after(away1, away2, 2*cv, true, craw);
    float mwc_win_b  = get_met_after(away1, away2, 3*cv, true, craw);
    float mwc_lose_s = get_met_after(away1, away2, cv, false, craw);
    float mwc_lose_g = get_met_after(away1, away2, 2*cv, false, craw);
    float mwc_lose_b = get_met_after(away1, away2, 3*cv, false, craw);

    float mwc_cash = mwc_win_s;

    // Weighted MWCs
    float mwc_win_all = (1.0f - rG0 - rBG0) * mwc_win_s
                      + rG0 * mwc_win_g + rBG0 * mwc_win_b;
    float mwc_lose_all = (1.0f - rG1 - rBG1) * mwc_lose_s
                       + rG1 * mwc_lose_g + rBG1 * mwc_lose_b;

    // Player's cash point using recursive algorithm (GNUbg GetPoints)
    float player_cp_val, opp_cp_val;
    get_match_points(away1, away2, cv, craw, rG0, rBG0, rG1, rBG1,
                     player_cp_val, opp_cp_val);
    float player_tg = player_cp_val;

    float mwc_live;
    if (p_win <= player_tg) {
        // Region 1: Below cash point
        // Linear from (0, mwc_lose_all) to (player_tg, mwc_cash)
        if (player_tg > 1e-10f) {
            mwc_live = mwc_lose_all + (mwc_cash - mwc_lose_all) * p_win / player_tg;
        } else {
            mwc_live = mwc_lose_all;
        }
    } else {
        // Region 2: Player too good to double (play on for gammon)
        // Linear from (player_tg, mwc_cash) to (1, mwc_win_all)
        float range = 1.0f - player_tg;
        if (range > 1e-10f) {
            mwc_live = mwc_cash + (mwc_win_all - mwc_cash) * (p_win - player_tg) / range;
        } else {
            mwc_live = mwc_win_all;
        }
    }

    return mwc_dead * (1.0f - cube_x) + mwc_live * cube_x;
}

// Compute cubeful MWC for match play — opponent owns cube (unavailable to player).
// Piecewise-linear interpolation with 2 regions:
//   p < opponent's TG: opponent too good to double
//   p >= opponent's TG: above opponent's cash point
static float cl2cf_match_unavailable(
    const std::array<float, NUM_OUTPUTS>& probs,
    const CubeInfo& cube,
    float cube_x)
{
    int away1 = cube.match.away1;
    int away2 = cube.match.away2;
    int cv = cube.cube_value;
    bool craw = cube.match.is_crawford;

    float p_win = probs[0];

    // Gammon/backgammon ratios
    float rG0, rBG0, rG1, rBG1;
    if (p_win > 1e-7f) {
        rG0 = (probs[1] - probs[2]) / p_win;
        rBG0 = probs[2] / p_win;
    } else {
        rG0 = 0.0f;
        rBG0 = 0.0f;
    }
    if (p_win < 1.0f - 1e-7f) {
        rG1 = (probs[3] - probs[4]) / (1.0f - p_win);
        rBG1 = probs[4] / (1.0f - p_win);
    } else {
        rG1 = 0.0f;
        rBG1 = 0.0f;
    }

    // Dead cube MWC (uses MET-weighted outcome values, not linear approximation)
    float mwc_dead = cubeless_mwc(probs, away1, away2, cv, craw);

    // MET lookups at current cube value
    float mwc_win_s  = get_met_after(away1, away2, cv, true, craw);
    float mwc_win_g  = get_met_after(away1, away2, 2*cv, true, craw);
    float mwc_win_b  = get_met_after(away1, away2, 3*cv, true, craw);
    float mwc_lose_s = get_met_after(away1, away2, cv, false, craw);
    float mwc_lose_g = get_met_after(away1, away2, 2*cv, false, craw);
    float mwc_lose_b = get_met_after(away1, away2, 3*cv, false, craw);

    float mwc_opp_cash = mwc_lose_s;

    // Weighted MWCs
    float mwc_win_all = (1.0f - rG0 - rBG0) * mwc_win_s
                      + rG0 * mwc_win_g + rBG0 * mwc_win_b;
    float mwc_lose_all = (1.0f - rG1 - rBG1) * mwc_lose_s
                       + rG1 * mwc_lose_g + rBG1 * mwc_lose_b;

    // Opponent's cash point using recursive algorithm (GNUbg GetPoints)
    float player_cp_val, opp_cp_val;
    get_match_points(away1, away2, cv, craw, rG0, rBG0, rG1, rBG1,
                     player_cp_val, opp_cp_val);
    float opp_tg = 1.0f - opp_cp_val;

    float mwc_live;
    if (p_win <= opp_tg) {
        // Region 1: Opponent too good to double
        // Linear from (0, mwc_lose_all) to (opp_tg, mwc_opp_cash)
        if (opp_tg > 1e-10f) {
            mwc_live = mwc_lose_all + (mwc_opp_cash - mwc_lose_all) * p_win / opp_tg;
        } else {
            mwc_live = mwc_lose_all;
        }
    } else {
        // Region 2: Above opponent's cash point
        // Linear from (opp_tg, mwc_opp_cash) to (1, mwc_win_all)
        float range = 1.0f - opp_tg;
        if (range > 1e-10f) {
            mwc_live = mwc_opp_cash + (mwc_win_all - mwc_opp_cash) * (p_win - opp_tg) / range;
        } else {
            mwc_live = mwc_win_all;
        }
    }

    return mwc_dead * (1.0f - cube_x) + mwc_live * cube_x;
}

// Post-Crawford automatic double. With one side 1-away after the Crawford
// game, the trailer doubles at the first opportunity: the leader cannot gain
// from the cube, and the trailer can only win on their own turns, all of which
// come after that next pre-roll opportunity. So a cube the trailer may still
// turn (centered or trailer-owned, and not yet worth the trailer's match) is
// valued as doubled once rather than through Janowski: the dead-cube anchor at
// the current value scores a trailer's gammon at 3-away/1-away as 2 points
// instead of the match, which put No Double up to 0.18 equity below
// Double/Take at 2-ply although rollouts (GNUbg's and our own) show them equal.
//
// The trailer may also never double (play on at the current cube), and after
// the double the leader takes (2x cube, then dead -- only the 1-away leader
// could turn it) or passes. Returns true and sets `mwc` (player's view) when
// the rule applies.
static bool post_crawford_auto_double_mwc(
    const std::array<float, NUM_OUTPUTS>& probs,
    const CubeInfo& cube,
    float& mwc)
{
    const MatchInfo& m = cube.match;
    if (cube.is_money() || m.is_crawford) return false;
    const bool player_trails = (m.away2 == 1 && m.away1 > 1);
    const bool opponent_trails = (m.away1 == 1 && m.away2 > 1);
    if (!player_trails && !opponent_trails) return false;

    const int cv = cube.cube_value;
    const int trailer_away = player_trails ? m.away1 : m.away2;
    const CubeOwner trailer_owns =
        player_trails ? CubeOwner::PLAYER : CubeOwner::OPPONENT;
    if (cube.owner != CubeOwner::CENTERED && cube.owner != trailer_owns)
        return false;
    if (trailer_away <= cv) return false;  // any trailer win already takes the match
    if (cube.max_cube_value > 0 && 2 * cv > cube.max_cube_value) return false;

    const float play_on = cubeless_mwc(probs, m.away1, m.away2, cv, false);
    const float take = cubeless_mwc(probs, m.away1, m.away2, 2 * cv, false);
    // The trailer cashes cv points when the leader passes.
    const float pass = get_met_after(m.away1, m.away2, cv, player_trails, false);
    mwc = player_trails ? std::max(play_on, std::min(take, pass))
                        : std::min(play_on, std::max(take, pass));
    return true;
}

float cl2cf_match(const std::array<float, NUM_OUTPUTS>& probs,
                  const CubeInfo& cube, float cube_x)
{
    // Dead cube: return cubeless MWC directly
    if (is_dead_cube(cube)) {
        return cubeless_mwc(probs, cube.match.away1, cube.match.away2,
                            cube.cube_value, cube.match.is_crawford);
    }

    float auto_double_mwc;
    if (post_crawford_auto_double_mwc(probs, cube, auto_double_mwc)) {
        return auto_double_mwc;
    }

    // Dispatch by cube ownership
    switch (cube.owner) {
        case CubeOwner::CENTERED:
            return cl2cf_match_centered(probs, cube, cube_x);
        case CubeOwner::PLAYER:
            return cl2cf_match_owned(probs, cube, cube_x);
        case CubeOwner::OPPONENT:
            return cl2cf_match_unavailable(probs, cube, cube_x);
    }
    return 0.0f;  // unreachable
}

float cl2cf(const std::array<float, NUM_OUTPUTS>& probs,
            const CubeInfo& cube, float cube_x)
{
    // Dead cube: return cubeless equity directly (no Janowski interpolation)
    if (cube_is_dead(cube)) {
        if (cube.is_money()) {
            return cubeless_equity(probs);
        }
        // Match play: return cubeless MWC converted to equity
        float mwc = cubeless_mwc(probs, cube.match.away1, cube.match.away2,
                                  cube.cube_value, cube.match.is_crawford);
        return mwc2eq(mwc, cube.match.away1, cube.match.away2,
                      cube.cube_value, cube.match.is_crawford);
    }

    if (cube.is_money()) {
        return cl2cf_money(probs, cube.owner, cube_x, cube.jacoby_active());
    }
    // Match play: cl2cf_match returns MWC, convert to equity
    float mwc = cl2cf_match(probs, cube, cube_x);
    return mwc2eq(mwc, cube.match.away1, cube.match.away2,
                  cube.cube_value, cube.match.is_crawford);
}

float cube_efficiency(
    const std::array<float, NUM_OUTPUTS>& /*probs*/,
    bool is_race_pos,
    int player_pips,
    int opponent_pips)
{
    // Current implementation ignores probs; reserved for a future ML formula
    // trained on (probs, is_race, pips) -> implied Janowski cube life index.
    if (!is_race_pos) {
        return 0.68f;  // Contact/crashed
    }
    // Race: linear in average pip count (both players), clamped to [0.6, 0.7].
    // Using the average instead of just the roller's pips makes the Janowski
    // cubeful conversion perspective-independent, eliminating a systematic
    // bias at odd ply levels in N-ply match play evaluation.
    float avg_pips = static_cast<float>(player_pips + opponent_pips) * 0.5f;
    float x = 0.55f + 0.00125f * avg_pips;
    return std::clamp(x, 0.6f, 0.7f);
}

float contact_cube_x(const CubeInfo& cube, int player_pips, int opponent_pips)
{
    if (cube.contact_x_pip_slope == 0.0f) return cube.contact_x;
    float avg_pips = static_cast<float>(player_pips + opponent_pips) * 0.5f;
    float x = cube.contact_x
              + cube.contact_x_pip_slope * (avg_pips - CONTACT_X_PIP_PIVOT);
    return std::clamp(x, 0.5f, 1.0f);
}

float resolve_cube_x(
    const std::array<float, NUM_OUTPUTS>& probs,
    const CubeInfo& cube,
    bool is_race_pos,
    int player_pips,
    int opponent_pips)
{
    if (cube.cube_x_override >= 0.0f) return cube.cube_x_override;
    if (!is_race_pos && cube.contact_x >= 0.0f)
        return contact_cube_x(cube, player_pips, opponent_pips);
    return cube_efficiency(probs, is_race_pos, player_pips, opponent_pips);
}

// Money game cube decision (1-ply / raw NN).
static CubeDecision cube_decision_1ply_money(
    const std::array<float, NUM_OUTPUTS>& probs,
    const CubeInfo& cube,
    float cube_x)
{
    CubeDecision result;
    result.equity_dp = 1.0f;  // Double/Pass: always +1.0 for money games

    // When cube is dead (max_cube_value reached), no cube actions possible.
    // ND = cubeless equity, DT/DP are degenerate, should_double = false.
    if (cube_is_dead(cube)) {
        float cl_eq = cubeless_equity(probs);
        result.equity_nd = cl_eq;
        result.equity_dt = cl_eq;  // Degenerate — can't double
        result.equity_dp = 1.0f;
        result.should_double = false;
        result.should_take = true;
        result.optimal_equity = cl_eq;
        result.is_beaver = false;
        return result;
    }

    // No Double equity: cubeful equity with current cube state.
    // With Jacoby and centered cube, gammons are zeroed (W=1, L=1).
    result.equity_nd = cl2cf_money(probs, cube.owner, cube_x, cube.jacoby_active());

    // Double/Take equity: cubeful equity if cube is doubled and opponent takes.
    // After doubling, the opponent owns the cube at 2x the current value.
    // Jacoby is NOT active here — the cube has been turned, gammons count.
    float actual_dt = 2.0f * cl2cf_money(probs, CubeOwner::OPPONENT, cube_x, false);

    // Beaver check: if beavers are allowed and DT < 0, the opponent beavers.
    // DB = 2 * DT (beaver doubles the cube value, opponent retains ownership).
    // This is exact for money games because cl2cf_money is cube-value-independent.
    if (cube.beaver && actual_dt < 0.0f) {
        result.equity_dt = 2.0f * actual_dt;  // DB equity
        result.is_beaver = true;
    } else {
        result.equity_dt = actual_dt;
        result.is_beaver = false;
    }

    // A player who cannot double (the opponent owns the cube) has no
    // decision: report the equities and No Double, exactly as the match path
    // and every N-ply path do. (Until 2026-09-06 this path had no gate and
    // "doubled" whenever DT beat ND.)
    if (!can_double(cube)) {
        result.should_double = false;
        result.should_take = true;
        result.optimal_equity = result.equity_nd;
        return result;
    }

    // Decision logic
    // If we double, the opponent picks the response that gives us LESS equity.
    // With beaver: opponent already chose min(DT, DB, DP) — result.equity_dt
    // reflects the effective response (DB if beaver, DT otherwise).
    float best_double = std::min(result.equity_dt, result.equity_dp);
    result.should_double = (best_double > result.equity_nd);

    // Should opponent take? Take if effective DT < DP (beaver counts as take)
    result.should_take = (result.equity_dt <= result.equity_dp);

    // Optimal equity after both sides play optimally
    if (result.should_double) {
        result.optimal_equity = std::min(result.equity_dt, result.equity_dp);
    } else {
        result.optimal_equity = result.equity_nd;
    }

    return result;
}

// Match play cube decision (1-ply / raw NN).
// Computes ND/DT/DP in MWC space, then converts to equity at original cube value.
static CubeDecision cube_decision_1ply_match(
    const std::array<float, NUM_OUTPUTS>& probs,
    const CubeInfo& cube,
    float cube_x)
{
    int away1 = cube.match.away1;
    int away2 = cube.match.away2;
    int cv = cube.cube_value;
    bool craw = cube.match.is_crawford;

    CubeDecision result;

    // When cube is dead via max_cube_value, return cubeless MWC as equity.
    if (cube_is_dead(cube)) {
        float mwc = cubeless_mwc(probs, away1, away2, cv, craw);
        float eq = mwc2eq(mwc, away1, away2, cv, craw);
        result.equity_nd = eq;
        result.equity_dt = eq;
        result.equity_dp = mwc2eq(dp_mwc(away1, away2, cv, craw), away1, away2, cv, craw);
        result.should_double = false;
        result.should_take = true;
        result.optimal_equity = eq;
        result.is_beaver = false;
        return result;
    }

    // DP MWC: opponent passes, player wins cv points
    float dp_m = dp_mwc(away1, away2, cv, craw);

    // ND MWC: cubeful MWC with current cube state
    float nd_m = cl2cf_match(probs, cube, cube_x);

    // DT MWC: cube is doubled and opponent takes.
    // After doubling: cube_value = 2*cv, opponent owns the cube.
    CubeInfo dt_cube;
    dt_cube.cube_value = 2 * cv;
    dt_cube.owner = CubeOwner::OPPONENT;
    dt_cube.match = cube.match;
    float dt_m = cl2cf_match(probs, dt_cube, cube_x);

    // Convert MWC values to equity for display.
    // All three use the SAME normalization (original cube value) so they're
    // directly comparable — matching GNUbg's FindCubeDecision convention.
    result.equity_nd = mwc2eq(nd_m, away1, away2, cv, craw);
    result.equity_dt = mwc2eq(dt_m, away1, away2, cv, craw);
    result.equity_dp = mwc2eq(dp_m, away1, away2, cv, craw);

    // Decision logic in MWC space for correctness (equities are all on the
    // same scale now, but MWC decisions are the canonical approach for match play).
    bool player_can_double = can_double(cube);

    // Post-Crawford automatic double: the trailing player (away1 > 1,
    // opponent at 1-away) should always double — losing costs the match
    // regardless of cube value, but winning scores more points.
    bool auto_double = (!craw && player_can_double && away1 > 1 && away2 == 1);

    if (!player_can_double) {
        // Player cannot double (Crawford, dead cube, etc.)
        result.should_double = false;
        result.should_take = true;  // Irrelevant since no double
        result.optimal_equity = result.equity_nd;
    } else if (auto_double) {
        result.should_double = true;
        result.should_take = (dt_m <= dp_m);
        result.optimal_equity = std::min(result.equity_dt, result.equity_dp);
    } else {
        float best_double_mwc = std::min(dt_m, dp_m);
        result.should_double = (best_double_mwc > nd_m);
        result.should_take = (dt_m <= dp_m);

        if (result.should_double) {
            result.optimal_equity = std::min(result.equity_dt, result.equity_dp);
        } else {
            result.optimal_equity = result.equity_nd;
        }
    }

    return result;
}

CubeDecision cube_decision_1ply(
    const std::array<float, NUM_OUTPUTS>& probs,
    const CubeInfo& cube,
    float cube_x)
{
    if (cube.is_money()) {
        return cube_decision_1ply_money(probs, cube, cube_x);
    }
    return cube_decision_1ply_match(probs, cube, cube_x);
}

CubeDecision cube_decision_1ply(
    const std::array<float, NUM_OUTPUTS>& probs,
    const CubeInfo& cube,
    const Board& board,
    bool is_race_pos)
{
    auto [pp, op] = pip_counts(board);
    return cube_decision_1ply(probs, cube,
                              resolve_cube_x(probs, cube, is_race_pos, pp, op));
}

// ---------------------------------------------------------------------------
// N-ply cubeful evaluation
// ---------------------------------------------------------------------------

float cubeful_equity_nply(
    const Board& board,
    CubeOwner owner,
    const Strategy& strategy,
    int n_plies,
    const MoveFilter& filter,
    int n_threads,
    const Strategy* move_filter)
{
    CubeInfo cube;
    cube.cube_value = 1;
    cube.owner = owner;
    float out = 0.0f;
    cubeful_equity_nply_multi(board, &cube, 1, strategy, n_plies, &out,
                              filter, n_threads, move_filter);
    return out;
}

// Match play overload: returns cubeful equity (NOT MWC).
float cubeful_equity_nply(
    const Board& board,
    const CubeInfo& cube,
    const Strategy& strategy,
    int n_plies,
    const MoveFilter& filter,
    int n_threads,
    const Strategy* move_filter)
{
    float out = 0.0f;
    cubeful_equity_nply_multi(board, &cube, 1, strategy, n_plies, &out,
                              filter, n_threads, move_filter);
    return out;
}

// Returns post-roll probabilities from the N-ply CUBE-AWARE tree, in the
// player-on-roll's perspective at `board`. Same traversal as
// cubeful_equity_nply (the probs are accumulated through the tree), so the
// returned probs reflect cube/match-aware interior move picks throughout.
//
// Used by `post_move_analytics` when the inner analyzer is N-ply and the
// caller wants probs that depend on cube state (e.g. for a 1-away match score).
std::array<float, NUM_OUTPUTS> cubeful_probs_nply(
    const Board& board,
    const CubeInfo& cube,
    const Strategy& strategy,
    int n_plies,
    const MoveFilter& filter,
    int n_threads,
    const Strategy* move_filter)
{
    float out = 0.0f;
    std::array<float, NUM_OUTPUTS> probs{};
    cubeful_equity_nply_multi(board, &cube, 1, strategy, n_plies, &out,
                              filter, n_threads, move_filter, false, &probs);
    return probs;
}

// Returns BOTH the cube-aware probs and the cubeful equity from a single
// traversal of the cubeful tree. Avoids a second tree walk when callers want
// both (per-candidate checker play analytics).
CubefulProbsAndEquity cubeful_probs_and_equity_nply(
    const Board& board,
    const CubeInfo& cube,
    const Strategy& strategy,
    int n_plies,
    const MoveFilter& filter,
    int n_threads,
    const Strategy* move_filter,
    const Board* root_board)
{
    float out = 0.0f;
    std::array<float, NUM_OUTPUTS> probs{};
    cubeful_equity_nply_multi(board, &cube, 1, strategy, n_plies, &out,
                              filter, n_threads, move_filter, false, &probs,
                              /*deep_prefilter=*/false, root_board);
    return {probs, out};
}


CubeDecision cube_decision_nply(
    const Board& board,
    const CubeInfo& cube,
    const Strategy& strategy,
    int n_plies,
    const MoveFilter& filter,
    int n_threads,
    const Strategy* move_filter)
{
    // The multi entry with n=1 applies the identical 1-ply path, money/match
    // unpack and post-Crawford auto-double logic.
    CubeDecision out{};
    cube_decision_nply_multi(board, &cube, 1, strategy, n_plies, &out,
                             filter, n_threads, move_filter);
    return out;
}

} // namespace bgbot
