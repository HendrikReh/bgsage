// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Mark Higgins
#pragma once

// Portable, exception-safe C boundary for native clients. No Python, platform
// framework, network, or host-application dependency. Calls on a handle must be
// serialized by its owner. Returned JSON is owned by the caller; free it below.
#ifdef __cplusplus
extern "C" {
#endif

typedef struct NeongammonMobileEngine NeongammonMobileEngine;
typedef struct {
    int board[26];
    int die1, die2, ply;
    int cube_value, cube_owner; // 0 centered, 1 player, 2 opponent
    int away1, away2, is_crawford, jacoby, beaver;
    int budget_ms; // checked between candidates; never creates engine threads
} NeongammonMobileRequest;

NeongammonMobileEngine* neongammon_mobile_create(const char* strategy_type,
    const char* const* paths, const int* hidden_sizes, int count,
    const char* bearoff_path);
void neongammon_mobile_destroy(NeongammonMobileEngine* engine);
// operation: 0 checker; 1 cube. Null means invalid, over budget, or failure;
// native caller falls back to the cloud. Does not produce a partial result.
char* neongammon_mobile_analyze(NeongammonMobileEngine* engine, int operation,
    const NeongammonMobileRequest* request);
void neongammon_mobile_free(char* text);
const char* neongammon_mobile_last_error(void);

#ifdef __cplusplus
}
#endif
