#pragma once

#include <cstddef>

constexpr size_t GRU_INPUT_SIZE = 5;
constexpr size_t GRU_HIDDEN_SIZE = 32;
constexpr size_t GRU_OUTPUT_SIZE = 1;

static const float GRU_WIR[GRU_HIDDEN_SIZE * GRU_INPUT_SIZE] = {0.0f};
static const float GRU_WIZ[GRU_HIDDEN_SIZE * GRU_INPUT_SIZE] = {0.0f};
static const float GRU_WIN[GRU_HIDDEN_SIZE * GRU_INPUT_SIZE] = {0.0f};
static const float GRU_BIR[GRU_HIDDEN_SIZE] = {0.0f};
static const float GRU_BIZ[GRU_HIDDEN_SIZE] = {0.0f};
static const float GRU_BIN[GRU_HIDDEN_SIZE] = {0.0f};

static const float GRU_WHR[GRU_HIDDEN_SIZE * GRU_HIDDEN_SIZE] = {0.0f};
static const float GRU_WHZ[GRU_HIDDEN_SIZE * GRU_HIDDEN_SIZE] = {0.0f};
static const float GRU_WHN[GRU_HIDDEN_SIZE * GRU_HIDDEN_SIZE] = {0.0f};
static const float GRU_BHR[GRU_HIDDEN_SIZE] = {0.0f};
static const float GRU_BHZ[GRU_HIDDEN_SIZE] = {0.0f};
static const float GRU_BHN[GRU_HIDDEN_SIZE] = {0.0f};

static const float GRU_WO[GRU_OUTPUT_SIZE * GRU_HIDDEN_SIZE] = {0.0f};
static const float GRU_BO[GRU_OUTPUT_SIZE] = {0.0f};
