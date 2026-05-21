#pragma once

#include <cstddef>
#include <vector>

class RuntimeGRU{
public:
    static constexpr size_t INPUT_SIZE = 5;
    static constexpr size_t HIDDEN_SIZE = 32;
    static constexpr size_t OUTPUT_SIZE = 1;

    RuntimeGRU() = default;
    float predict(float vpn, float accessDelta, float accessType, float timestamp, float reuseDistance) const;
    float predictLogit(const float* sequence, size_t seqLen) const;
    float predictSequence(const float* sequence, size_t seqLen) const;
    float predictSequence(const std::vector<float>& sequence, size_t seqLen) const;
};
