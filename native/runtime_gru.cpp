#include "runtime_gru.hpp"
#include "runtime_gru_params.hpp"
#include <cmath>
#include <stdexcept>

static inline float sigmoid(float x){
    return 1.0f / (1.0f + std::exp(-x));
}

static inline float dotProduct(const float* a, const float* b, size_t length){
    float result = 0.0f;
    for (size_t i = 0; i < length; ++i){
        result += a[i] * b[i];
    }
    return result;
}

static inline void matVec(const float* matrix, const float* vector, float* output, size_t rows, size_t cols){
    for (size_t r = 0; r < rows; ++r){
        const float* rowBase = matrix + r * cols;
        output[r] = dotProduct(rowBase, vector, cols);
    }
}

float RuntimeGRU::predictLogit(const float* sequence, size_t seqLen) const{
    float hidden[HIDDEN_SIZE] = {0.0f};

    for (size_t t = 0; t < seqLen; ++t){
        const float* input = sequence + t * INPUT_SIZE;
        float hiddenPrev[HIDDEN_SIZE];
        for (size_t i = 0; i < HIDDEN_SIZE; ++i){
            hiddenPrev[i] = hidden[i];
        }

        float inputReset[HIDDEN_SIZE];
        float inputUpdate[HIDDEN_SIZE];
        float inputNew[HIDDEN_SIZE];
        float recurrentReset[HIDDEN_SIZE];
        float recurrentUpdate[HIDDEN_SIZE];

        matVec(GRU_WIR, input, inputReset, HIDDEN_SIZE, INPUT_SIZE);
        matVec(GRU_WIZ, input, inputUpdate, HIDDEN_SIZE, INPUT_SIZE);
        matVec(GRU_WIN, input, inputNew, HIDDEN_SIZE, INPUT_SIZE);
        matVec(GRU_WHR, hiddenPrev, recurrentReset, HIDDEN_SIZE, HIDDEN_SIZE);
        matVec(GRU_WHZ, hiddenPrev, recurrentUpdate, HIDDEN_SIZE, HIDDEN_SIZE);

        float resetGate[HIDDEN_SIZE];
        float updateGate[HIDDEN_SIZE];
        float resetState[HIDDEN_SIZE];
        for (size_t i = 0; i < HIDDEN_SIZE; ++i){
            resetGate[i] = sigmoid(inputReset[i] + recurrentReset[i] + GRU_BIR[i] + GRU_BIR_HIDDEN[i]);
            updateGate[i] = sigmoid(inputUpdate[i] + recurrentUpdate[i] + GRU_BIZ[i] + GRU_BIZ_HIDDEN[i]);
            resetState[i] = resetGate[i] * hiddenPrev[i];
        }

        float recurrentNew[HIDDEN_SIZE];
        matVec(GRU_WHN, resetState, recurrentNew, HIDDEN_SIZE, HIDDEN_SIZE);

        for (size_t i = 0; i < HIDDEN_SIZE; ++i){
            float candidateInput = inputNew[i] + recurrentNew[i] + GRU_BIN[i] + GRU_BHN[i];
            float candidate = std::tanh(candidateInput);
            hidden[i] = (1.0f - updateGate[i]) * candidate + updateGate[i] * hiddenPrev[i];
        }
    }

    return dotProduct(GRU_WO, hidden, HIDDEN_SIZE) + GRU_BO[0];
}

float RuntimeGRU::predictSequence(const float* sequence, size_t seqLen) const{
    return sigmoid(predictLogit(sequence, seqLen));
}

float RuntimeGRU::predictSequence(const std::vector<float>& sequence, size_t seqLen) const{
    if (sequence.size() != seqLen * INPUT_SIZE){
        throw std::invalid_argument("RuntimeGRU sequence size must equal seqLen * INPUT_SIZE");
    }
    return predictSequence(sequence.data(), seqLen);
}

float RuntimeGRU::predict(float vpn, float accessDelta, float accessType, float timestamp, float reuseDistance) const{
    float input[INPUT_SIZE] = {vpn, accessDelta, accessType, timestamp, reuseDistance};
    return predictSequence(input, 1);
}
