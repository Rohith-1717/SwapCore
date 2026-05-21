#pragma once

#include "Common.hpp"
#include "PageTable.hpp"
#include "SwapManager.hpp"
#include "EvictionManager.hpp"
#include "runtime_gru.hpp"
#include <array>
#include <deque>
#include <vector>
#include <queue>
#include <chrono>

class MemoryManager{
public:
    explicit MemoryManager(EvictionPolicy policy = EvictionPolicy::LRU, bool learned = true);
    ~MemoryManager();

    void initPageTable(size_t num_pages);
    u32 allocFrame(u64 vpn);
    void freeFrame(u32 frameIndex);
    void* frameData(u32 frameIndex);
    PageTable& pageTbl();
    void touchPage(u64 vpn);
    void loadPage(u64 vpn, u32 frameIndex);
    void addFaultLatencyNs(uint64_t ns);
    uint64_t faultLatencyNs() const;
    uint64_t swapWriteLatencyNs() const;
    uint64_t swapReadLatencyNs() const;
    uint64_t learnedEvictionCount() const;
    bool learnedEvictionActive() const;

private:
    u64 chooseLearnedVictim();
    float computeLearnedScore(const PageMeta& entry, u64 vpn) const;
    PageTable pageTbl_;
    std::vector<u8> frames;
    std::queue<u32> freeFrames;
    SwapManager swapMgr;
    EvictionManager eviction;
    RuntimeGRU learnedPredictor;
    bool learnedEvictionEnabled;
    float learnedRecencyWeight;
    float learnedFrequencyWeight;
    float learnedPredictionWeight;
    u64 accessCounter;
    uint64_t learnedEvictions;
    uint64_t faultNs;
    uint64_t swapWriteNs;
    uint64_t swapReadNs;
    std::deque<std::array<float, RuntimeGRU::INPUT_SIZE>> gruHistory;
};
