#include "MemoryManager.hpp"
#include <iostream>
#include <cassert>
#include <cstring>
#include <limits>

MemoryManager::MemoryManager(EvictionPolicy policy, bool learned)
    : frames(NUM_FRAMES * PAGE_SIZE, 0),
      swapMgr(1024, "swapfile.bin"),
      eviction(policy),
      learnedPredictor(),
      learnedEvictionEnabled(learned),
      learnedRecencyWeight(0.45f),
      learnedFrequencyWeight(0.25f),
      learnedPredictionWeight(0.30f),
      accessCounter(0),
      learnedEvictions(0),
      faultNs(0),
      swapWriteNs(0),
      swapReadNs(0){
    for (u32 i = 0; i < NUM_FRAMES; ++i){
        freeFrames.push(i);
    }
    std::cout << "MemoryManager initialized with " << NUM_FRAMES << " frames" << std::endl;
}

MemoryManager::~MemoryManager(){
    std::cout << "MemoryManager destroyed" << std::endl;
}

void MemoryManager::initPageTable(size_t num_pages){
    pageTbl_.resize(num_pages);
    std::cout << "Page table initialized for " << num_pages << " pages" << std::endl;
}

u32 MemoryManager::allocFrame(u64 vpn){
    if (!freeFrames.empty()){
        u32 idx = freeFrames.front();
        freeFrames.pop();
        std::cout << "Allocated frame " << idx << " for VPN " << vpn << std::endl;
        return idx;
    }

    u64 victim_vpn = learnedEvictionEnabled ? chooseLearnedVictim() : eviction.choose_victim(pageTbl_);
    auto& victim_entry = pageTbl_.getEntry(victim_vpn);
    assert(victim_entry.resident);
    u32 frameIndex = victim_entry.frameIndex;
    std::cout << "Evicting VPN " << victim_vpn << " from frame " << frameIndex << std::endl;
    if (victim_entry.dirty){
        if (victim_entry.swapSlot == SWAP_SLOT_INVALID){
            victim_entry.swapSlot = swapMgr.allocSlot();
        }
        auto start = std::chrono::high_resolution_clock::now();
        swapMgr.writePage(victim_entry.swapSlot, frameData(frameIndex));
        auto end = std::chrono::high_resolution_clock::now();
        swapWriteNs += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        std::cout << "Wrote dirty VPN " << victim_vpn << " to swap slot " << victim_entry.swapSlot << "\n";
    }
    victim_entry.resident = false;
    victim_entry.frameIndex = SWAP_SLOT_INVALID;
    victim_entry.dirty = false;
    victim_entry.reference = false;
    return frameIndex;
}

void MemoryManager::freeFrame(u32 frameIndex){
    freeFrames.push(frameIndex);
    std::cout << "Freed frame " << frameIndex << std::endl;
}

void* MemoryManager::frameData(u32 frameIndex){
    assert(frameIndex < NUM_FRAMES);
    return frames.data() + frameIndex * PAGE_SIZE;
}

PageTable& MemoryManager::pageTbl(){
    return pageTbl_;
}

void MemoryManager::touchPage(u64 vpn){
    assert(vpn < pageTbl_.size());
    auto& entry = pageTbl_.getEntry(vpn);
    entry.previousAccess = entry.lastAccess;
    entry.lastAccess = ++accessCounter;
    entry.frequency += 1;
    std::array<float, RuntimeGRU::INPUT_SIZE> features = {
        float(vpn),
        float(entry.lastAccess - entry.previousAccess),
        entry.dirty ? 1.0f : 0.0f,
        float(entry.lastAccess),
        0.0f
    };
    gruHistory.push_back(features);
    while (gruHistory.size() > 32){
        gruHistory.pop_front();
    }

    std::vector<float> sequence(32 * RuntimeGRU::INPUT_SIZE, 0.0f);
    size_t offset = 32 - gruHistory.size();
    for (size_t i = 0; i < gruHistory.size(); ++i){
        for (size_t j = 0; j < RuntimeGRU::INPUT_SIZE; ++j){
            sequence[(offset + i) * RuntimeGRU::INPUT_SIZE + j] = gruHistory[i][j];
        }
    }
    entry.predictedReuse = learnedPredictor.predictSequence(sequence, 32);
    eviction.touch(vpn, entry, accessCounter);
}

void MemoryManager::loadPage(u64 vpn, u32 frameIndex){
    auto& entry = pageTbl_.getEntry(vpn);
    if (entry.swapSlot != SWAP_SLOT_INVALID){
        auto start = std::chrono::high_resolution_clock::now();
        swapMgr.readPage(entry.swapSlot, frameData(frameIndex));
        auto end = std::chrono::high_resolution_clock::now();
        swapReadNs += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        std::cout << "Read VPN " << vpn << " from swap slot " << entry.swapSlot << "\n";
    } else{
        std::memset(frameData(frameIndex), 0, PAGE_SIZE);
    }
    entry.resident = true;
    entry.frameIndex = frameIndex;
    entry.dirty = false;
    touchPage(vpn);
}

void MemoryManager::addFaultLatencyNs(uint64_t ns){
    faultNs += ns;
}

uint64_t MemoryManager::faultLatencyNs() const {
    return faultNs;
}

float MemoryManager::computeLearnedScore(const PageMeta& entry, u64 vpn) const{
    float recency = float(accessCounter - entry.lastAccess);
    float frequency = float(entry.frequency);
    float mlScore = entry.predictedReuse;
    float score = learnedRecencyWeight * recency;
    score += learnedFrequencyWeight * (1.0f / (1.0f + frequency));
    score += learnedPredictionWeight * (1.0f - mlScore);
    return score;
}

u64 MemoryManager::chooseLearnedVictim(){
    u64 victim_vpn = SWAP_SLOT_INVALID;
    float best_score = -std::numeric_limits<float>::infinity();
    size_t num_pages = pageTbl_.size();
    for (u64 vpn = 0; vpn < num_pages; ++vpn){
        const auto& entry = pageTbl_.getEntry(vpn);
        if (!entry.resident){
            continue;
        }
        float score = computeLearnedScore(entry, vpn);
        if (score > best_score){
            best_score = score;
            victim_vpn = vpn;
        }
    }
    if (victim_vpn == SWAP_SLOT_INVALID){
        return eviction.choose_victim(pageTbl_);
    }
    learnedEvictions += 1;
    return victim_vpn;
}

uint64_t MemoryManager::swapWriteLatencyNs() const {
    return swapWriteNs;
}

uint64_t MemoryManager::swapReadLatencyNs() const {
    return swapReadNs;
}

uint64_t MemoryManager::learnedEvictionCount() const {
    return learnedEvictions;
}

bool MemoryManager::learnedEvictionActive() const {
    return learnedEvictionEnabled;
}
