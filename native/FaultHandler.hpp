#pragma once

#include "Common.hpp"
#include "MemoryManager.hpp"
#include <vector>
#include <thread>
#include <linux/userfaultfd.h>

class FaultHandler{
public:
    FaultHandler(MemoryManager* manager);
    ~FaultHandler();
    void registerRegion(void* addr, size_t size);
    void start();
    void stop();
    MemoryManager* getManager();

private:
    int uffd;
    std::thread handlerThread;
    std::vector<struct uffdio_register> regions;
    bool running;
    void* zeroPage;
    MemoryManager* manager;
    void* regionBase;
    size_t regionSize;
    void handleFaults();
};
