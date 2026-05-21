#include "MemRegion.hpp"
#include <sys/mman.h>
#include <cassert>
#include <iostream>

MemRegion::MemRegion(size_t size, FaultHandler* handler) : size(size), handler(handler), manager(handler->getManager()){
    addr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (addr == MAP_FAILED){
        std::cerr << "Failed to mmap region" << std::endl;
        exit(1);
    }
    handler->registerRegion(addr, size);
}

MemRegion::~MemRegion(){
    if (addr){
        munmap(addr, size);
    }
}

u8 MemRegion::get(size_t offset){
    assert(offset < size);
    u64 vpn = offset / PAGE_SIZE;
    if (manager){
        manager->touchPage(vpn);
    }
    return *(u8*)((char*)addr + offset);
}

void MemRegion::set(size_t offset, u8 value){
    assert(offset < size);
    *(u8*)((char*)addr + offset) = value;
    u64 vpn = offset / PAGE_SIZE;
    if (manager){
        auto& entry = manager->pageTbl().getEntry(vpn);
        entry.dirty = true;
    }
}
