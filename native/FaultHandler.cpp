#include "FaultHandler.hpp"
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <cstring>
#include <cassert>
#include <iostream>
#include <poll.h>
#include <linux/userfaultfd.h>
#include <chrono>

FaultHandler::FaultHandler(MemoryManager* mgr)
    : uffd(-1)
    , running(false)
    , zeroPage(nullptr)
    , manager(mgr)
    , regionBase(nullptr)
    , regionSize(0){
    uffd = static_cast<int>(syscall(__NR_userfaultfd, 0));
    if (uffd == -1){
        std::cerr << "Failed to create userfaultfd" << std::endl;
        exit(1);
    }

    struct uffdio_api api;
    api.api = UFFD_API;
    api.features = 0;
    if (ioctl(uffd, UFFDIO_API, &api) == -1){
        std::cerr << "UFFDIO_API failed" << std::endl;
        exit(1);
    }

    zeroPage = mmap(nullptr, PAGE_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (zeroPage == MAP_FAILED){
        std::cerr << "Failed to allocate zero page" << std::endl;
        exit(1);
    }
    memset(zeroPage, 0, PAGE_SIZE);
}

FaultHandler::~FaultHandler(){
    if (running){
        stop();
    }
    if (zeroPage){
        munmap(zeroPage, PAGE_SIZE);
    }
    if (uffd != -1){
        close(uffd);
    }
}

void FaultHandler::registerRegion(void* addr, size_t size){
    regionBase = addr;
    regionSize = size;
    size_t pageCount = size / PAGE_SIZE;
    manager->initPageTable(pageCount);

    struct uffdio_register reg;
    reg.range.start = (unsigned long)addr;
    reg.range.len = size;
    reg.mode = UFFDIO_REGISTER_MODE_MISSING;
    if (ioctl(uffd, UFFDIO_REGISTER, &reg) == -1){
        std::cerr << "UFFDIO_REGISTER failed" << std::endl;
        exit(1);
    }
    regions.push_back(reg);
}

void FaultHandler::start(){
    running = true;
    handlerThread = std::thread(&FaultHandler::handleFaults, this);
}

void FaultHandler::stop(){
    running = false;
    if (handlerThread.joinable()){
        handlerThread.join();
    }
}

MemoryManager* FaultHandler::getManager(){
    return manager;
}

void FaultHandler::handleFaults(){
    while (running){
        struct pollfd pfd = {uffd, POLLIN, 0};
        int ret = poll(&pfd, 1, 1000);
        if (ret == -1){
            std::cerr << "Poll failed" << std::endl;
            break;
        }
        if (ret == 0)
            continue;
        if (pfd.revents & POLLIN){
            struct uffd_msg msg;
            ssize_t nread = read(uffd, &msg, sizeof(msg));
            if (nread == 0){
                std::cerr << "EOF on userfaultfd" << std::endl;
                break;
            }
            if (nread == -1){
                std::cerr << "Read failed" << std::endl;
                break;
            }
            if (msg.event != UFFD_EVENT_PAGEFAULT){
                std::cerr << "Unexpected event" << std::endl;
                continue;
            }

            void* faultAddr = (void*)msg.arg.pagefault.address;
            void* pageAddr = (void*)((unsigned long)faultAddr & ~(PAGE_SIZE - 1));
            u64 vpn = ((u64)faultAddr - (u64)regionBase) / PAGE_SIZE;
            assert(vpn < regionSize / PAGE_SIZE);

            auto& entry = manager->pageTbl().getEntry(vpn);
            auto faultStart = std::chrono::high_resolution_clock::now();
            if (!entry.resident){
                u32 frame = manager->allocFrame(vpn);
                manager->loadPage(vpn, frame);
                struct uffdio_copy copy;
                copy.src = (unsigned long)manager->frameData(frame);
                copy.dst = (unsigned long)pageAddr;
                copy.len = PAGE_SIZE;
                copy.mode = 0;
                copy.copy = 0;
                if (ioctl(uffd, UFFDIO_COPY, &copy) == -1){
                    std::cerr << "UFFDIO_COPY failed" << std::endl;
                    break;
                }
                std::cout << "Fault at VPN " << vpn << ", loaded frame " << frame << "\n";
            } else{
                manager->touchPage(vpn);
                struct uffdio_copy copy;
                copy.src = (unsigned long)manager->frameData(entry.frameIndex);
                copy.dst = (unsigned long)pageAddr;
                copy.len = PAGE_SIZE;
                copy.mode = 0;
                copy.copy = 0;
                if (ioctl(uffd, UFFDIO_COPY, &copy) == -1){
                    std::cerr << "UFFDIO_COPY failed" << std::endl;
                    break;
                }
                std::cout << "Re-fault for resident VPN " << vpn << "\n";
            }

            auto faultEnd = std::chrono::high_resolution_clock::now();
            manager->addFaultLatencyNs(std::chrono::duration_cast<std::chrono::nanoseconds>(faultEnd - faultStart).count());
        }
    }
}
