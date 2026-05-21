#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include "EvictionManager.hpp"
#include "FaultHandler.hpp"
#include "MemRegion.hpp"
#include "MemoryManager.hpp"
#include "runtime_gru.hpp"

namespace py = pybind11;

PYBIND11_MODULE(swapcore_native, m){
    py::enum_<EvictionPolicy>(m, "EvictionPolicy")
        .value("LRU", EvictionPolicy::LRU)
        .value("CLOCK", EvictionPolicy::CLOCK);

    py::class_<MemoryManager>(m, "MemoryManager")
        .def(py::init<>())
        .def(py::init<EvictionPolicy, bool>())
        .def("initPageTable", &MemoryManager::initPageTable)
        .def("allocFrame", &MemoryManager::allocFrame)
        .def("loadPage", &MemoryManager::loadPage)
        .def("faultLatencyNs", &MemoryManager::faultLatencyNs)
        .def("swapWriteLatencyNs", &MemoryManager::swapWriteLatencyNs)
        .def("swapReadLatencyNs", &MemoryManager::swapReadLatencyNs)
        .def("learnedEvictionCount", &MemoryManager::learnedEvictionCount)
        .def("learnedEvictionActive", &MemoryManager::learnedEvictionActive);

    py::class_<FaultHandler>(m, "FaultHandler")
        .def(py::init<MemoryManager*>())
        .def("start", &FaultHandler::start)
        .def("stop", &FaultHandler::stop);

    py::class_<MemRegion>(m, "MemRegion")
        .def(py::init<size_t, FaultHandler*>())
        .def("get", &MemRegion::get)
        .def("set", &MemRegion::set)
        .def("getSize", &MemRegion::getSize);

    py::class_<RuntimeGRU>(m, "RuntimeGRU")
        .def(py::init<>())
        .def("predict", &RuntimeGRU::predict)
        .def("predictSequence", py::overload_cast<const std::vector<float>&, size_t>(&RuntimeGRU::predictSequence, py::const_))
        .def("predictLogit", [](const RuntimeGRU& gru, const std::vector<float>& sequence, size_t seqLen){
            if (sequence.size() != seqLen * RuntimeGRU::INPUT_SIZE){
                throw std::invalid_argument("sequence size must equal seqLen * input size");
            }
            return gru.predictLogit(sequence.data(), seqLen);
        });
}
