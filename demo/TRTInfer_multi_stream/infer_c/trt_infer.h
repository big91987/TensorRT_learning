/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef INFER_C_TRT_INFER_H
#define INFER_C_TRT_INFER_H

#include "common.h"
// #include "buffers.h"
#include "logging.h"
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <fstream>
#include <numeric>
#include <string.h>
#include <vector>
#include <chrono>
#include <thread>
#include "sampleDevice.h"
#include "sampleUtils.h"
// #include "sampleReporting.h"
#include <array>
#include <cuda_profiler_api.h>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <utility>

using namespace nvinfer1;
using namespace sample;

#if defined(__QNX__)
using TimePoint = double;
#else
using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
#endif

template <typename T>
struct TrtDestroyer
{
    void operator()(T* t)
    {
        t->destroy();
    }
};
template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDestroyer<T>>;

TimePoint getCurrentTime()
{
#if defined(__QNX__)
    uint64_t const currentCycles = ClockCycles();
    uint64_t const cyclesPerSecond = SYSPAGE_ENTRY(qtime)->cycles_per_sec;
    // Return current timestamp in ms.
    return static_cast<TimePoint>(currentCycles) * 1000. / cyclesPerSecond;
#else
    return std::chrono::high_resolution_clock::now();
#endif
}

//!
//! \struct SyncStruct
//! \brief Threads synchronization structure
//!
struct SyncStruct
{
    std::mutex mutex;
    TrtCudaStream mainStream;
    TrtCudaEvent gpuStart{cudaEventBlockingSync};
    TimePoint cpuStart{};
    float sleep{};
};

enum class StreamType : int32_t
{
    kINPUT = 0,
    kCOMPUTE = 1,
    kOUTPUT = 2,
    kNUM = 3
};

enum class EventType : int32_t
{
    kINPUT_S = 0, 
    kINPUT_E = 1,
    kCOMPUTE_S = 2,
    kCOMPUTE_E = 3,
    kOUTPUT_S = 4,
    kOUTPUT_E = 5,
    kNUM = 6
};

using EnqueueFunction = std::function<bool(TrtCudaStream&)>;
using MultiStream = std::array<TrtCudaStream, static_cast<int32_t>(StreamType::kNUM)>;
using MultiEvent = std::array<std::unique_ptr<TrtCudaEvent>, static_cast<int32_t>(EventType::kNUM)>;
using EnqueueTimes = std::array<TimePoint, 2>;

// // 获取容量
// inline int64_t volume(const nvinfer1::Dims& d) {
//     return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
// }

// 根据type获取占用空间
inline uint32_t elementSize(nvinfer1::DataType t) {
    switch (t) {
        case nvinfer1::DataType::kINT32:
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
        case nvinfer1::DataType::kBOOL: return 1;
    }
    return 0;
}

// template <class EngineType, class ContextType>
// class FillBindingClosure
// {
// private:
//     using InputsMap = std::unordered_map<std::string, std::string>;
//     using BindingsVector = std::vector<std::unique_ptr<Bindings>>;

//     EngineType const* engine;
//     ContextType const* context;
//     // InputsMap const& inputs;
//     BindingsVector& bindings; //给每个 stream存了一份bindings
//     // int32_t batch;
//     // int32_t endBindingIndex;

//     void fillOneBinding(int32_t bindingIndex, int64_t vol)
//     {
//         auto const dims = getDims(bindingIndex);
//         auto const name = engine->getBindingName(bindingIndex);
//         auto const isInput = engine->bindingIsInput(bindingIndex);
//         auto const dataType = engine->getBindingDataType(bindingIndex);
//         auto const *bindingInOutStr = isInput ? "input" : "output";
//         // 遍历 iEnv里的每个 stream对应的bings中对应[bindingIndex]的那个binding
//         for (auto& binding : bindings)
//         {
//             // auto const input = inputs.find(name);
//             // if (isInput && input != inputs.end())
//             // {
//             //     sample::gLogInfo << "Using values loaded from " << input->second << " for input " << name << std::endl;
//             //     binding->addBinding(bindingIndex, name, isInput, vol, dataType, input->second);
//             // }
//             // else
//             // {
//             sample::gLogInfo << "Using random values for " << bindingInOutStr << " " << name << std::endl;
//             binding->addBinding(bindingIndex, name, isInput, vol, dataType);
//             // }
//             sample::gLogInfo << "Created " << bindingInOutStr <<" binding for " << name << " with dimensions " << dims << std::endl;
//         }
//     }

//     bool fillAllBindings(int32_t batch, int32_t endBindingIndex)
//     {
//         // if (!validateTensorNames(inputs, engine, endBindingIndex))
//         // {
//         //     sample::gLogError << "Invalid tensor names found in --loadInputs flag." << std::endl;
//         //     return false;
//         // }

//         for (int32_t b = 0; b < endBindingIndex; b++)
//         {
//             auto const dims = getDims(b);
//             auto const comps = engine->getBindingComponentsPerElement(b);
//             auto const strides = context->getStrides(b);
//             int32_t const vectorDimIndex = engine->getBindingVectorizedDim(b);
//             auto const vol = volume(dims, strides, vectorDimIndex, comps, batch);
//             fillOneBinding(b, vol);
//         }
//         return true;
//     }

//     Dims getDims(int32_t bindingIndex);

// public:
//     FillBindingClosure(
//         EngineType const* _engine, ContextType const* _context, BindingsVector& _bindings)
//         : engine(_engine)
//         , context(_context)
//         // , inputs(_inputs)
//         , bindings(_bindings)
//         // , batch(_batch)
//         // , endBindingIndex(_engine->get)
//     {
//         int32_t const nbOptProfiles = engine->getNbOptimizationProfiles();
//         int32_t const nbBindings = engine->getNbBindings();
//         int32_t const bindingsInProfile = nbOptProfiles > 0 ? nbBindings / nbOptProfiles : 0;
//         // endBindingIndex = bindingsInProfile ? bindingsInProfile : engine->getNbBindings();
//     }

//     // bool operator()()
//     // {
//     //     return fillAllBindings(batch, endBindingIndex);
//     // }

//     bool fill(int32_t b, int32_t e) {
//         return fillAllBindings(b, e);
//     }
// };

// template <>
// Dims FillBindingClosure<nvinfer1::ICudaEngine, nvinfer1::IExecutionContext>::getDims(int32_t bindingIndex)
// {
//     return context->getBindingDimensions(bindingIndex);
// }

// template <>
// Dims FillBindingClosure<nvinfer1::safe::ICudaEngine, nvinfer1::safe::IExecutionContext>::getDims(int32_t bindingIndex)
// {
//     return engine->getBindingDimensions(bindingIndex);
// }

// 一个binging，及buffer
struct Binding
{
    bool isInput{false};
    std::unique_ptr<IMirroredBuffer> buffer;
    int64_t volume{0};
    nvinfer1::DataType dataType{nvinfer1::DataType::kFLOAT};

    //void fill(std::string const& fileName);
    // 从文件填充
    void fill(std::string const& fileName)
    {
        loadFromFile(fileName, static_cast<char*>(buffer->getHostBuffer()), buffer->getSize());
    }

    void fill(void* data)
    {
        // loadFromFile(fileName, static_cast<char*>(buffer->getHostBuffer()), buffer->getSize());

        // buffer->getHostBuffer() = data;
        memcpy(buffer->getHostBuffer(), data, buffer->getSize());
    }

    // void fill();
    // 随机填充
    void fill()
    {
        switch (dataType)
        {
        case nvinfer1::DataType::kBOOL:
        {
            fillBuffer<bool>(buffer->getHostBuffer(), volume, 0, 1);
            break;
        }
        case nvinfer1::DataType::kINT32:
        {
            fillBuffer<int32_t>(buffer->getHostBuffer(), volume, -128, 127);
            break;
        }
        case nvinfer1::DataType::kINT8:
        {
            fillBuffer<int8_t>(buffer->getHostBuffer(), volume, -128, 127);
            break;
        }
        case nvinfer1::DataType::kFLOAT:
        {
            fillBuffer<float>(buffer->getHostBuffer(), volume, -1.0F, 1.0F);
            break;
        }
        case nvinfer1::DataType::kHALF:
        {
            fillBuffer<__half>(buffer->getHostBuffer(), volume, -1.0F, 1.0F);
            break;
        }
        }
    }

    // void dump(std::ostream& os, Dims dims, Dims strides, int32_t vectorDim, int32_t spv,
    //     std::string const separator = " ") const;

    void dump(std::ostream& os, Dims dims, Dims strides, int32_t vectorDim, int32_t spv,
        std::string const separator /*= " "*/) const
    {
        switch (dataType)
        {
        case nvinfer1::DataType::kBOOL:
        {
            dumpBuffer<bool>(buffer->getHostBuffer(), separator, os, dims, strides, vectorDim, spv);
            break;
        }
        case nvinfer1::DataType::kINT32:
        {
            dumpBuffer<int32_t>(buffer->getHostBuffer(), separator, os, dims, strides, vectorDim, spv);
            break;
        }
        case nvinfer1::DataType::kINT8:
        {
            dumpBuffer<int8_t>(buffer->getHostBuffer(), separator, os, dims, strides, vectorDim, spv);
            break;
        }
        case nvinfer1::DataType::kFLOAT:
        {
            dumpBuffer<float>(buffer->getHostBuffer(), separator, os, dims, strides, vectorDim, spv);
            break;
        }
        case nvinfer1::DataType::kHALF:
        {
            dumpBuffer<__half>(buffer->getHostBuffer(), separator, os, dims, strides, vectorDim, spv);
            break;
        }
        }
    }
};

// 
class Bindings
{
public:
    Bindings() = delete;
    explicit Bindings(bool useManaged)
        : mUseManaged(useManaged)
    {
    }
    
    void addBinding(int b, std::string const& name, bool isInput, int64_t volume, nvinfer1::DataType dataType,
        std::string const& fileName /*= ""*/)
    {
        while (mBindings.size() <= static_cast<size_t>(b))
        {
            mBindings.emplace_back();
            mDevicePointers.emplace_back();
        }
        mNames[name] = b;
        if (mBindings[b].buffer == nullptr)
        {
            if (mUseManaged)
            {
                mBindings[b].buffer.reset(new UnifiedMirroredBuffer);
            }
            else
            {
                mBindings[b].buffer.reset(new DiscreteMirroredBuffer);
            }
        }
        mBindings[b].isInput = isInput;
        // Some memory allocators return nullptr when allocating zero bytes, but TensorRT requires a non-null ptr
        // even for empty tensors, so allocate a dummy byte.
        if (volume == 0)
        {
            mBindings[b].buffer->allocate(1);
        }
        else
        {
            mBindings[b].buffer->allocate(static_cast<size_t>(volume) * static_cast<size_t>(dataTypeSize(dataType)));
        }
        mBindings[b].volume = volume;
        mBindings[b].dataType = dataType;
        mDevicePointers[b] = mBindings[b].buffer->getDeviceBuffer();
        if (isInput)
        {
            if (fileName.empty())
            {
                fill(b);
            }
            else
            {
                fill(b, fileName);
            }
        }
    }


    void addBinding(int b, std::string const& name, bool isInput, int64_t volume, nvinfer1::DataType dataType)
    {
        while (mBindings.size() <= static_cast<size_t>(b))
        {
            mBindings.emplace_back();
            mDevicePointers.emplace_back();
        }
        mNames[name] = b;
        if (mBindings[b].buffer == nullptr)
        {
            if (mUseManaged)
            {
                mBindings[b].buffer.reset(new UnifiedMirroredBuffer);
            }
            else
            {
                mBindings[b].buffer.reset(new DiscreteMirroredBuffer);
            }
        }
        mBindings[b].isInput = isInput;
        // Some memory allocators return nullptr when allocating zero bytes, but TensorRT requires a non-null ptr
        // even for empty tensors, so allocate a dummy byte.
        if (volume == 0)
        {
            mBindings[b].buffer->allocate(1);
        }
        else
        {
            mBindings[b].buffer->allocate(static_cast<size_t>(volume) * static_cast<size_t>(dataTypeSize(dataType)));
        }
        mBindings[b].volume = volume;
        mBindings[b].dataType = dataType;
        mDevicePointers[b] = mBindings[b].buffer->getDeviceBuffer();
    }


    void** getDeviceBuffers()
    {
        return mDevicePointers.data();
    }

    void transferInputToDevice(TrtCudaStream& stream)
    {
        for (auto& b : mNames)
        {
            if (mBindings[b.second].isInput)
            {
                mBindings[b.second].buffer->hostToDevice(stream);
            }
        }
    }

    void transferOutputToHost(TrtCudaStream& stream)
    {
        for (auto& b : mNames)
        {
            if (!mBindings[b.second].isInput)
            {
                mBindings[b.second].buffer->deviceToHost(stream);
            }
        }
    }

    // void dumpBindingDimensions(int binding, nvinfer1::IExecutionContext const& context, std::ostream& os) const
    // {
    //     auto const dims = context.getBindingDimensions(binding);
    //     // Do not add a newline terminator, because the caller may be outputting a JSON string.
    //     os << dims;
    // }

    // void dumpBindingValues(nvinfer1::IExecutionContext const& context, int binding, std::ostream& os,
    //     std::string const& separator /*= " "*/, int32_t batch /*= 1*/) const
    // {
    //     Dims dims = context.getBindingDimensions(binding);
    //     Dims strides = context.getStrides(binding);
    //     int32_t vectorDim = context.getEngine().getBindingVectorizedDim(binding);
    //     int32_t const spv = context.getEngine().getBindingComponentsPerElement(binding);

    //     if (context.getEngine().hasImplicitBatchDimension())
    //     {
    //         auto insertN = [](Dims& d, int32_t bs) {
    //             int32_t const nbDims = d.nbDims;
    //             ASSERT(nbDims < Dims::MAX_DIMS);
    //             std::copy_backward(&d.d[0], &d.d[nbDims], &d.d[nbDims + 1]);
    //             d.d[0] = bs;
    //             d.nbDims = nbDims + 1;
    //         };
    //         int32_t batchStride = 0;
    //         for (int32_t i = 0; i < strides.nbDims; ++i)
    //         {
    //             if (strides.d[i] * dims.d[i] > batchStride)
    //             {
    //                 batchStride = strides.d[i] * dims.d[i];
    //             }
    //         }
    //         insertN(dims, batch);
    //         insertN(strides, batchStride);
    //         vectorDim = (vectorDim == -1) ? -1 : vectorDim + 1;
    //     }

    //     mBindings[binding].dump(os, dims, strides, vectorDim, spv, separator);
    // }

    // void dumpBindings(
    //     nvinfer1::IExecutionContext const& context, std::function<bool(Binding const&)> predicate, std::ostream& os) const
    // {
    //     for (auto const& n : mNames)
    //     {
    //         auto const binding = n.second;
    //         if (predicate(mBindings[binding]))
    //         {
    //             os << n.first << ": (";
    //             dumpBindingDimensions(binding, context, os);
    //             os << ")" << std::endl;

    //             dumpBindingValues(context, binding, os);
    //             os << std::endl;
    //         }
    //     }
    // }

    std::unordered_map<std::string, int> getBindings(std::function<bool(Binding const&)> predicate) const
    {
        std::unordered_map<std::string, int> bindings;
        for (auto const& n : mNames)
        {
            auto const binding = n.second;
            if (predicate(mBindings[binding]))
            {
                bindings.insert(n);
            }
        }
        return bindings;
    }

    void fill(int binding, std::string const& fileName)
    {
        mBindings[binding].fill(fileName);
    }

    void fill(int binding)
    {
        mBindings[binding].fill();
    }

    void fill(int binding, void* data)
    {
        mBindings[binding].fill(data);
    }

private:
    std::unordered_map<std::string, int32_t> mNames;
    std::vector<Binding> mBindings;
    std::vector<void*> mDevicePointers;
    bool mUseManaged{false};
};


struct Enqueue
{
    explicit Enqueue(nvinfer1::IExecutionContext& context, void** buffers)
        : mContext(context)
        , mBuffers(buffers)
    {
    }

    nvinfer1::IExecutionContext& mContext;
    void** mBuffers{};
};

//!
//! \class EnqueueImplicit
//! \brief Functor to enqueue inference with implict batch
//!
class EnqueueImplicit : private Enqueue
{

public:
    explicit EnqueueImplicit(nvinfer1::IExecutionContext& context, void** buffers, int32_t batch)
        : Enqueue(context, buffers)
        , mBatch(batch)
    {
    }

    bool operator()(TrtCudaStream& stream) const
    {
        if (mContext.enqueue(mBatch, mBuffers, stream.get(), nullptr))
        {
            // Collecting layer timing info from current profile index of execution context
            if (mContext.getProfiler() && !mContext.getEnqueueEmitsProfile() && !mContext.reportToProfiler())
            {
                gLogWarning << "Failed to collect layer timing info from previous enqueue()" << std::endl;
            }
            return true;
        }
        return false;
    }

private:
    int32_t mBatch;
};

//!
//! \class EnqueueExplicit
//! \brief Functor to enqueue inference with explict batch
//!
class EnqueueExplicit : private Enqueue
{

public:
    explicit EnqueueExplicit(nvinfer1::IExecutionContext& context, void** buffers)
        : Enqueue(context, buffers)
    {
    }

    bool operator()(TrtCudaStream& stream) const
    {
        if (mContext.enqueueV2(mBuffers, stream.get(), nullptr))
        {
            // Collecting layer timing info from current profile index of execution context
            if (mContext.getProfiler() && !mContext.getEnqueueEmitsProfile() && !mContext.reportToProfiler())
            {
                gLogWarning << "Failed to collect layer timing info from previous enqueueV2()" << std::endl;
            }
            return true;
        }
        return false;
    }
};

//!
//! \class EnqueueGraph
//! \brief Functor to enqueue inference from CUDA Graph
//!
class EnqueueGraph
{

public:
    explicit EnqueueGraph(nvinfer1::IExecutionContext& context, TrtCudaGraph& graph)
        : mGraph(graph)
        , mContext(context)
    {
    }

    bool operator()(TrtCudaStream& stream) const
    {
        if (mGraph.launch(stream))
        {
            // Collecting layer timing info from current profile index of execution context
            if (mContext.getProfiler() && !mContext.getEnqueueEmitsProfile() && !mContext.reportToProfiler())
            {
                gLogWarning << "Failed to collect layer timing info from previous CUDA graph launch" << std::endl;
            }
            return true;
        }
        return false;
    }

    TrtCudaGraph& mGraph;
    nvinfer1::IExecutionContext& mContext;
};

// //!
// //! \class EnqueueGraphSafe
// //! \brief Functor to enqueue inference from CUDA Graph
// //!
// class EnqueueGraphSafe
// {

// public:
//     explicit EnqueueGraphSafe(TrtCudaGraph& graph)
//         : mGraph(graph)
//     {
//     }

//     bool operator()(TrtCudaStream& stream) const
//     {
//         return mGraph.launch(stream);
//     }

//     TrtCudaGraph& mGraph;
// };

// //!
// //! \class EnqueueSafe
// //! \brief Functor to enqueue safe execution context
// //!
// class EnqueueSafe
// {
// public:
//     explicit EnqueueSafe(nvinfer1::safe::IExecutionContext& context, void** buffers)
//         : mContext(context)
//         , mBuffers(buffers)
//     {
//     }

//     bool operator()(TrtCudaStream& stream) const
//     {
//         if (mContext.enqueueV2(mBuffers, stream.get(), nullptr))
//         {
//             return true;
//         }
//         return false;
//     }

//     nvinfer1::safe::IExecutionContext& mContext;
//     void** mBuffers{};
// };

//!
//! \class Iteration
//! \brief Inference iteration and streams management
//!
// 一个 iteration 对应一个stream（multi-stream）
template <class ContextType>
class Iteration
{

public:
    // 每个id对应一个MultipleThread
    // Iteration(int32_t id, InferenceOptions const& inference, ContextType& context, Bindings& bindings
    Iteration(
        int32_t id, ContextType& context, Bindings& bindings,
        int overlap, bool spin, int batch, bool graph
    ): mBindings(bindings)
        , mStreamId(id)
        // , mDepth(1 + inference.overlap) // overlap = !exposeDMA 串行化进出设备的 DMA 传输。 （默认 = 禁用）
        , mDepth(1 + overlap) // overlap = !exposeDMA 串行化进出设备的 DMA 传输。 （默认 = 禁用）
        , mActive(mDepth)
        , mEvents(mDepth)
        , mEnqueueTimes(mDepth)
        , mContext(&context)
    {   
        // 初始化event
        for (int32_t d = 0; d < mDepth; ++d)
        {
            for (int32_t e = 0; e < static_cast<int32_t>(EventType::kNUM); ++e)
            {
                mEvents[d][e].reset(new TrtCudaEvent(!spin));
            }
        }
        // 根据 inference选项  context 和 bindings 设置mEqueue func
        createEnqueueFunction(context, bindings, batch, graph);

        // nvinfer1::IExecutionContext& context, Bindings& bindings, int batch, bool graph
    }


    // 发起推理任务， 返回是否执行成功
    bool query(bool skipTransfers)
    {
        if (mActive[mNext])
        {
            return true;
        }

        // 如果不禁用传输，则进行 D2H
        if (!skipTransfers)
        {   
            // 给 input 流插桩 start， 给 input 对应的 stream，插入 input_s事件
            record(EventType::kINPUT_S, StreamType::kINPUT);
            // H2D 不sync
            setInputData(false);
            // 给 inputs流插桩 end 给 input 对应的 stream，插入 input_e事件
            record(EventType::kINPUT_E, StreamType::kINPUT);
            // compute 流等待 传输流完成
            wait(EventType::kINPUT_E, StreamType::kCOMPUTE); // Wait for input DMA before compute
        }

        record(EventType::kCOMPUTE_S, StreamType::kCOMPUTE);
        recordEnqueueTime();
        // RUN (mEnqueue) 
        if (!mEnqueue(getStream(StreamType::kCOMPUTE)))
        {
            return false;
        }
        recordEnqueueTime();
        record(EventType::kCOMPUTE_E, StreamType::kCOMPUTE);

        // 如果不禁用传输，则进行 H2D
        if (!skipTransfers)
        {
            wait(EventType::kCOMPUTE_E, StreamType::kOUTPUT); // Wait for compute before output DMA
            record(EventType::kOUTPUT_S, StreamType::kOUTPUT);
            fetchOutputData(false);
            record(EventType::kOUTPUT_E, StreamType::kOUTPUT);
        }
        mActive[mNext] = true;
        moveNext(); // 这没看懂
        return true;
    }

    // 等待推理任务结束，记录结果到trace，返回 从 gpu启动到计算开始的时间
    float sync(
        // TimePoint const& cpuStart, TrtCudaEvent const& gpuStart, std::vector<InferenceTrace>& trace, bool skipTransfers
        TimePoint const& cpuStart, TrtCudaEvent const& gpuStart, bool skipTransfers
    )
    {
        if (mActive[mNext])
        {
            if (skipTransfers)
            {
                getEvent(EventType::kCOMPUTE_E).synchronize();
            }
            else
            {
                getEvent(EventType::kOUTPUT_E).synchronize();
            }
            // trace.emplace_back(getTrace(cpuStart, gpuStart, skipTransfers));
            mActive[mNext] = false;
            return getEvent(EventType::kCOMPUTE_S) - gpuStart;
        }
        return 0;
    }
    

    void syncAll(
        TimePoint const& cpuStart, TrtCudaEvent const& gpuStart, bool skipTransfers)
    {
        for (int32_t d = 0; d < mDepth; ++d)
        {
            sync(cpuStart, gpuStart, skipTransfers);
            moveNext();
        }
    }

    void wait(TrtCudaEvent& gpuStart)
    {
        getStream(StreamType::kINPUT).wait(gpuStart);
    }

    // H2D ， 如果 sync则等待传输完毕
    void setInputData(bool sync)
    {
        mBindings.transferInputToDevice(getStream(StreamType::kINPUT));
        // additional sync to avoid overlapping with inference execution.
        if (sync)
        {
            getStream(StreamType::kINPUT).synchronize();
        }
    }

    // D2H，如果sync true则等待传输完成
    void fetchOutputData(bool sync)
    {
        mBindings.transferOutputToHost(getStream(StreamType::kOUTPUT));
        // additional sync to avoid overlapping with inference execution.
        if (sync)
        {
            getStream(StreamType::kOUTPUT).synchronize();
        }
    }

private:
    void moveNext()
    {   
        // d = 1
        // 0 -> 0
        //
         
        // d = 2
        // 0 --> 1
        // 1 --> 0

        // d = 3
        // 0 --> 2
        // 2 --> 0
        mNext = mDepth - 1 - mNext;
    }

    TrtCudaStream& getStream(StreamType t)
    {
        return mStream[static_cast<int32_t>(t)];
    }

    TrtCudaEvent& getEvent(EventType t)
    {
        return *mEvents[mNext][static_cast<int32_t>(t)];
    }

    // 插桩
    void record(EventType e, StreamType s)
    {
        getEvent(e).record(getStream(s));
    }

    void recordEnqueueTime()
    {
        mEnqueueTimes[mNext][enqueueStart] = getCurrentTime();
        enqueueStart = 1 - enqueueStart; //自旋
    }

    TimePoint getEnqueueTime(bool start)
    {
        return mEnqueueTimes[mNext][start ? 0 : 1];
    }

    void wait(EventType e, StreamType s)
    {
        getStream(s).wait(getEvent(e));
    }

    // // cpuStart 和 gpuStart要分开（测量方式不同，其实是一个时间点）
    // InferenceTrace getTrace(TimePoint const& cpuStart, TrtCudaEvent const& gpuStart, bool skipTransfers)
    // {
    //     float is
    //         = skipTransfers ? getEvent(EventType::kCOMPUTE_S) - gpuStart : getEvent(EventType::kINPUT_S) - gpuStart;
    //     float ie
    //         = skipTransfers ? getEvent(EventType::kCOMPUTE_S) - gpuStart : getEvent(EventType::kINPUT_E) - gpuStart;
    //     float os
    //         = skipTransfers ? getEvent(EventType::kCOMPUTE_E) - gpuStart : getEvent(EventType::kOUTPUT_S) - gpuStart;
    //     float oe
    //         = skipTransfers ? getEvent(EventType::kCOMPUTE_E) - gpuStart : getEvent(EventType::kOUTPUT_E) - gpuStart;

    //     return InferenceTrace(mStreamId,
    //         std::chrono::duration<float, std::milli>(getEnqueueTime(true) - cpuStart).count(),
    //         std::chrono::duration<float, std::milli>(getEnqueueTime(false) - cpuStart).count(), is, ie,
    //         getEvent(EventType::kCOMPUTE_S) - gpuStart, getEvent(EventType::kCOMPUTE_E) - gpuStart, os, oe);
    // }

    void createEnqueueFunction(
        // InferenceOptions const& inference, nvinfer1::IExecutionContext& context, Bindings& bindings
        nvinfer1::IExecutionContext& context, Bindings& bindings, int batch, bool graph)
    {
        if (context.getEngine().hasImplicitBatchDimension())
        {
            mEnqueue = EnqueueFunction(EnqueueImplicit(context, mBindings.getDeviceBuffers(), batch));
        }
        else
        {
            mEnqueue = EnqueueFunction(EnqueueExplicit(context, mBindings.getDeviceBuffers()));
        }
        if (graph)
        {
            TrtCudaStream& stream = getStream(StreamType::kCOMPUTE);
            // Avoid capturing initialization calls by executing the enqueue function at least
            // once before starting CUDA graph capture.
            auto const ret = mEnqueue(stream);
            assert(ret);
            stream.synchronize();

            mGraph.beginCapture(stream);
            // The built TRT engine may contain operations that are not permitted under CUDA graph capture mode.
            // When the stream is capturing, the enqueue call may return false if the current CUDA graph capture fails.
            if (mEnqueue(stream))
            {
                mGraph.endCapture(stream);
                mEnqueue = EnqueueFunction(EnqueueGraph(context, mGraph));
            }
            else
            {
                mGraph.endCaptureOnError(stream);
                // Ensure any CUDA error has been cleaned up.
                cudaCheck(cudaGetLastError());
                gLogWarning << "The built TensorRT engine contains operations that are not permitted under "
                                       "CUDA graph capture mode."
                                    << std::endl;
                gLogWarning << "The specified --useCudaGraph flag has been ignored. The inference will be "
                                       "launched without using CUDA graph launch."
                                    << std::endl;
            }
        }
    }

    // // safety 方式 
    // void createEnqueueFunction(
    //     // InferenceOptions const& inference, nvinfer1::safe::IExecutionContext& context, Bindings&
    //     nvinfer1::safe::IExecutionContext& context, Bindings&, int batch, bool graph
    //     )
    // {
    //     mEnqueue = EnqueueFunction(EnqueueSafe(context, mBindings.getDeviceBuffers()));
    //     // 如果使用 cudaGraph则初始化
    //     if (graph)
    //     {
    //         TrtCudaStream& stream = getStream(StreamType::kCOMPUTE);
    //         ASSERT(mEnqueue(stream));
    //         stream.synchronize();
    //         mGraph.beginCapture(stream);
    //         ASSERT(mEnqueue(stream));
    //         mGraph.endCapture(stream);
    //         mEnqueue = EnqueueFunction(EnqueueGraphSafe(mGraph));
    //     }

    // }

    Bindings& mBindings;

    TrtCudaGraph mGraph;
    EnqueueFunction mEnqueue; // 函数指针 std::function<bool(TrtCudaStream&)>;

    int32_t mStreamId{0};
    int32_t mNext{0};
    int32_t mDepth{2}; // default to double buffer to hide DMA transfers

    std::vector<bool> mActive;
    MultiStream mStream; //初始化3个流 H2D run D2H 两个 depth共用 Stream
    std::vector<MultiEvent> mEvents;  //初始化6个event 对应上面3个开始及结束 两个depth 都有独立的六个 event

    int32_t enqueueStart{0};
    std::vector<EnqueueTimes> mEnqueueTimes;
    ContextType* mContext{nullptr};
};


// 根据type获取占用空间
inline std::string getDtype(nvinfer1::DataType t) {
    switch (t) {
        case nvinfer1::DataType::kINT32: return "int32";
        case nvinfer1::DataType::kFLOAT: return "float32";
        case nvinfer1::DataType::kHALF: return "float16";
        case nvinfer1::DataType::kINT8: return "int8";
        case nvinfer1::DataType::kBOOL: return "bool";
    }
    return "";
}


inline std::vector<int> getShape(nvinfer1::Dims dims) {
    std::vector<int> shape;
    if (dims.nbDims == 0) {
        shape.emplace_back(0);
        return shape;
    }
    for (int idx = 0; idx < dims.nbDims; idx ++) {
        shape.emplace_back(dims.d[idx]);
    }
    return shape;
}

// bool processInput(const samplesCommon::BufferManager& buffers, const float* data, size)
// {
//     // Fill data buffer
//     float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
//     std::memcpy(hostDataBuffer, data, mParams.batchSize * samplesCommon::volume(mInputDims) * sizeof(float));
//     return true;
// }

struct TrtInference {
    TrtInference(
        const std::string& enginePath, 
        // const int maxBatchSize = 1,
        const bool enableGraph = false,
        const bool spin = false,
        const int overloap = 0,
        const int batch = 1,
        const int nStreams = 1
    ): mEnableGraph(enableGraph) {   
        mStreamNum = nStreams;
        gLogInfo << "--------------------\n";
        // gLogInfo << "Using BERT inference C++\n";
        if (enableGraph)
        {
            gLogInfo << "CUDA Graph is enabled\n";
        }
        else
        {
            gLogInfo << "CUDA Graph is disabled\n";
        }

        gLogInfo << "--------------------\n";

        initLibNvInferPlugins(&gLogger, "");

        // gLogInfo << "Loading Inference Engine ... \n";
        // mEngine = loadEngine(enginePath);

        // if (mEngine == nullptr) {
        //     gLogError << "Error loading engine\n";
        //     exit(-1);
        // }
        // gLogInfo << "Done ... \n";

        // // 创建context
        // gLogInfo << "Creating Context ... \n";


        // // 创建流
        // gLogInfo << "Creating Stream ... \n";
        // gpuErrChk(cudaStreamCreate(&mStream));

        setUp(enginePath, nStreams, batch, overloap, spin, enableGraph);

        // gLogInfo << "Alloc Memory on Device & Host ... \n";
        // allocateBindings(maxBatchSize);
    }

    TrtUniquePtr<ICudaEngine> loadEngine(const std::string& enginePath) {
        
        std::ifstream input(enginePath, std::ios::binary);
        if (!input) {
            gLogError << "Error opening engine file: " << enginePath << "\n";
            return nullptr;
        }

        input.seekg(0, input.end);
        const size_t fsize = input.tellg();
        input.seekg(0, input.beg);

        std::vector<char> bytes(fsize);
        input.read(bytes.data(), fsize);

        auto runtime = TrtUniquePtr<IRuntime>(createInferRuntime(gLogger));
        if (runtime == nullptr) {
            gLogError << "Error creating TRT runtime\n";
            return nullptr;
        }

        auto engine = TrtUniquePtr<ICudaEngine>(runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
        if (engine == nullptr) {
            gLogError << "Error deserializing CUDA engine\n";
            return nullptr;
        }

        return engine;
    }

    bool setUp(
        const std::string& enginePath, int streamNum, int batch,
        const int overlap, bool spin, bool graph) {

        auto useManagedMemory = false;
        // load engine
        mEngine = loadEngine(enginePath);
        if (mEngine == nullptr) {
            gLogError << "Error loading engine\n";
            exit(-1);
        }

        // 初始化 contexts, bindings, fillobj, iStreams
        for (auto i = 0; i < streamNum; i++) {
            // 为每个stream构建contexts
            // auto tmp_context = std::unique_ptr<IExecutionContext>(mEngine->createExecutionContext());
            auto tmp_context = mEngine->createExecutionContext();
            if (!tmp_context) {
                gLogError << "Error creating execution context\n";
                exit(-1);
            }
            mContexts.emplace_back(tmp_context);

            // 为每个stream构建 bindings
            auto tmp_bindings = new Bindings(useManagedMemory);
            int32_t const nbOptProfiles = mEngine->getNbOptimizationProfiles();
            int32_t const nbBindings = mEngine->getNbBindings();
            int32_t const bindingsInProfile = nbOptProfiles > 0 ? nbBindings / nbOptProfiles : 0;
            int32_t const endBindingIndex = bindingsInProfile ? bindingsInProfile : mEngine->getNbBindings();
            
            for (int32_t b = 0; b < endBindingIndex; b++) {
                auto const dims = tmp_context->getBindingDimensions(b);
                // auto const comps = mEngine->getBindingComponentsPerElement(b);
                // auto const strides = tmp_context->getStrides(b);
                // int32_t const vectorDimIndex = mEngine->getBindingVectorizedDim(b);
                // auto const vol = volume(dims, strides, vectorDimIndex, comps, batch);
                auto const vol = volume(dims);
                auto const name = mEngine->getBindingName(b);
                auto const isInput = mEngine->bindingIsInput(b);
                auto const dataType = mEngine->getBindingDataType(b);
                auto const *bindingInOutStr = isInput ? "input" : "output";
                tmp_bindings->addBinding(b, name, isInput, vol, dataType);

                auto size = vol * elementSize(dataType);
                if (i == 0) {
                    if (isInput) {
                        mInputNames.emplace_back(name);
                        mInputElemSizeList.emplace_back(elementSize(dataType));
                        mInputDataTypeList.emplace_back(dataType);
                        mInputSizeList.emplace_back(size);
                        mInputShapeList.emplace_back(getShape(dims));
                    } else {
                        mOutputNames.emplace_back(name);
                        mOutputElemSizeList.emplace_back(elementSize(dataType));
                        mOutputDataTypeList.emplace_back(dataType);
                        mOutputSizeList.emplace_back(size);
                        mOutputShapeList.emplace_back(getShape(dims));
                    }
                }
            }
            mBindings.emplace_back(tmp_bindings);

            // 构建 stream
            auto* tmp_iteration = new Iteration<IExecutionContext>(
                (int32_t)i, *tmp_context, *tmp_bindings,
                overlap , spin, batch, mEnableGraph
            );
            mStreams.emplace_back(tmp_iteration);
        }
        return true;
    }



    // 提前malloc 还是
    // bool allocateBindings(const int maxBatchSize) {

    //     assert(mEngine != nullptr);
        
    //     for (int32_t b = 0; b < mEngine->getNbBindings(); ++b) {

    //         // auto dims = mContext->getBindingDimensions(b);
    //         auto dims = mEngine->getBindingDimensions(b);
    //         std::string name(mEngine->getBindingName(b));
            
    //         auto isInput = mEngine->bindingIsInput(b);
    //         nvinfer1::DataType dataType = mEngine->getBindingDataType(b);
    //         auto const *bindingInOutStr = isInput ? "input" : "output";
            
    //         auto size = volume(dims) * elementSize(dataType);
    //         auto elemSize = elementSize(dataType);
    //         void* devBuf = nullptr;
    //         gpuErrChk(cudaMalloc(&devBuf, size));
    //         gpuErrChk(cudaMemset(devBuf, 0, size));

    //         mDevBufferMap[name] = devBuf;
    //         mSizeMap[name] = size;
    //         mShapeMap[name] = getShape(dims);
    //         mDtypeMap[name] = getDtype(dataType);
    //         // 警惕 double free
    //         mBindings.emplace_back(devBuf);

    //         if (isInput) {
    //             mInputNames.emplace_back(name);
    //             mInputDevBufferList.emplace_back(devBuf);
    //             mInputSizeList.emplace_back(size);
    //             mInputShapeList.emplace_back(getShape(dims));
    //             mInputElemSizeList.emplace_back(elemSize);
    //             mInputDataTypeList.emplace_back(dataType);

    //         } else {
    //             void* hostBuf = nullptr;
    //             hostBuf = malloc(size);
    //             memset(hostBuf, 0, size);
    //             mHostBufferMap[name] = hostBuf;
    //             mOutputNames.emplace_back(name);
    //             mOutputDevBufferList.emplace_back(devBuf);
    //             mOutputHostBufferList.emplace_back(hostBuf);
    //             mOutputSizeList.emplace_back(size);
    //             mOutputShapeList.emplace_back(getShape(dims));
    //             mOutputElemSizeList.emplace_back(elemSize);
    //             mOutputDataTypeList.emplace_back(dataType);
    //         }
    //     }
    // }

    // void prepare(int profIdx) {

    //     if (!mContext->allInputDimensionsSpecified())
    //     {
    //         gLogError << "Not all input dimensions are specified for the exeuction context\n";
    //         exit(-1);
    //     }

    //     if (mEnableGraph)
    //     {
    //         cudaGraph_t graph;
    //         cudaGraphExec_t exec;
    //         // warm up and let mContext do cublas initialization
    //         bool status = mContext->enqueueV2(mBindings.data(), mStream, nullptr);
    //         if (!status)
    //         {
    //             gLogError << "Enqueue failed\n";
    //             exit(-1);
    //         }
    //         gLogVerbose << "Capturing graph\n";

    //         gpuErrChk(cudaStreamBeginCapture(mStream, cudaStreamCaptureModeRelaxed));
    //         status = mContext->enqueueV2(mBindings.data(), mStream, nullptr);
    //         if (!status)
    //         {
    //             gLogError << "Enqueue failed\n";
    //             exit(-1);
    //         }

    //         gpuErrChk(cudaStreamEndCapture(mStream, &graph));
    //         gpuErrChk(cudaStreamSynchronize(mStream));

    //         gpuErrChk(cudaGraphInstantiate(&exec, graph, NULL, NULL, 0));
    //         mExecGraph = exec;
    //     }
    //     // mCuSeqlens.resize(batchSize + 1);
    //     // std::generate(mCuSeqlens.begin(), mCuSeqlens.end(), [pos = -mSeqLength, this]() mutable{ pos += mSeqLength; return pos; });
    // }

    bool setBufOneBinding(void* data, int streamId, int bindingIndex) {
        mBindings[streamId]->fill(bindingIndex, data);
        return true;
    }

    bool setBuf(void* data, int bindingIndex, int64_t dataSize) {
        auto bindingSize = mInputSizeList[bindingIndex];
        
        if (dataSize != mStreamNum * bindingSize) {
            gLogError << "set buf("<< mInputNames[bindingIndex] <<
                ") error,  bindings size(" << bindingSize << ")* streamNum(" <<
                mStreamNum << ") != dataSize(" << dataSize <<")" ;
            exit(-1);
        }

        for (auto i=0; i<mStreamNum; i++) {
            setBufOneBinding((uint8_t *)data + i * bindingSize, i, bindingIndex);
        }
        return true;
    }

    void infer(std::vector<void*> inputBufferList, std::vector<int64_t> inputBufferSizeList) {

        SyncStruct sync;
        // sync.sleep = inference.sleep;
        sync.sleep = 0;
        sync.mainStream.sleep(&sync.sleep);
        // 这俩是一个时间点，但是因为cuda记录要在 加时间戳
        sync.cpuStart = getCurrentTime();
        sync.gpuStart.record(sync.mainStream);

        bool skipTransfers = false;

        // gLogInfo << "H2D ... split each input to streams\n";
        std::cout << "H2D ... split each input to streams\n";
        const auto t0 = std::chrono::high_resolution_clock::now();

        if (mInputNames.size() != inputBufferList.size()) {
            gLogError << "input buf size != engine inputs num" ;
            return;
        }

        for (auto i=0; i < inputBufferList.size(); i++) {
            // setBuf(inputBufferList[i], i, mInputSizeList[i]);
            setBuf(inputBufferList[i], i, inputBufferSizeList[i]);
        }

        for (auto& s : mStreams)
        {   
            if (!s->query(skipTransfers))
            {   
                gLogError << "infer fail 1" ;
                return;
            }
        }
        
        for (auto& s : mStreams)
        {
            s->sync(sync.cpuStart, sync.gpuStart, skipTransfers);
        }   
    }


    // void run2(std::vector<void*> inputBufferList) {   

    //     gLogInfo << "H2D ... \n";
    //     const auto t0 = std::chrono::high_resolution_clock::now();
    //     // float dt_d2h_all = 0;

    //     for (auto i = 0; i < mInputNames.size(); i++) {
    //         gpuErrChk(
    //             cudaMemcpyAsync(mInputDevBufferList[i], inputBufferList[i], mInputSizeList[i], cudaMemcpyHostToDevice, mStream));
    //     }


    //     // for (auto inName: mInputNames) {
    //     //     const auto t00 = std::chrono::high_resolution_clock::now();
    //     //     gpuErrChk(
    //     //         cudaMemcpyAsync(mDevBufferMap[inName], inputBufferMap[inName], mSizeMap[inName], cudaMemcpyHostToDevice, mStream));
    //     //     const auto t01 = std::chrono::high_resolution_clock::now();
    //     //     const float dt_each_h2d = std::chrono::duration<float, std::milli>(t01 - t00).count();
    //     //     // gLogInfo << "H2D: " <<  inName <<", dt ==>" << dt_each_h2d << "\n";
    //     //     dt_d2h_all += dt_each_h2d;
    //     // }
    //     const auto t1 = std::chrono::high_resolution_clock::now();
    //     const float dt_h2d = std::chrono::duration<float, std::milli>(t1 - t0).count();
    //     gLogInfo << "H2D End, dt ==>" << dt_h2d << "\n";

    //     cudaEvent_t start, stop;
    //     gpuErrChk(cudaEventCreate(&start));
    //     gpuErrChk(cudaEventCreate(&stop));
    //     gpuErrChk(cudaEventRecord(start, mStream));
    //     gLogInfo << "Run ... \n";
    //     if (mEnableGraph) {
    //         // gLogInfo << "cudaGraphLaunch ... \n";
    //         gpuErrChk(cudaGraphLaunch(mExecGraph, mStream));
    //     } else {
    //         // gLogInfo << "enqueueV2 ... \n";
    //         bool status = mContext->enqueueV2(mBindings.data(), mStream, nullptr);
    //         if (!status) {
    //             gLogError << "Enqueue failed\n";
    //             exit(-1);
    //         }
    //     }
    //     gpuErrChk(cudaEventRecord(stop, mStream));
    //     const auto t2 = std::chrono::high_resolution_clock::now();
    //     const float dt_run = std::chrono::duration<float, std::milli>(t2 - t1).count();
    //     gLogInfo << "RUN End, dt ==>" << dt_run << "\n";

    //     // d2h
    //     gLogInfo << "D2H ... \n";
    //     // for (auto outName: mOutputNames) {
    //     //     gpuErrChk(cudaMemcpyAsync(
    //     //         mHostBufferMap[outName], mDevBufferMap[outName], mSizeMap[outName], cudaMemcpyDeviceToHost, mStream));
    //     // }

    //     for (auto i = 0; i < mOutputNames.size(); i++) {
    //         gpuErrChk(
    //             cudaMemcpyAsync(mOutputHostBufferList[i], mOutputDevBufferList[i], mOutputSizeList[i], cudaMemcpyDeviceToHost, mStream));
    //     }


    //     const auto t3 = std::chrono::high_resolution_clock::now();
    //     const float dt_d2h = std::chrono::duration<float, std::milli>(t3 - t2).count();
    //     gLogInfo << "D2H End, dt ==>" << dt_d2h << "\n";
    //     // sync
    //     // gLogInfo << "cudaStreamSynchronize ... \n";
    //     gLogInfo << "Sync ... \n";
    //     gpuErrChk(cudaStreamSynchronize(mStream));
    //     const auto t4 = std::chrono::high_resolution_clock::now();
    //     const float dt_sync = std::chrono::duration<float, std::milli>(t4 - t3).count();
    //     gLogInfo << "Sync End, dt ==>" << dt_sync << "\n";

    //     const float dt_infer = std::chrono::duration<float, std::milli>(t4 - t0).count();
    //     gLogInfo << "Infer End, dt ==>" << dt_infer << "\n";

    //     float dt_event;
    //     gpuErrChk(cudaEventElapsedTime(&dt_event, start, stop));
    //     gLogInfo << "Infer End, dt_event ==>" << dt_event << "\n";

    // }


    // void run(std::map<std::string, void*> inputBufferMap) {   

    //     gLogInfo << "H2D ... \n";
    //     const auto t0 = std::chrono::high_resolution_clock::now();
    //     float dt_d2h_all = 0;
    //     for (auto inName: mInputNames) {
    //         const auto t00 = std::chrono::high_resolution_clock::now();
    //         gpuErrChk(
    //             cudaMemcpyAsync(mDevBufferMap[inName], inputBufferMap[inName], mSizeMap[inName], cudaMemcpyHostToDevice, mStream));
    //         const auto t01 = std::chrono::high_resolution_clock::now();
    //         const float dt_each_h2d = std::chrono::duration<float, std::milli>(t01 - t00).count();
    //         // gLogInfo << "H2D: " <<  inName <<", dt ==>" << dt_each_h2d << "\n";
    //         dt_d2h_all += dt_each_h2d;
    //     }
    //     const auto t1 = std::chrono::high_resolution_clock::now();
    //     const float dt_h2d = std::chrono::duration<float, std::milli>(t1 - t0).count();
    //     gLogInfo << "H2D End, dt ==>" << dt_h2d << "\n";
    //     gLogInfo << "H2D End, dt self sum ==>" << dt_d2h_all << "\n";

    //     cudaEvent_t start, stop;
    //     gpuErrChk(cudaEventCreate(&start));
    //     gpuErrChk(cudaEventCreate(&stop));
    //     gpuErrChk(cudaEventRecord(start, mStream));
    //     gLogInfo << "Run ... \n";
    //     if (mEnableGraph) {
    //         // gLogInfo << "cudaGraphLaunch ... \n";
    //         gpuErrChk(cudaGraphLaunch(mExecGraph, mStream));
    //     } else {
    //         // gLogInfo << "enqueueV2 ... \n";
    //         bool status = mContext->enqueueV2(mBindings.data(), mStream, nullptr);
    //         if (!status) {
    //             gLogError << "Enqueue failed\n";
    //             exit(-1);
    //         }
    //     }
    //     gpuErrChk(cudaEventRecord(stop, mStream));
    //     const auto t2 = std::chrono::high_resolution_clock::now();
    //     const float dt_run = std::chrono::duration<float, std::milli>(t2 - t1).count();
    //     gLogInfo << "RUN End, dt ==>" << dt_run << "\n";

    //     // d2h
    //     gLogInfo << "D2H ... \n";
    //     for (auto outName: mOutputNames) {
    //         gpuErrChk(cudaMemcpyAsync(
    //             mHostBufferMap[outName], mDevBufferMap[outName], mSizeMap[outName], cudaMemcpyDeviceToHost, mStream));
    //     }
    //     const auto t3 = std::chrono::high_resolution_clock::now();
    //     const float dt_d2h = std::chrono::duration<float, std::milli>(t3 - t2).count();
    //     gLogInfo << "D2H End, dt ==>" << dt_d2h << "\n";
    //     // sync
    //     // gLogInfo << "cudaStreamSynchronize ... \n";
    //     gLogInfo << "Sync ... \n";
    //     gpuErrChk(cudaStreamSynchronize(mStream));
    //     const auto t4 = std::chrono::high_resolution_clock::now();
    //     const float dt_sync = std::chrono::duration<float, std::milli>(t4 - t3).count();
    //     gLogInfo << "Sync End, dt ==>" << dt_sync << "\n";

    //     const float dt_infer = std::chrono::duration<float, std::milli>(t4 - t0).count();
    //     gLogInfo << "Infer End, dt ==>" << dt_infer << "\n";

    //     float dt_event;
    //     gpuErrChk(cudaEventElapsedTime(&dt_event, start, stop));
    //     gLogInfo << "Infer End, dt_event ==>" << dt_event << "\n";

    // }

    ~TrtInference()
    {

        // gpuErrChk(cudaStreamDestroy(mStream));

        // for (auto iter = mDevBufferMap.begin(); iter != mDevBufferMap.end(); iter++) {
        //     gpuErrChk(cudaFree(iter->second));
        // }

        // for (auto iter = mHostBufferMap.begin(); iter != mHostBufferMap.end(); iter++) {
        //     free(iter->second);
        // }

        // if (mExecGraph) {
        //     cudaGraphExecDestroy(mExecGraph);
        // }
    }

    bool mEnableGraph{false};

    TrtUniquePtr<ICudaEngine> mEngine{nullptr};
    int mStreamNum{1};
    std::vector<std::unique_ptr<IExecutionContext>> mContexts; // N个context，用于管理上下文
    std::vector<std::unique_ptr<Bindings>> mBindings;
    std::vector<std::unique_ptr<Iteration<nvinfer1::IExecutionContext>>> mStreams; //N 个Iteration，用于执行推理

    // cudaStream_t mStream{NULL};
    std::vector<std::string> mInputNames;
    // std::vector<void*> mInputDevBufferList;
    // std::vector<void*> mInputHostBufferList;
    std::vector<size_t> mInputSizeList;
    std::vector<size_t> mInputElemSizeList;
    std::vector<std::vector<int>>mInputShapeList;
    std::vector<DataType> mInputDataTypeList;


    std::vector<std::string> mOutputNames;
    std::vector<size_t> mOutputSizeList;
    std::vector<size_t> mOutputElemSizeList;
    std::vector<std::vector<int>>mOutputShapeList;
    std::vector<DataType> mOutputDataTypeList;
    // std::vector<InferenceTrace> mLocalTrace;
    // SyncStruct mSync;



    // std::map<std::string, void*> mDevBufferMap;
    // std::map<std::string, size_t> mSizeMap;
    // std::map<std::string, DataType> mDataTypeMap;
    // std::map<std::string, void*> mHostBufferMap;
    // std::map<std::string, std::vector<int>> mShapeMap;
    // std::map<std::string, std::string> mDtypeMap;
    // std::map<std::string, std::string> mBindingTypeMap;
    
    // cudaGraphExec_t mExecGraph{};
};

#endif // INFER_C_TRT_INFER_H
