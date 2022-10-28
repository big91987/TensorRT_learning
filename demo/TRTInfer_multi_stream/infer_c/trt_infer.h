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


// 一个binging，及buffer
struct Binding
{
    bool isInput{false};
    std::unique_ptr<IMirroredBuffer> buffer;
    int64_t volume{0};
    nvinfer1::DataType dataType{nvinfer1::DataType::kFLOAT};

    // 从文件填充
    void fill(std::string const& fileName)
    {
        loadFromFile(fileName, static_cast<char*>(buffer->getHostBuffer()), buffer->getSize());
    }

    // 从内存拷贝到锁页内存
    void fill(void* data)
    {
        memcpy(static_cast<char*>(buffer->getHostBuffer()), data, buffer->getSize());
    }

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

    void* getHostBuffer(int bindingIndex) {
        return mBindings[bindingIndex].buffer->getHostBuffer();
    }

    void* getHostBuffer(std::string const& name) {
        return mBindings[mNames[name]].buffer->getHostBuffer();
    }

    void* getDeviceBuffer(int bindingIndex) {
        return mBindings[bindingIndex].buffer->getDeviceBuffer();
    }

    void* getDeviceBuffer(std::string const& name) {
        return mBindings[mNames[name]].buffer->getDeviceBuffer();
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


    void createEnqueueFunction(
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

        setUp(enginePath, nStreams, batch, overloap, spin, enableGraph);

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
            auto set_ptr = static_cast<char*>(data);
            setBufOneBinding(static_cast<char*>(data) + i * bindingSize, i, bindingIndex);
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
        // std::cout << "H2D ... split each input to streams\n";
        const auto t0 = std::chrono::high_resolution_clock::now();

        if (mInputNames.size() != inputBufferList.size()) {
            gLogError << "input buf size != engine inputs num" ;
            return;
        }

        for (auto i=0; i < inputBufferList.size(); i++) {
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
            // s->sync(sync.cpuStart, sync.gpuStart, skipTransfers);
            s->syncAll(sync.cpuStart, sync.gpuStart, skipTransfers);
        }   
    }

    ~TrtInference()
    {
    }

    bool mEnableGraph{false};
    TrtUniquePtr<ICudaEngine> mEngine{nullptr};
    int mStreamNum{1};
    std::vector<std::unique_ptr<IExecutionContext>> mContexts; // N个context，用于管理上下文
    std::vector<std::unique_ptr<Bindings>> mBindings;
    std::vector<std::unique_ptr<Iteration<nvinfer1::IExecutionContext>>> mStreams; //N 个Iteration，用于执行推理
    std::vector<std::string> mInputNames;
    std::vector<size_t> mInputSizeList;
    std::vector<size_t> mInputElemSizeList;
    std::vector<std::vector<int>>mInputShapeList;
    std::vector<DataType> mInputDataTypeList;
    std::vector<std::string> mOutputNames;
    std::vector<size_t> mOutputSizeList;
    std::vector<size_t> mOutputElemSizeList;
    std::vector<std::vector<int>>mOutputShapeList;
    std::vector<DataType> mOutputDataTypeList;
};

#endif // INFER_C_TRT_INFER_H
