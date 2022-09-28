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

using namespace nvinfer1;

// 获取容量
inline int64_t volume(const nvinfer1::Dims& d) {
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

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



// 根据type获取占用空间
inline std::string getDtype(nvinfer1::DataType t) {
    switch (t) {
        case nvinfer1::DataType::kINT32: return "int";
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
        const int maxBatchSize = 1,
        const bool enableGraph = false,
        const int nStreams = 1
    ) : mEnableGraph(enableGraph) {   
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

        gLogInfo << "Loading Inference Engine ... \n";
        mEngine = loadEngine(enginePath);

        if (mEngine == nullptr) {
            gLogError << "Error loading engine\n";
            exit(-1);
        }
        gLogInfo << "Done ... \n";

        // 创建context
        gLogInfo << "Creating Context ... \n";
        mContext = TrtUniquePtr<IExecutionContext>(mEngine->createExecutionContext());
        if (!mContext)
        {
            gLogError << "Error creating execution context\n";
            exit(-1);
        }

        // 创建流
        gLogInfo << "Creating Stream ... \n";
        gpuErrChk(cudaStreamCreate(&mStream));

        gLogInfo << "Alloc Memory on Device & Host ... \n";
        allocateBindings(maxBatchSize);
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

    // 提前malloc 还是
    bool allocateBindings(const int maxBatchSize) {

        assert(mEngine != nullptr);
        
        for (int32_t b = 0; b < mEngine->getNbBindings(); ++b) {

            // auto dims = mContext->getBindingDimensions(b);
            auto dims = mEngine->getBindingDimensions(b);
            std::string name(mEngine->getBindingName(b));
            
            auto isInput = mEngine->bindingIsInput(b);
            nvinfer1::DataType dataType = mEngine->getBindingDataType(b);
            auto const *bindingInOutStr = isInput ? "input" : "output";
            
            auto size = volume(dims) * elementSize(dataType);
            auto elemSize = elementSize(dataType);
            void* devBuf = nullptr;
            gpuErrChk(cudaMalloc(&devBuf, size));
            gpuErrChk(cudaMemset(devBuf, 0, size));

            mDevBufferMap[name] = devBuf;
            mSizeMap[name] = size;
            mShapeMap[name] = getShape(dims);
            mDtypeMap[name] = getDtype(dataType);
            // 警惕 double free
            mBindings.emplace_back(devBuf);

            if (isInput) {
                mInputNames.emplace_back(name);
                mInputDevBufferList.emplace_back(devBuf);
                mInputSizeList.emplace_back(size);
                mInputShapeList.emplace_back(getShape(dims));
                mInputElemSizeList.emplace_back(elemSize);
                mInputDataTypeList.emplace_back(dataType);

            } else {
                void* hostBuf = nullptr;
                hostBuf = malloc(size);
                memset(hostBuf, 0, size);
                mHostBufferMap[name] = hostBuf;
                mOutputNames.emplace_back(name);
                mOutputDevBufferList.emplace_back(devBuf);
                mOutputHostBufferList.emplace_back(hostBuf);
                mOutputSizeList.emplace_back(size);
                mOutputShapeList.emplace_back(getShape(dims));
                mOutputElemSizeList.emplace_back(elemSize);
                mOutputDataTypeList.emplace_back(dataType);
            }
        }
    }

    void prepare(int profIdx) {

        if (!mContext->allInputDimensionsSpecified())
        {
            gLogError << "Not all input dimensions are specified for the exeuction context\n";
            exit(-1);
        }

        if (mEnableGraph)
        {
            cudaGraph_t graph;
            cudaGraphExec_t exec;
            // warm up and let mContext do cublas initialization
            bool status = mContext->enqueueV2(mBindings.data(), mStream, nullptr);
            if (!status)
            {
                gLogError << "Enqueue failed\n";
                exit(-1);
            }
            gLogVerbose << "Capturing graph\n";

            gpuErrChk(cudaStreamBeginCapture(mStream, cudaStreamCaptureModeRelaxed));
            status = mContext->enqueueV2(mBindings.data(), mStream, nullptr);
            if (!status)
            {
                gLogError << "Enqueue failed\n";
                exit(-1);
            }

            gpuErrChk(cudaStreamEndCapture(mStream, &graph));
            gpuErrChk(cudaStreamSynchronize(mStream));

            gpuErrChk(cudaGraphInstantiate(&exec, graph, NULL, NULL, 0));
            mExecGraph = exec;
        }
        // mCuSeqlens.resize(batchSize + 1);
        // std::generate(mCuSeqlens.begin(), mCuSeqlens.end(), [pos = -mSeqLength, this]() mutable{ pos += mSeqLength; return pos; });
    }


    void run2(std::vector<void*> inputBufferList) {   

        gLogInfo << "H2D ... \n";
        const auto t0 = std::chrono::high_resolution_clock::now();
        // float dt_d2h_all = 0;

        for (auto i = 0; i < mInputNames.size(); i++) {
            gpuErrChk(
                cudaMemcpyAsync(mInputDevBufferList[i], inputBufferList[i], mInputSizeList[i], cudaMemcpyHostToDevice, mStream));
        }


        // for (auto inName: mInputNames) {
        //     const auto t00 = std::chrono::high_resolution_clock::now();
        //     gpuErrChk(
        //         cudaMemcpyAsync(mDevBufferMap[inName], inputBufferMap[inName], mSizeMap[inName], cudaMemcpyHostToDevice, mStream));
        //     const auto t01 = std::chrono::high_resolution_clock::now();
        //     const float dt_each_h2d = std::chrono::duration<float, std::milli>(t01 - t00).count();
        //     // gLogInfo << "H2D: " <<  inName <<", dt ==>" << dt_each_h2d << "\n";
        //     dt_d2h_all += dt_each_h2d;
        // }
        const auto t1 = std::chrono::high_resolution_clock::now();
        const float dt_h2d = std::chrono::duration<float, std::milli>(t1 - t0).count();
        gLogInfo << "H2D End, dt ==>" << dt_h2d << "\n";

        cudaEvent_t start, stop;
        gpuErrChk(cudaEventCreate(&start));
        gpuErrChk(cudaEventCreate(&stop));
        gpuErrChk(cudaEventRecord(start, mStream));
        gLogInfo << "Run ... \n";
        if (mEnableGraph) {
            // gLogInfo << "cudaGraphLaunch ... \n";
            gpuErrChk(cudaGraphLaunch(mExecGraph, mStream));
        } else {
            // gLogInfo << "enqueueV2 ... \n";
            bool status = mContext->enqueueV2(mBindings.data(), mStream, nullptr);
            if (!status) {
                gLogError << "Enqueue failed\n";
                exit(-1);
            }
        }
        gpuErrChk(cudaEventRecord(stop, mStream));
        const auto t2 = std::chrono::high_resolution_clock::now();
        const float dt_run = std::chrono::duration<float, std::milli>(t2 - t1).count();
        gLogInfo << "RUN End, dt ==>" << dt_run << "\n";

        // d2h
        gLogInfo << "D2H ... \n";
        // for (auto outName: mOutputNames) {
        //     gpuErrChk(cudaMemcpyAsync(
        //         mHostBufferMap[outName], mDevBufferMap[outName], mSizeMap[outName], cudaMemcpyDeviceToHost, mStream));
        // }

        for (auto i = 0; i < mOutputNames.size(); i++) {
            gpuErrChk(
                cudaMemcpyAsync(mOutputHostBufferList[i], mOutputDevBufferList[i], mOutputSizeList[i], cudaMemcpyDeviceToHost, mStream));
        }


        const auto t3 = std::chrono::high_resolution_clock::now();
        const float dt_d2h = std::chrono::duration<float, std::milli>(t3 - t2).count();
        gLogInfo << "D2H End, dt ==>" << dt_d2h << "\n";
        // sync
        // gLogInfo << "cudaStreamSynchronize ... \n";
        gLogInfo << "Sync ... \n";
        gpuErrChk(cudaStreamSynchronize(mStream));
        const auto t4 = std::chrono::high_resolution_clock::now();
        const float dt_sync = std::chrono::duration<float, std::milli>(t4 - t3).count();
        gLogInfo << "Sync End, dt ==>" << dt_sync << "\n";

        const float dt_infer = std::chrono::duration<float, std::milli>(t4 - t0).count();
        gLogInfo << "Infer End, dt ==>" << dt_infer << "\n";

        float dt_event;
        gpuErrChk(cudaEventElapsedTime(&dt_event, start, stop));
        gLogInfo << "Infer End, dt_event ==>" << dt_event << "\n";

    }


    void run(std::map<std::string, void*> inputBufferMap) {   

        gLogInfo << "H2D ... \n";
        const auto t0 = std::chrono::high_resolution_clock::now();
        float dt_d2h_all = 0;
        for (auto inName: mInputNames) {
            const auto t00 = std::chrono::high_resolution_clock::now();
            gpuErrChk(
                cudaMemcpyAsync(mDevBufferMap[inName], inputBufferMap[inName], mSizeMap[inName], cudaMemcpyHostToDevice, mStream));
            const auto t01 = std::chrono::high_resolution_clock::now();
            const float dt_each_h2d = std::chrono::duration<float, std::milli>(t01 - t00).count();
            // gLogInfo << "H2D: " <<  inName <<", dt ==>" << dt_each_h2d << "\n";
            dt_d2h_all += dt_each_h2d;
        }
        const auto t1 = std::chrono::high_resolution_clock::now();
        const float dt_h2d = std::chrono::duration<float, std::milli>(t1 - t0).count();
        gLogInfo << "H2D End, dt ==>" << dt_h2d << "\n";
        gLogInfo << "H2D End, dt self sum ==>" << dt_d2h_all << "\n";

        cudaEvent_t start, stop;
        gpuErrChk(cudaEventCreate(&start));
        gpuErrChk(cudaEventCreate(&stop));
        gpuErrChk(cudaEventRecord(start, mStream));
        gLogInfo << "Run ... \n";
        if (mEnableGraph) {
            // gLogInfo << "cudaGraphLaunch ... \n";
            gpuErrChk(cudaGraphLaunch(mExecGraph, mStream));
        } else {
            // gLogInfo << "enqueueV2 ... \n";
            bool status = mContext->enqueueV2(mBindings.data(), mStream, nullptr);
            if (!status) {
                gLogError << "Enqueue failed\n";
                exit(-1);
            }
        }
        gpuErrChk(cudaEventRecord(stop, mStream));
        const auto t2 = std::chrono::high_resolution_clock::now();
        const float dt_run = std::chrono::duration<float, std::milli>(t2 - t1).count();
        gLogInfo << "RUN End, dt ==>" << dt_run << "\n";

        // d2h
        gLogInfo << "D2H ... \n";
        for (auto outName: mOutputNames) {
            gpuErrChk(cudaMemcpyAsync(
                mHostBufferMap[outName], mDevBufferMap[outName], mSizeMap[outName], cudaMemcpyDeviceToHost, mStream));
        }
        const auto t3 = std::chrono::high_resolution_clock::now();
        const float dt_d2h = std::chrono::duration<float, std::milli>(t3 - t2).count();
        gLogInfo << "D2H End, dt ==>" << dt_d2h << "\n";
        // sync
        // gLogInfo << "cudaStreamSynchronize ... \n";
        gLogInfo << "Sync ... \n";
        gpuErrChk(cudaStreamSynchronize(mStream));
        const auto t4 = std::chrono::high_resolution_clock::now();
        const float dt_sync = std::chrono::duration<float, std::milli>(t4 - t3).count();
        gLogInfo << "Sync End, dt ==>" << dt_sync << "\n";

        const float dt_infer = std::chrono::duration<float, std::milli>(t4 - t0).count();
        gLogInfo << "Infer End, dt ==>" << dt_infer << "\n";

        float dt_event;
        gpuErrChk(cudaEventElapsedTime(&dt_event, start, stop));
        gLogInfo << "Infer End, dt_event ==>" << dt_event << "\n";

    }

    ~TrtInference()
    {

        gpuErrChk(cudaStreamDestroy(mStream));

        for (auto iter = mDevBufferMap.begin(); iter != mDevBufferMap.end(); iter++) {
            gpuErrChk(cudaFree(iter->second));
        }

        for (auto iter = mHostBufferMap.begin(); iter != mHostBufferMap.end(); iter++) {
            free(iter->second);
        }

        if (mExecGraph) {
            cudaGraphExecDestroy(mExecGraph);
        }
    }

    // static const int kBERT_INPUT_NUM = 3;

    // const int mSeqLength;
    const bool mEnableGraph;

    TrtUniquePtr<ICudaEngine> mEngine{nullptr};
    TrtUniquePtr<IExecutionContext> mContext{nullptr};
    
    int mStreamNum = 1;
    // std::vector<std::unique_ptr<nvinfer1::IExecutionContext>> mContexts;
    
    std::vector<void*> mBindings;
    // bool mEnableVariableLen;
    std::vector<int> mCuSeqlens;

    cudaStream_t mStream{NULL};
    // std::vector<void*> mDeviceBuffers;
    // std::vector<float> mHostOutput;
    
    // size_t mOutputSize;
    // std::vector<int> mOutputDims;

    // std::vector<std::vector<float>> mTimes;

    // std::vector<void*> mInputDeviceBuffers;
    // std::vector<char*> mInputNames;
    // std::vector<size_t> mInputSizes;

    // std::vector<void*> mOutputDeviceBuffers;
    // std::vector<char*> mOutputNames;
    // std::vector<size_t> mOutputSizes;
    // std::vector<void*> mOutputHostBuffers;

    
    

    std::vector<std::string> mInputNames;
    std::vector<void*> mInputDevBufferList;
    std::vector<void*> mInputHostBufferList;
    std::vector<size_t> mInputSizeList;
    std::vector<size_t> mInputElemSizeList;
    std::vector<std::vector<int>>mInputShapeList;
    std::vector<DataType> mInputDataTypeList;


    std::vector<std::string> mOutputNames;
    std::vector<void*> mOutputDevBufferList;
    std::vector<void*> mOutputHostBufferList;
    std::vector<size_t> mOutputSizeList;
    std::vector<size_t> mOutputElemSizeList;
    std::vector<std::vector<int>>mOutputShapeList;
    std::vector<DataType> mOutputDataTypeList;



    std::map<std::string, void*> mDevBufferMap;
    std::map<std::string, size_t> mSizeMap;
    std::map<std::string, DataType> mDataTypeMap;
    std::map<std::string, void*> mHostBufferMap;
    std::map<std::string, std::vector<int>> mShapeMap;
    std::map<std::string, std::string> mDtypeMap;
    std::map<std::string, std::string> mBindingTypeMap;
    
    cudaGraphExec_t mExecGraph{};
};

#endif // INFER_C_TRT_INFER_H
