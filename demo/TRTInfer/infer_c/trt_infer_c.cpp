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

// #include "bert_infer.h"
#include "trt_infer.h"
#include "half.h"
#include "pybind11_numpy_scalar.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using float16 = half_float::half;
static_assert(sizeof(float16) == 2, "Bad size");

// void py_cast

struct TrtInferenceRunner {
    TrtInferenceRunner(
        const std::string& enginePath, const int maxBatchSize, const bool enableGraph)
        : trt{enginePath, maxBatchSize, enableGraph} {
    }

    void prepare() {
        trt.prepare(0);
    }


    py::list inputs_name() {
        py::list ret;
        for (auto input_name: trt.mInputNames) {
            ret.append(input_name);
        }
        return ret;
    }


    py::dict inputs_shape() {
        py::dict ret;
        for (auto input_name: trt.mInputNames) {
            auto shape_ = trt.mShapeMap[input_name];
            py::list shape;
            for (auto s: shape_) {
                shape.append(s);
            }
            ret[input_name.c_str()] = shape;
        }
        return ret;
    }


    py::dict inputs_dtype() {
        py::dict ret;
        for (auto input_name: trt.mInputNames) {
            auto shape_ = trt.mShapeMap[input_name];
            ret[input_name.c_str()] = trt.mDtypeMap[input_name];
        }
        return ret;
    }


    py::list outputs_name() {
        py::list ret;
        for (auto output_name: trt.mOutputNames) {
            ret.append(output_name);
        }
        return ret;
    }

    py::dict outputs_shape() {
        py::dict ret;
        for (auto output_name: trt.mOutputNames) {
            auto shape_ = trt.mShapeMap[output_name];
            py::list shape;
            for (auto s: shape_) {
                shape.append(s);
            }
            ret[output_name.c_str()] = shape;
        }
        return ret;
    }

    py::dict outputs_dtype() {
        py::dict ret;
        for (auto output_name: trt.mOutputNames) {
            auto shape_ = trt.mShapeMap[output_name];
            ret[output_name.c_str()] = trt.mDtypeMap[output_name];
        }
        return ret;
    }

    int run(const py::dict &input_dict) {

        std::map<std::string, void*> inputs_buf;

        // std::cout << "!!!!!!!!!!!!!!!??????????????????!!!!"<< std::endl;
        const auto t00 = std::chrono::high_resolution_clock::now();
        for (auto input_name: trt.mInputNames) {
            // std::cout << "!!!!!!!!!!!!!!!!!!!"<< std::endl;
            // std::cout << input_name << std::endl;
            // std::cout << input_dict.contains(input_name.c_str()) << std::endl;
            // std::cout << input_dict[input_name.c_str()].is_none() << std::endl;
            // std::cout << (input_dict.contains(input_name.c_str()) && !input_dict[input_name.c_str()].is_none()) << std::endl;
            if (input_dict.contains(input_name.c_str()) && !input_dict[input_name.c_str()].is_none()) {
                // // 取出 numpy array
                auto tmp_input = py::cast<py::array>(input_dict[input_name.c_str()]);
                py::buffer_info input_buf = tmp_input.request();
                auto input_file_size = tmp_input.nbytes();
                auto input_elem_size = tmp_input.itemsize();
                // auto input_type = tmp_input.dtype().char_();
                auto input_shape_size = input_file_size / input_elem_size;

                // // TODO 增加 dtype校验，判断是否和 engings中的一致
                inputs_buf[input_name] = input_buf.ptr;


                // inputs_buf[input_name] = (void*)input_dict[input_name.c_str()];
                // std::cout << "aaaaaaaaaaaaaaaaaaaaaa"<< std::endl;

            } else {
                std::cout << "can not find input name: " << input_name << " in input_dict" << std::endl;
                exit(1);
            }
        }
        const auto t01 = std::chrono::high_resolution_clock::now();
        const float dt_py_cast = std::chrono::duration<float, std::milli>(t01 - t00).count();
        std::cout << "dt_py_cast: " << dt_py_cast << " ms" << std::endl;
        // auto input = py::cast<py::array>(input_dict[input_name.c_str()]);

        trt.run(inputs_buf);
        return 0;

        //TODO 取回结果
    }


    py::list run2(const py::list &inputs) {

        std::vector<void*> inputs_buf;

        // std::cout << "!!!!!!!!!!!!!!!??????????????????!!!!"<< std::endl;
        const auto t00 = std::chrono::high_resolution_clock::now();

        for (auto i = 0; i < trt.mInputNames.size(); i++) {
            // auto tmp_input = py::cast<py::array>(inputs[i])
            auto tmp_input = py::cast<py::array>(inputs[i]);
            py::buffer_info tmp_buf = tmp_input.request();
            inputs_buf.emplace_back(tmp_buf.ptr);
        }

        const auto t01 = std::chrono::high_resolution_clock::now();
        const float dt_py_cast = std::chrono::duration<float, std::milli>(t01 - t00).count();
        std::cout << "dt_py_cast: " << dt_py_cast << " ms" << std::endl;

        // auto input = py::cast<py::array>(input_dict[input_name.c_str()]);
        trt.run2(inputs_buf);

        py::module np = py::module::import("numpy");
        py::list rets;
        for (auto i = 0; i < trt.mOutputNames.size(); i++) {
            // auto ret = py::cast<py::array_t<float>>(trt.mOutputHostBufferList[i]).attr("reshape")(trt.mOutputShapeList[i]);
            // rets.append(ret);
            DataType o_dtype = trt.mOutputDataTypeList[i];
            auto o_elem_count = trt.mOutputElemSizeList[i];
            if (o_dtype == DataType::kFLOAT) {
                auto ret = py::array_t<float>(trt.mOutputShapeList[i], (float*) trt.mOutputHostBufferList[i]);
                rets.append(ret); 
            } else if (o_dtype == DataType::kHALF) {
                auto ret = py::array_t<float16>(trt.mOutputShapeList[i], (float16*) trt.mOutputHostBufferList[i]);
                rets.append(ret);
            } else {
                auto ret = py::array_t<int>(trt.mOutputShapeList[i], (int*) trt.mOutputHostBufferList[i]);
                rets.append(ret);
            }
        }

        return rets;

        // return 0;

        //TODO 取回结果
    }

    TrtInference trt;
};

PYBIND11_MODULE(trt_infer_c, m) {
    m.doc() = "Pybind11 plugin for Trt inference";

    py::class_<TrtInferenceRunner>(m, "trt_inf")
    .def(py::init<const std::string&, const int, const bool>(),
        py::arg("engine_path"),
        py::arg("max_batch_size") = 1,
        py::arg("use_cuda_graph") = false)
    .def("run", &TrtInferenceRunner::run, "infer",
        py::arg("inputs_dict")
    )
    .def("run2", &TrtInferenceRunner::run2, "infer",
        py::arg("inputs")
    )
    .def("prepare", &TrtInferenceRunner::prepare, "prepare && warmup")
    .def("inputs_name", &TrtInferenceRunner::inputs_name, "get inputs name")
    .def("inputs_shape", &TrtInferenceRunner::inputs_shape, "get input_shape")
    .def("inputs_dtype", &TrtInferenceRunner::inputs_dtype, "get input_dtype")
    .def("outputs_name", &TrtInferenceRunner::outputs_name, "get output_names")
    .def("outputs_shape", &TrtInferenceRunner::outputs_shape, "get output_shape")
    .def("outputs_dtype", &TrtInferenceRunner::outputs_dtype, "get output_dtype");
    
}
