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

#include "trt_infer.h"
#include "half.h"
#include "pybind11_numpy_scalar.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using float16 = half_float::half;
static_assert(sizeof(float16) == 2, "Bad size");


struct TrtInferenceRunner {
    TrtInferenceRunner(
        const std::string& enginePath, const bool enableGraph, const bool spin, const int overloap, const int batch, const int nStreams)
        : trt{enginePath, enableGraph, spin, overloap, batch, nStreams} {
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
        for (auto i=0; i<trt.mInputNames.size(); i++) {
            auto shape_ = trt.mInputShapeList[i];
            py::list shape;
            for (auto s: shape_) {
                shape.append(s);
            }
            ret[trt.mInputNames[i].c_str()] = shape;
        }
        return ret;
    }

    py::dict inputs_dtype() {
        py::dict ret;
        for (auto i=0; i<trt.mInputNames.size(); i++) {
            ret[trt.mInputNames[i].c_str()] = getDtype(trt.mInputDataTypeList[i]);
        }
        return ret;
    }

    py::dict inputs_size() {
        py::dict ret;
        for (auto i=0; i<trt.mInputNames.size(); i++) {
            ret[trt.mInputNames[i].c_str()] = trt.mInputSizeList[i];
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
        for (auto i=0; i<trt.mOutputNames.size(); i++) {
            auto shape_ = trt.mOutputShapeList[i];
            py::list shape;
            for (auto s: shape_) {
                shape.append(s);
            }
            ret[trt.mOutputNames[i].c_str()] = shape;
        }
        return ret;
    }

    py::dict outputs_dtype() {
        py::dict ret;
        for (auto i=0; i<trt.mOutputNames.size(); i++) {
            ret[trt.mOutputNames[i].c_str()] = getDtype(trt.mOutputDataTypeList[i]);
        }
        return ret;
    }

    py::dict outputs_size() {
        py::dict ret;
        for (auto i=0; i<trt.mOutputNames.size(); i++) {
            ret[trt.mOutputNames[i].c_str()] = trt.mOutputSizeList[i];
        }
        return ret;
    }

    py::list infer(const py::list &inputs) {

        std::vector<void*> inputs_buf;
        std::vector<int64_t> inputs_size;

        const auto t00 = std::chrono::high_resolution_clock::now();

        for (auto i = 0; i < trt.mInputNames.size(); i++) {
            auto tmp_input = py::cast<py::array>(inputs[i]);
            py::buffer_info tmp_buf = tmp_input.request();
            inputs_buf.emplace_back(tmp_buf.ptr);;
            inputs_size.emplace_back(tmp_input.nbytes());
            auto size = tmp_input.size();
        }

        const auto t01 = std::chrono::high_resolution_clock::now();
        const float dt_py_cast = std::chrono::duration<float, std::milli>(t01 - t00).count();
        // std::cout << "dt_py_cast: " << dt_py_cast << " ms" << std::endl;

        trt.infer(inputs_buf, inputs_size);

        // 查看inputs_buffer
        py::module np = py::module::import("numpy");
        py::list rets;
        for (auto i = 0; i < trt.mOutputNames.size(); i++) {
            py::list o_rets;
            DataType o_dtype = trt.mOutputDataTypeList[i];
            for (auto j = 0; j < trt.mStreamNum; j++) {
                void* o_buff = trt.mBindings[j]->getHostBuffer(trt.mOutputNames[i]);
                if (o_dtype == DataType::kFLOAT) {
                    auto ret = py::array_t<float>(trt.mOutputShapeList[i], (float*)o_buff);
                    o_rets.append(ret);
                } else if (o_dtype == DataType::kHALF) {
                    auto ret = py::array_t<float16>(trt.mOutputShapeList[i], (float16*)o_buff);
                    o_rets.append(ret);
                } else {
                    auto ret = py::array_t<int>(trt.mOutputShapeList[i], (int*)o_buff);
                    o_rets.append(ret);
                }
            }
            // concat by axis 0
            rets.append(np.attr("concatenate")(o_rets, 0));
        }

        return rets;
    }

    TrtInference trt;
};

PYBIND11_MODULE(trt_infer_c, m) {
    m.doc() = "Pybind11 plugin for Trt inference";

    py::class_<TrtInferenceRunner>(m, "trt_inf")
    .def(py::init<const std::string&, const bool, const bool, const int, const int, const int>(),
        py::arg("engine_path"),
        py::arg("use_cuda_graph") = false,
        py::arg("spin_wait") = false,
        py::arg("overloap") = 0,
        py::arg("batch") = 1,
        py::arg("stream_num") = 1)
    .def("infer", &TrtInferenceRunner::infer, "infer",
        py::arg("inputs")
    )
    .def("inputs_name", &TrtInferenceRunner::inputs_name, "get inputs name")
    .def("inputs_shape", &TrtInferenceRunner::inputs_shape, "get input_shape")
    .def("inputs_dtype", &TrtInferenceRunner::inputs_dtype, "get input_dtype")
    .def("inputs_size", &TrtInferenceRunner::inputs_size, "get input_size")
    .def("outputs_name", &TrtInferenceRunner::outputs_name, "get output_names")
    .def("outputs_shape", &TrtInferenceRunner::outputs_shape, "get output_shape")
    .def("outputs_dtype", &TrtInferenceRunner::outputs_dtype, "get output_dtype")
    .def("outputs_size", &TrtInferenceRunner::outputs_size, "get output_size");
}
