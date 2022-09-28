import sys
import numpy as np
from functools import reduce
import time


sys.path.insert(0, '/home/TensorRT/demo/TRTInfer/build/')
import trt_infer_c


def make_random_input(inputs_shape, inputs_dtype, data_parall=1):
    input_dict = {}
    for i_name in inputs_shape.keys():
        i_shape = inputs_shape[i_name]
        i_dtype = inputs_dtype[i_name]
        shape_size = reduce(lambda x, y : x * y, list(inputs_shape[i_name]))
        one_batch_data = np.random.rand(shape_size).astype(i_dtype).reshape(i_shape)
        input_dict[i_name] = np.repeat(one_batch_data, data_parall, axis=0)
    return input_dict


if __name__ == '__main__':
    batch = 1024
    obj_path = '/home/gpu/ctr_er.trt'
    use_cuda_graph = True

    # trt exec true 150w  false 60w 多线程没啥用 
    # cpp: true 62w false 47w 
    # gpu_inference: 71w false
    # 目前看 py数据转换比较浪费时间，但是 stc_infer也是如此

    update_span = 0.5
    total_epochs = 10000
    
    # aa  = trt_infer_c.trt_inf(engine_path='/runner_cache/gpu/resnet34.trt', max_batch_size=1, use_cuda_graph=True)
    aa  = trt_infer_c.trt_inf(engine_path=obj_path, max_batch_size=1, use_cuda_graph=use_cuda_graph)

    aa.prepare()
    
    input_dict = make_random_input(aa.inputs_shape(), aa.inputs_dtype())

    inputs = [input_dict[x] for x in aa.inputs_name()]

    # print(aa.inputs_name())

    t_last_update = time.time()
    for i in range(total_epochs):

        print('\n\nepoch:      {}/{}'.format(i+1, total_epochs))
        t_loop_start = time.time()
        rets = aa.run2(inputs)
        t_loop_end = time.time()
        dt = t_loop_end - t_loop_start
        print('dt ===> {}'.format(t_loop_end - t_loop_start))

        # print('epoch:      {}/{}'.format(i+1, total_epochs))
        print('latency:    {}'.format(dt))
        print('throughput: {}'.format(batch / dt))

        # if t_loop_end - t_last_update >= update_span:
        #     dt = t_loop_end - t_loop_start
        #     print('epoch:      {}/{}'.format(i+1, total_epochs))
        #     print('latency:    {}'.format(dt))
        #     print('throughput: {}'.format(batch / dt))
        #     t_last_update = t_loop_end