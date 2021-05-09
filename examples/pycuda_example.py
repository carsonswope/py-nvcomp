import pycuda.driver as cu
import pycuda.autoinit
import pycuda.gpuarray as cu_array

import numpy as np

import nvcomp

num_elements = 10000
d = np.zeros((num_elements,), dtype=np.int32)

# fill with whatever..
for i in range(num_elements):
    d[i] = (i % 4) * 3

d_size = d.size * d.itemsize
d_cu = cu_array.to_gpu(d)

compressor = nvcomp.CascadedCompressor('i4', 2, 1, True)
compressor_temp_size, compressor_output_max_size = compressor.configure(d_size)

print('input size:                 ', d_size)
print('compressor temp size:       ', compressor_temp_size)
print('compressor output max size: ', compressor_output_max_size)

compressor_temp_cu = cu_array.GPUArray((compressor_temp_size,), dtype=np.uint8) # just raw bytes
compressor_output_cu = cu_array.GPUArray((compressor_output_max_size,), dtype=np.uint8)

compressor_output_size = cu.pagelocked_zeros((1,), np.int64)

print('compressing..')

compressor.compress(
    d_cu.ptr,
    d_size,
    compressor_temp_cu.ptr,
    compressor_temp_size,
    compressor_output_cu.ptr,
    compressor_output_size.__array_interface__['data'][0])

cu.Context.synchronize()

print('compressor output size:     ', compressor_output_size[0])

decompressor = nvcomp.CascadedDecompressor()
decompressor_temp_size, decompressor_output_size = decompressor.configure(
    compressor_output_cu.ptr,
    compressor_output_size[0])

assert decompressor_output_size == d_size

decompressor_temp_cu = cu_array.GPUArray((decompressor_temp_size,), dtype=np.uint8)

# clear the data array, to be sure it's actually getting filled up by the decompressor
d_cu.fill(np.int32(0))

decompressor.decompress(
    compressor_output_cu.ptr,
    compressor_output_size[0],
    decompressor_temp_cu.ptr,
    decompressor_temp_size,
    d_cu.ptr,
    d_size)

d_decompressed = d_cu.get()

assert np.all(d == d_decompressed)
