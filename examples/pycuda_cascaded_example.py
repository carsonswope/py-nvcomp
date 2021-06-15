import pycuda.driver as cu
import pycuda.autoinit
import pycuda.gpuarray as cu_array

import numpy as np

import nvcomp
from nvcomp import nvcompType_t

def main():
    num_elements = 10000
    d = np.zeros((num_elements,), dtype=np.int32)
    for i in range(num_elements):
        d[i] = (i % 4) * 3  # fill with whatever..

    d_size = d.size * d.itemsize
    d_cu = cu_array.to_gpu(d)

    compressor = nvcomp.CascadedCompressor(nvcompType_t.NVCOMP_TYPE_INT, 2, 1, True)

    # compressor.configure is fully synchronous (presumably fully host-side code), no need for pagelock
    compressor_temp_size = np.zeros((1,), dtype=np.int64)
    compressor_max_output_size = np.zeros((1,), dtype=np.int64)
    compressor.configure(
        d_size,
        compressor_temp_size,
        compressor_max_output_size)

    print('input size:                 ', d_size)
    print('compressor temp size:       ', compressor_temp_size[0])
    print('compressor output max size: ', compressor_max_output_size[0])

    compressor_temp_cu = cu_array.GPUArray((compressor_temp_size[0],), dtype=np.uint8) # just raw bytes
    compressor_output_cu = cu_array.GPUArray((compressor_max_output_size[0],), dtype=np.uint8)

    compressor_output_size = cu.pagelocked_zeros((1,), np.int64)

    s = cu.Stream()

    compressor.compress_async(
        d_cu,
        d_size,
        compressor_temp_cu,
        compressor_temp_size,
        compressor_output_cu,
        compressor_output_size,
        stream=s.handle)

    s.synchronize()

    print('compressor output size:     ', compressor_output_size[0])

    decompressor = nvcomp.CascadedDecompressor()
    decompressor_temp_size = cu.pagelocked_zeros((1,), np.int64)
    decompressor_output_size = cu.pagelocked_zeros((1,), np.int64)
    decompressor.configure(
        compressor_output_cu,
        compressor_output_size[0],
        decompressor_temp_size,
        decompressor_output_size,
        stream=s.handle)

    s.synchronize()

    print('decompressor temp size:     ', decompressor_temp_size[0])

    assert decompressor_output_size[0] == d_size

    decompressor_temp_cu = cu_array.GPUArray((decompressor_temp_size[0],), dtype=np.uint8)

    # clear the data array, to be sure it's actually getting filled up by the decompressor
    d_cu.fill(np.int32(0))

    decompressor.decompress_async(
        compressor_output_cu,
        compressor_output_size[0],
        decompressor_temp_cu,
        decompressor_temp_size[0],
        d_cu,
        d_size,
        stream=s.handle)

    s.synchronize()

    d_decompressed = d_cu.get()

    assert np.all(d == d_decompressed)

    print('Decompression success!')

if __name__ == '__main__':
    main()
