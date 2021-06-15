import pycuda.driver as cu
import pycuda.autoinit
import pycuda.gpuarray as cu_array

import numpy as np

import nvcomp

def main():
    BATCH_SIZE = 8
    BYTES_PER_BATCH = 1024

    s = cu.Stream()

    # make data
    d = np.random.randint(0, 8, (BATCH_SIZE, BYTES_PER_BATCH), dtype=np.uint8)
    d_cu = cu_array.to_gpu(d)
    d_ptrs = cu.pagelocked_zeros((BATCH_SIZE,), np.int64)
    for i in range(BATCH_SIZE):
        d_ptrs[i] = d_cu[i].__cuda_array_interface__['data'][0]
    d_bytes = cu.pagelocked_zeros((BATCH_SIZE,), np.int64)
    d_bytes[:] = BYTES_PER_BATCH

    # allocate temporary space
    temp_bytes = cu.pagelocked_zeros((1,), np.int64)
    nvcomp.batchedLZ4CompressGetTempSize(BATCH_SIZE, BYTES_PER_BATCH, temp_bytes)
    temp_cu = cu_array.GPUArray((temp_bytes[0]), dtype=np.uint8)

    # allocate output space
    max_compressed_bytes = cu.pagelocked_zeros((1,), np.int64)
    nvcomp.batchedLZ4CompressGetMaxOutputChunkSize(BYTES_PER_BATCH, max_compressed_bytes)
    compressed_cu = cu_array.GPUArray((BATCH_SIZE, max_compressed_bytes[0]), dtype=np.uint8)
    compressed_cu_ptrs = cu.pagelocked_zeros((BATCH_SIZE,), np.int64)
    compressed_cu_bytes = cu.pagelocked_zeros((BATCH_SIZE,), np.int64)
    for i in range(BATCH_SIZE):
        compressed_cu_ptrs[i] = compressed_cu[i].__cuda_array_interface__['data'][0]

    # compress!
    nvcomp.batchedLZ4CompressAsync(d_ptrs, d_bytes, BATCH_SIZE, temp_cu, temp_bytes[0], compressed_cu_ptrs, compressed_cu_bytes, stream=s.handle)
    s.synchronize()

    # decompress!
    d_cu.fill(0)
    nvcomp.batchedLZ4DecompressAsync(compressed_cu_ptrs, compressed_cu_bytes, BATCH_SIZE, d_ptrs, stream=s.handle)
    s.synchronize()

    # test :)
    d_decompressed = d_cu.get()
    assert(np.all(d_decompressed == d))
    print('Success!')

if __name__ == '__main__':
    main()