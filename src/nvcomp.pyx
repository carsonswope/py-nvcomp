from libc.stdint cimport uintptr_t
from libcpp cimport bool

from nvcomp cimport cudaStream_t, nvcompType_t, \
    _LZ4Compressor, _LZ4Decompressor, \
    _CascadedCompressor, _CascadedDecompressor, \
    _nvcompBatchedLZ4DecompressAsync # low-level API

cpdef __get_array_interface_ptr(a):
    return a.__array_interface__['data'][0]

cpdef __get_cuda_array_interface_ptr(a):
    return a.__cuda_array_interface__['data'][0]

# could be either __array_interface__ or __cuda_array_interface__
cpdef __get_ptr(a):
    # this has to be slow though... is there a better way? try/catch?
    d = a.__dir__()
    if '__array_interface__' in d:
        return __get_array_interface_ptr(a)
    elif '__cuda_array_interface__' in d:
        return __get_cuda_array_interface_ptr(a)
    else:
        raise AttributeError('Argument does not implement __cuda_array_interface__ or __array_interface__')

# LZ4 Compressor / Decompressor
cdef class LZ4Compressor:
    cdef _LZ4Compressor* c

    def __cinit__(self, size_t chunk_size=0):
        self.c = new _LZ4Compressor(chunk_size)
    
    def __dealloc__(self):
        del self.c
    
    def configure(self, in_bytes, temp_bytes, out_bytes):
        cdef uintptr_t temp_bytes_ptr = __get_array_interface_ptr(temp_bytes)
        cdef uintptr_t out_bytes_ptr = __get_array_interface_ptr(out_bytes)
        self.c.configure(
            <size_t>in_bytes,
            <size_t*>temp_bytes_ptr,
            <size_t*>out_bytes_ptr)

    def compress_async(self, in_arr, in_bytes, temp_arr, temp_bytes, out_arr, out_bytes, uintptr_t stream = 0):
        cdef uintptr_t in_ptr = __get_cuda_array_interface_ptr(in_arr)
        cdef uintptr_t temp_ptr = __get_cuda_array_interface_ptr(temp_arr)
        cdef uintptr_t out_ptr = __get_cuda_array_interface_ptr(out_arr)
        cdef uintptr_t out_bytes_ptr = __get_ptr(out_bytes)
        self.c.compress_async(
            <void*>in_ptr,
            <size_t>in_bytes,
            <void*>temp_ptr,
            <size_t>temp_bytes,
            <void*>out_ptr,
            <size_t*>out_bytes_ptr,
            <cudaStream_t>stream)

cdef class LZ4Decompressor:
    cdef _LZ4Decompressor* d

    def __cinit__(self):
        self.d = new _LZ4Decompressor()

    def __dealloc__(self):
        del self.d

    cpdef configure(self, in_arr, in_bytes, temp_bytes, out_bytes, uintptr_t stream = 0):
        cdef uintptr_t in_ptr = __get_cuda_array_interface_ptr(in_arr)
        cdef uintptr_t temp_bytes_ptr = __get_ptr(temp_bytes)
        cdef uintptr_t out_bytes_ptr = __get_ptr(out_bytes)
        self.d.configure(
            <void*>in_ptr,
            <size_t>in_bytes,
            <size_t*>temp_bytes_ptr,
            <size_t*>out_bytes_ptr,
            <cudaStream_t>stream)

    def decompress_async(self, in_arr, in_bytes, temp_arr, temp_bytes, out_arr, out_bytes, uintptr_t stream = 0):
        cdef uintptr_t in_ptr = __get_cuda_array_interface_ptr(in_arr)
        cdef uintptr_t temp_ptr = __get_cuda_array_interface_ptr(temp_arr)
        cdef uintptr_t out_ptr = __get_cuda_array_interface_ptr(out_arr)
        self.d.decompress_async(
            <void*>in_ptr,
            <size_t>in_bytes,
            <void*>temp_ptr,
            <size_t>temp_bytes,
            <void*>out_ptr,
            <size_t>out_bytes,
            <cudaStream_t>stream)

# Cascaded Compressor / Decompressor
cdef class CascadedCompressor:
    cdef _CascadedCompressor* c

    def __cinit__(self, nvcompType_t t, int num_RLEs, int num_deltas, bool use_bp):
        self.c = new _CascadedCompressor(t, num_RLEs, num_deltas, use_bp)

    def __dealloc__(self):
        del self.c
    
    def configure(self, in_bytes, temp_bytes, out_bytes):
        cdef uintptr_t temp_bytes_ptr = __get_array_interface_ptr(temp_bytes)
        cdef uintptr_t out_bytes_ptr = __get_array_interface_ptr(out_bytes)
        self.c.configure(
            <size_t>in_bytes,
            <size_t*>temp_bytes_ptr,
            <size_t*>out_bytes_ptr)

    def compress_async(self, in_arr, in_bytes, temp_arr, temp_bytes, out_arr, out_bytes, uintptr_t stream = 0):
        cdef uintptr_t in_ptr = __get_cuda_array_interface_ptr(in_arr)
        cdef uintptr_t temp_ptr = __get_cuda_array_interface_ptr(temp_arr)
        cdef uintptr_t out_ptr = __get_cuda_array_interface_ptr(out_arr)
        cdef uintptr_t out_bytes_ptr = __get_ptr(out_bytes)
        self.c.compress_async(
            <void*>in_ptr,
            <size_t>in_bytes,
            <void*>temp_ptr,
            <size_t>temp_bytes,
            <void*>out_ptr,
            <size_t*>out_bytes_ptr,
            <cudaStream_t>stream)

cdef class CascadedDecompressor:
    cdef _CascadedDecompressor* d

    def __cinit__(self):
        self.d = new _CascadedDecompressor()

    def __dealloc__(self):
        del self.d

    cpdef configure(self, in_arr, in_bytes, temp_bytes, out_bytes, uintptr_t stream = 0):
        cdef uintptr_t in_ptr = __get_cuda_array_interface_ptr(in_arr)
        cdef uintptr_t temp_bytes_ptr = __get_ptr(temp_bytes)
        cdef uintptr_t out_bytes_ptr = __get_ptr(out_bytes)
        self.d.configure(
            <void*>in_ptr,
            <size_t>in_bytes,
            <size_t*>temp_bytes_ptr,
            <size_t*>out_bytes_ptr,
            <cudaStream_t>stream)

    def decompress_async(self, in_arr, in_bytes, temp_arr, temp_bytes, out_arr, out_bytes, uintptr_t stream = 0):
        cdef uintptr_t in_ptr = __get_cuda_array_interface_ptr(in_arr)
        cdef uintptr_t temp_ptr = __get_cuda_array_interface_ptr(temp_arr)
        cdef uintptr_t out_ptr = __get_cuda_array_interface_ptr(out_arr)
        self.d.decompress_async(
            <void*>in_ptr,
            <size_t>in_bytes,
            <void*>temp_ptr,
            <size_t>temp_bytes,
            <void*>out_ptr,
            <size_t>out_bytes,
            <cudaStream_t>stream)

# Batched LZ4
cpdef batchedLZ4CompressGetTempSize(batch_size, max_uncompressed_chunk_bytes, temp_bytes):
    cdef uintptr_t temp_bytes_ptr = __get_ptr(temp_bytes)
    _nvcompBatchedLZ4CompressGetTempSize(
        <size_t> batch_size,
        <size_t> max_uncompressed_chunk_bytes,
        <size_t*> temp_bytes_ptr)

cpdef batchedLZ4CompressGetMaxOutputChunkSize(max_uncompressed_chunk_bytes, max_compressed_bytes):
    cdef uintptr_t max_compressed_bytes_ptr = __get_ptr(max_compressed_bytes)
    _nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
        <size_t> max_uncompressed_chunk_bytes,
        <size_t*>max_compressed_bytes_ptr)

cpdef batchedLZ4CompressAsync(device_in_ptrs_arr, device_in_bytes_arr, batch_size, temp_arr, temp_bytes, device_out_ptrs_arr, device_out_bytes_arr, uintptr_t stream=0):
    cdef uintptr_t device_in_ptrs_ptr = __get_ptr(device_in_ptrs_arr)
    cdef uintptr_t device_in_bytes_ptr = __get_ptr(device_in_bytes_arr)
    cdef uintptr_t device_out_ptrs_ptr = __get_ptr(device_out_ptrs_arr)
    cdef uintptr_t device_out_bytes_ptr = __get_ptr(device_out_bytes_arr)
    cdef uintptr_t temp_ptr = __get_ptr(temp_arr)
    _nvcompBatchedLZ4CompressAsync(
        <void**>device_in_ptrs_ptr,
        <size_t*>device_in_bytes_ptr,
        0, # max_uncompressed_chunk_bytes
        <size_t>batch_size,
        <void*>temp_ptr,
        <size_t>temp_bytes,
        <void**>device_out_ptrs_ptr,
        <size_t*>device_out_bytes_ptr,
        <cudaStream_t>stream)

cpdef batchedLZ4DecompressAsync(device_in_ptrs_arr, device_in_bytes_arr, batch_size, device_out_ptrs_arr, uintptr_t stream=0):
    cdef uintptr_t device_in_ptrs_ptr = __get_ptr(device_in_ptrs_arr)
    cdef uintptr_t device_in_bytes_ptr = __get_ptr(device_in_bytes_arr)
    cdef uintptr_t device_out_ptrs_ptr = __get_ptr(device_out_ptrs_arr)
    
    # device_out_bytes and temp_ptr arguments to nvcompBatchedLZ4DecompressAsync are unused but CudaUtils::device_pointer() checks
    # to make sure that they are valid CUDA pointers, so need to pass something to those arguments that isn't nullptr
    cdef uintptr_t _unused_ptr = (<uintptr_t*>device_in_ptrs_ptr)[0]

    _nvcompBatchedLZ4DecompressAsync(
        <void**>device_in_ptrs_ptr,
        <size_t*>device_in_bytes_ptr,
        <size_t*>_unused_ptr, # device_out_bytes
        0, # max_uncompressed_chunk_bytes
        <size_t>batch_size,
        <void*>_unused_ptr, # temp ptr
        0, # temp bytes
        <void**>device_out_ptrs_ptr,
        <cudaStream_t>stream)
