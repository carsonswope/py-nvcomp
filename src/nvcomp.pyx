from libc.stdint cimport uintptr_t
from libcpp cimport bool

from nvcomp cimport cudaStream_t, nvcompType_t, _CascadedCompressor, _CascadedDecompressor

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
