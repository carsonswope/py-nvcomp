from libcpp cimport bool

from nvcomp cimport cudaStream_t, nvcompType_t, _CascadedCompressor, _CascadedDecompressor

ctypedef long long ptr_t

cdef class CascadedCompressor:
    cdef _CascadedCompressor* c

    def __cinit__(self, nvcompType_t t, int num_RLEs, int num_deltas, bool use_bp):
        self.c = new _CascadedCompressor(t, num_RLEs, num_deltas, use_bp)

    def __dealloc__(self):
        del self.c
    
    def configure(self, size_t in_bytes, ptr_t temp_bytes, ptr_t out_bytes):
        self.c.configure(in_bytes, <size_t*>temp_bytes, <size_t*>out_bytes)

    def compress_async(self, ptr_t in_ptr, size_t in_size, ptr_t temp_ptr, size_t temp_size, ptr_t out_ptr, ptr_t out_size, ptr_t stream = 0):
        self.c.compress_async(
            <void*>in_ptr,
            in_size,
            <void*>temp_ptr,
            temp_size,
            <void*>out_ptr,
            <size_t*>out_size,
            <cudaStream_t>stream)

cdef class CascadedDecompressor:
    cdef _CascadedDecompressor* d

    def __cinit__(self):
        self.d = new _CascadedDecompressor()

    def __dealloc__(self):
        del self.d

    def configure(self, ptr_t in_ptr, size_t in_bytes, ptr_t temp_bytes, ptr_t out_bytes, ptr_t stream = 0):
        self.d.configure(
            <void*>in_ptr,
            in_bytes,
            <size_t*>temp_bytes,
            <size_t*>out_bytes,
            <cudaStream_t>stream)

    def decompress_async(self, ptr_t in_ptr, size_t in_bytes, ptr_t temp_ptr, size_t temp_bytes, ptr_t out_ptr, size_t out_bytes, ptr_t stream = 0):
        self.d.decompress_async(
            <void*>in_ptr,
            in_bytes,
            <void*>temp_ptr,
            temp_bytes,
            <void*>out_ptr,
            out_bytes,
            <cudaStream_t>stream)
