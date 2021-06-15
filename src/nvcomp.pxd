cdef extern from "cuda_runtime.h":
    ctypedef void* cudaStream_t

cdef extern from "nvcomp.h":
    cpdef enum nvcompType_t:
        NVCOMP_TYPE_CHAR,
        NVCOMP_TYPE_UCHAR,
        NVCOMP_TYPE_SHORT,
        NVCOMP_TYPE_USHORT,
        NVCOMP_TYPE_INT,
        NVCOMP_TYPE_UINT,
        NVCOMP_TYPE_LONGLONG,
        NVCOMP_TYPE_ULONGLONG,
        NVCOMP_TYPE_BITS
    
    cpdef enum nvcompError_t:
        nvcompSuccess = 0,
        nvcompErrorInvalidValue = 10,
        nvcompErrorNotSupported = 11,
        nvcompErrorCudaError = 1000,
        nvcompErrorInternal = 10000

# LZ4 Compressor
cdef extern from "nvcomp/lz4.hpp" namespace 'nvcomp':
    cdef cppclass _LZ4Compressor "nvcomp::LZ4Compressor":
        _LZ4Compressor(size_t chunk_size) except+

        void configure(
            const size_t in_bytes,
            size_t* temp_bytes,
            size_t* out_bytes) except+
        
        void compress_async(
            const void* in_ptr,
            const size_t in_bytes,
            void* temp_ptr,
            const size_t temp_bytes,
            void* out_ptr,
            size_t* out_bytes,
            cudaStream_t stream) except+

    cdef cppclass _LZ4Decompressor "nvcomp::LZ4Decompressor":
        _LZ4Decompressor() except+

        void configure(
            const void* in_ptr,
            const size_t in_bytes,
            size_t* temp_bytes,
            size_t* out_bytes,
            cudaStream_t stream) except+
        
        void decompress_async(
            const void* in_ptr,
            const size_t in_bytes,
            void* temp_ptr,
            const size_t temp_bytes,
            void* out_ptr,
            const size_t out_bytes,
            cudaStream_t stream) except+

# Cascaded Compressor
cdef extern from "nvcomp/cascaded.hpp" namespace 'nvcomp':
    cdef cppclass _CascadedCompressor "nvcomp::CascadedCompressor":
        _CascadedCompressor(nvcompType_t, int, int, bool) except+

        void configure(
            const size_t in_bytes,
            size_t* temp_bytes,
            size_t* out_bytes) except+

        void compress_async(
            const void* in_ptr,
            const size_t in_bytes,
            void* temp_ptr,
            const size_t temp_bytes,
            void* out_ptr,
            size_t* out_bytes,
            cudaStream_t stream) except+

    cdef cppclass _CascadedDecompressor "nvcomp::CascadedDecompressor":
        _CascadedDecompressor() except+

        void configure(
            const void* in_ptr,
            const size_t in_bytes,
            size_t* temp_bytes,
            size_t* out_bytes,
            cudaStream_t stream) except+

        void decompress_async(
            const void* in_ptr,
            const size_t in_bytes,
            void* temp_ptr,
            const size_t temp_bytes,
            void* out_ptr,
            const size_t out_bytes,
            cudaStream_t stream) except+

# Low-level LZ4 API
cdef extern from "nvcomp/lz4.h":

    # _nvcompError_t
    cdef nvcompError_t _nvcompBatchedLZ4CompressGetTempSize "nvcompBatchedLZ4CompressGetTempSize" (
        size_t batch_size,
        size_t max_uncompressed_chunk_bytes,
        size_t* temp_bytes)except+

    cdef nvcompError_t _nvcompBatchedLZ4CompressGetMaxOutputChunkSize "nvcompBatchedLZ4CompressGetMaxOutputChunkSize" (
        size_t max_uncompressed_chunk_bytes,
        size_t* max_compressed_bytes)except+

    cdef nvcompError_t _nvcompBatchedLZ4CompressAsync "nvcompBatchedLZ4CompressAsync" (
        const void* const* device_in_ptr,
        const size_t* device_in_bytes,
        size_t max_uncompressed_chunk_bytes, # unused
        size_t batch_size,
        void* device_temp_ptr,
        size_t temp_bytes,
        void* const* device_out_ptr,
        size_t* device_out_bytes,
        cudaStream_t stream)except+

    cdef nvcompError_t _nvcompBatchedLZ4DecompressAsync "nvcompBatchedLZ4DecompressAsync" (
        const void* const* device_in_ptrs,
        const size_t* device_in_bytes,
        const size_t*  device_out_bytes, # unused
        size_t max_uncompressed_chunk_bytes, # unused
        size_t batch_size,
        void* const device_temp_ptr, # unused
        const size_t temp_bytes, # unused
        void* const* device_out_ptr,
        cudaStream_t stream) except+