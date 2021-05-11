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
