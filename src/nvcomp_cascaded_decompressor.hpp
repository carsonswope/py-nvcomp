typedef struct {
    PyObject_HEAD
    nvcomp::CascadedDecompressor* d;
} nvcomp_CascadedDecompressor;

static PyObject *
nvcomp_CascadedDecompressor_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    nvcomp_CascadedDecompressor *self;
    self = (nvcomp_CascadedDecompressor *) type->tp_alloc(type, 0);
    return (PyObject *) self;
}

static int
nvcomp_CascadedDecompressor_init(nvcomp_CascadedDecompressor *self, PyObject *args, PyObject *kwds)
{
    // No args!
    if (!PyArg_ParseTuple(args, "")) return -1;
    self->d = new nvcomp::CascadedDecompressor();
    return 0;
}

static void
nvcomp_CascadedDecompressor_dealloc(nvcomp_CascadedDecompressor *self)
{
    delete self->d;
    self->d = NULL;
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyMemberDef nvcomp_CascadedDecompressor_members[] = {
    {NULL}  /* Sentinel */
};

static PyObject *
nvcomp_CascadedDecompressor_configure(nvcomp_CascadedDecompressor* self, PyObject* args, PyObject* kwargs) {
    void* compressed_data;
    Py_ssize_t compressed_size;
    void* temp_size;
    void* output_size;
    void* stream = NULL;

    if (!PyArg_ParseTuple(args, "LnLL", &compressed_data, &compressed_size, &temp_size, &output_size)) return NULL;

    static char *kwlist[] = {
        "compressed_data",
        "compressed_size",
        "temp_size",
        "output_size",
        "stream",
        NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "LnLL|L", kwlist,
        &compressed_data, &compressed_size, &temp_size, &output_size, &stream)) return NULL;

    self->d->configure(
        compressed_data,
        compressed_size,
        (size_t*)temp_size,
        (size_t*)output_size,
        (cudaStream_t)stream);

    return PyLong_FromLong(0);
}

static PyObject *
nvcomp_CascadedDecompressor_decompress(nvcomp_CascadedDecompressor* self, PyObject* args, PyObject* kwargs) {

    // All pointers are CUDA device pointers.
    void* compressed_data;
    Py_ssize_t compressed_size;
    void* temp_data;
    Py_ssize_t temp_size;
    void* uncompressed_data;
    Py_ssize_t uncompressed_size;
    void* stream = NULL;

    static char* kwlist[] = {
        "compressed_data",
        "compressed_size",
        "temp_data",
        "temp_size",
        "uncompressed_data",
        "uncompressed_size",
        "stream",
        NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "LnLnLn|L", kwlist,
        &compressed_data, &compressed_size,
        &temp_data, &temp_size,
        &uncompressed_data, &uncompressed_size, &stream)) return NULL;

    self->d->decompress_async(
        compressed_data, compressed_size,
        temp_data, temp_size,
        uncompressed_data, uncompressed_size,
        (cudaStream_t)stream);

    return PyLong_FromLong(0);
}

static PyMethodDef nvcomp_CascadedDecompressor_methods[] = {
    {"configure", (PyCFunction) nvcomp_CascadedDecompressor_configure, METH_VARARGS | METH_KEYWORDS, "Determine memory usage for decompression"},
    {"decompress", (PyCFunction) nvcomp_CascadedDecompressor_decompress, METH_VARARGS | METH_KEYWORDS, "Do decompression"},
    {NULL}  /* Sentinel */
};

static PyTypeObject nvcomp_CascadedDecompressor_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "nvcomp.CascadedDecompressor", // tp_name
    sizeof(nvcomp_CascadedDecompressor), //
    0, // tp_itemsize
    (destructor)nvcomp_CascadedDecompressor_dealloc, // tp_dealloc
    0, // tp_vectorcall_offset
    0, // tp_getattr
    0, // tp_setattr
    0, // tp_as_async
    0, // tp_repr
    0, // tp_as_number
    0, // tp_as_sequence
    0, // tp_as_mapping
    0, // tp_hash
    0, // tp_call
    0, // tp_str
    0, // tp_getattro
    0, // tp_setattro
    0, // tp_as_buffer
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, // tp_flags
    "Cascaded Decompressor", // tp_doc
    0, // tp_traverse
    0, // tp_clear
    0, // tp_richcompare
    0, // tp_weaklistoffset
    0, // tp_iter
    0, // tp_iternext
    nvcomp_CascadedDecompressor_methods, // tp_methods
    nvcomp_CascadedDecompressor_members, // tp_members
    0, // tp_getset
    0, // tp_base
    0, // tp_dict
    0, // tp_descr_get
    0, // tp_descr_set
    0, // tp_dictoffset
    (initproc)nvcomp_CascadedDecompressor_init, // tp_init
    0, // tp_alloc
    nvcomp_CascadedDecompressor_new, // tp_new
};
