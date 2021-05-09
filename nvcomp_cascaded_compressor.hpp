typedef struct {
    PyObject_HEAD
    nvcomp::CascadedCompressor* c;
} nvcomp_CascadedCompressor;

static PyObject *
nvcomp_CascadedCompressor_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    nvcomp_CascadedCompressor *self;
    self = (nvcomp_CascadedCompressor *) type->tp_alloc(type, 0);
    return (PyObject *) self;
}

static int
nvcomp_CascadedCompressor_init(nvcomp_CascadedCompressor *self, PyObject *args, PyObject *kwds)
{
    const char *np_type_chars;
    int num_RLEs, num_deltas;
    bool use_bp;
    if (!PyArg_ParseTuple(args, "siip", &np_type_chars, &num_RLEs, &num_deltas, &use_bp)) return -1;

    nvcompType_t nvcomp_type;
    try {
        nvcomp_type = nvcomp_parse_np_type(np_type_chars);
    } catch (const std::runtime_error& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return -1;
    }

    self->c = new nvcomp::CascadedCompressor(nvcomp_type, num_RLEs, num_deltas, use_bp);
    return 0;
}

static void
nvcomp_CascadedCompressor_dealloc(nvcomp_CascadedCompressor *self)
{
    delete self->c;
    self->c = NULL;
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyMemberDef nvcomp_CascadedCompressor_members[] = {
    {NULL}
};

static PyObject *
nvcomp_CascadedCompressor_configure(nvcomp_CascadedCompressor *self, PyObject * args) {
    Py_ssize_t uncompressed_size;
    if (!PyArg_ParseTuple(args, "n", &uncompressed_size)) return NULL;

    size_t temp_size;
    size_t max_output_size;
    self->c->configure(uncompressed_size, &temp_size, &max_output_size);
    return Py_BuildValue("(nn)", temp_size, max_output_size);
}

static PyObject *
nvcomp_CascadedCompressor_compress(nvcomp_CascadedCompressor *self, PyObject *args, PyObject * kwargs) {

    // All pointers are CUDA device pointers.
    void* uncompressed_data;
    Py_ssize_t uncompressed_size;
    void* temp_data;
    Py_ssize_t temp_size;
    void* compressed_data;
    // page-locked single int64 value (cudaMallocHost)
    void* compressed_size;
    if (!PyArg_ParseTuple(args, "LnLnLL",
        &uncompressed_data, &uncompressed_size,
        &temp_data, &temp_size,
        &compressed_data, &compressed_size)) return NULL;

    self->c->compress_async(
        uncompressed_data, uncompressed_size,
        temp_data, temp_size,
        compressed_data, (size_t*)compressed_size,
        NULL);

    return PyLong_FromLong(0);
}

static PyMethodDef nvcomp_CascadedCompressor_methods[] = {
    {"configure", (PyCFunction) nvcomp_CascadedCompressor_configure, METH_VARARGS, "determine memory usage for compression"},
    {"compress", (PyCFunction) nvcomp_CascadedCompressor_compress, METH_VARARGS, "do compression"},
    {NULL}  /* Sentinel */
};


static PyTypeObject nvcomp_CascadedCompressor_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "nvcomp.CascadedCompressor", // tp_name
    sizeof(nvcomp_CascadedCompressor), //
    0, // tp_itemsize
    (destructor)nvcomp_CascadedCompressor_dealloc, // tp_dealloc
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
    "Cascaded Compressor", // tp_doc
    0, // tp_traverse
    0, // tp_clear
    0, // tp_richcompare
    0, // tp_weaklistoffset
    0, // tp_iter
    0, // tp_iternext
    nvcomp_CascadedCompressor_methods, // tp_methods
    nvcomp_CascadedCompressor_members, // tp_members
    0, // tp_getset
    0, // tp_base
    0, // tp_dict
    0, // tp_descr_get
    0, // tp_descr_set
    0, // tp_dictoffset
    (initproc)nvcomp_CascadedCompressor_init, // tp_init
    0, // tp_alloc
    nvcomp_CascadedCompressor_new, // tp_new
};
