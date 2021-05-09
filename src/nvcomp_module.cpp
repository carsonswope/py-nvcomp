#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#include <nvcomp.h>
#include <nvcomp/cascaded.hpp>

#include "nvcomp_shared.hpp"
#include "nvcomp_cascaded_compressor.hpp"
#include "nvcomp_cascaded_decompressor.hpp"

static PyModuleDef nvcomp_module = {
        PyModuleDef_HEAD_INIT,
        "nvcomp", // m_name
        "nvcomp docs!", // m_doc
        -1, // m_size
        NULL, // m_methods
        NULL, // m_slots
        NULL, // m_travers
        NULL, // m_clear
        NULL, // m_free
};

PyMODINIT_FUNC
PyInit_nvcomp(void)
{
    PyObject* m;
    if (PyType_Ready(&nvcomp_CascadedCompressor_Type) < 0) return NULL;
    if (PyType_Ready(&nvcomp_CascadedDecompressor_Type) < 0) return NULL;

    m = PyModule_Create(&nvcomp_module);
    if (m == NULL) return NULL;

    Py_INCREF(&nvcomp_CascadedCompressor_Type);
    if (PyModule_AddObject(m, "CascadedCompressor", (PyObject *) &nvcomp_CascadedCompressor_Type) < 0) {
        Py_DECREF(&nvcomp_CascadedCompressor_Type);
        Py_DECREF(m);
        return NULL;
    }

    Py_INCREF(&nvcomp_CascadedDecompressor_Type);
    if (PyModule_AddObject(m, "CascadedDecompressor", (PyObject *) &nvcomp_CascadedDecompressor_Type) < 0) {
        Py_DECREF(&nvcomp_CascadedDecompressor_Type);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
