# Python bindings for nvCOMP

To create the bindings:
- Build [nvCOMP](https://github.com/NVIDIA/nvcomp).
- Modify `setup.py` to point to the generated nvcomp include & lib directories. Modify it to point to your corresponding CUDA SDK directories as well. This is currently only tested on Windows, but it should work the same on Linux.
- (on windows) make sure you are in the MSVC command prompt context
- Run: `python setup.py install`

At this point, you should be able to open up a python console and type `import nvcomp`.

For an extended example using PyCUDA, look at `examples/pycuda_example.py`

# Current Limitations

- Only the `nvcomp::CascadedCompressor` and `nvcomp::CascadedDecompressor` classes have bindings
- Only `i4` and `u2` numpy type strings are correctly parsed at the moment (see `nvcomp_shared.hpp`)
