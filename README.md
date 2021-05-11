# Python bindings for nvCOMP

To create the bindings:
- Build [nvCOMP](https://github.com/NVIDIA/nvcomp).
- Modify `setup.py` to point to the generated nvcomp include & lib directories. Modify it to point to your corresponding CUDA SDK directories as well. Currently only tested on Windows, but should work the same on Linux.
- (on windows) make sure you are in MSVC command prompt context
- Run: `python setup.py build_ext --inplace`. This will generate the `.pyd` file for site-packages.

For an extended example / unittest using PyCUDA, look at `examples/pycuda_example.py`

# Current Limitations

- Only the `nvcomp::CascadedCompressor` and `nvcomp::CascadedDecompressor` classes have bindings
- Only `i4` and `u2` numpy type strings are correctly parsed at the moment. 
