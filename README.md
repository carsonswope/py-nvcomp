# Python bindings for nvCOMP

To create the bindings:
- Build [nvCOMP](https://github.com/NVIDIA/nvcomp).
- Modify `setup.py` to point to your generated nvcomp `include` & `lib` directories. Also modify to point to your corresponding CUDA SDK directories as well. This is currently only tested on Windows, but should work the same on Linux.
- (on windows) make sure you are in MSVC command prompt context. You should be able to type `cl` into the command line and see the MSVC compiler. This hasn't been tested with MinGW or cygwin.
- Run: `python setup.py build_ext --inplace`. This will generate a `.pyd` file that you can copy into `site-packages`.

For example usages / unit tests using PyCUDA, look in `examples` directory.

# Implemented bindings

- LZ4Compressor / LZ4Decompressor
- CascadedCompressor / CascadedDecompressor
- C-API ('low-level') access to LZ4 compression/decompression functions
