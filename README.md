# cml

`cml` is a C99-based machine learning library project.
It links against CBLAS for linear algebra operations.

The repository is in an early **work-in-progress** stage. Do not use! If you're seeing this, there's probably barely anything to use at all.

## Requirements

- CMake 3.21+
- A C compiler (GCC/Clang/MSVC)
- Ninja (recommended generator used in examples)
- A BLAS implementation with CBLAS support available as a static library (for example: OpenBLAS, BLIS, Intel MKL, or Apple Accelerate), or allow CMake to fetch OpenBLAS automatically
- (Optional) CUDA Toolkit with cuBLAS for GPU backend support

## Build

Configure and build:

```bash
cmake -Bbuild -GNinja .
cmake --build build
```

## Sanitizers

AddressSanitizer and UndefinedBehaviorSanitizer are enabled by default (non-MSVC builds).

To disable sanitizers:

```bash
cmake -Bbuild -GNinja . -DCML_ENABLE_SANITIZERS=OFF
```

To explicitly enable:

```bash
cmake -Bbuild -GNinja . -DCML_ENABLE_SANITIZERS=ON
```

When CUDA backend support is enabled, sanitizers are skipped.

## BLAS/CBLAS Dependency

`cml` uses CBLAS for linear algebra operations and links it into the `cml` target.

The build first tries to find a system BLAS with CBLAS headers and static libraries. If none is found, it fetches and builds OpenBLAS statically by default.

To disable auto-fetching and require a system BLAS:

```bash
cmake -Bbuild -GNinja . -DCML_FETCH_OPENBLAS=OFF
```

## CUDA Backend (Optional)

To compile CUDA support:

```bash
cmake -Bbuild -GNinja . -DCML_ENABLE_CUDA=ON
cmake --build build
```

`cml` passes `-allow-unsupported-compiler` and `-std=c++20` to `nvcc` by default to help with newer host compilers.
To disable that behavior, set `-DCML_CUDA_ALLOW_UNSUPPORTED_COMPILER=OFF`.

Use `cml_init_with_backend(..., CML_BACKEND_CUDA)` to request CUDA for a context.
`cml_init(...)` keeps the default CPU backend.

### CUDA parity tests

When `CML_ENABLE_CUDA=ON`, an additional test target (`test_cuda_parity`) is added.
It verifies parity between CPU and CUDA results for deterministic tensor operations.

If CUDA is compiled in but no runtime CUDA device/backend is available, parity checks are skipped.

## Run Tests

After configuring and building, run:

```bash
ctest --test-dir build --output-on-failure
```

Testing is wired through CTest, with a basic framework setup called [Unity](https://github.com/ThrowTheSwitch/Unity/).

This may be swapped in the future because it doesn't have many features as its target is embedded platforms, but
it provides everything currently needed.

## CI

GitHub Actions runs:
- `build-and-test`: regular CPU build/tests on Ubuntu and Windows.
- `build-and-test-cuda`: CUDA-enabled Ubuntu build/tests with CUDA toolkit installation and `-DCML_ENABLE_CUDA=ON`.
