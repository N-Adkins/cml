# cml

`cml` is a C99-based machine learning library project.
It links against CBLAS for linear algebra operations.

The repository is in an early **work-in-progress** stage. Do not use! If you're seeing this, there's probably barely anything to use at all.

## Requirements

- CMake 3.21+
- A C compiler (GCC/Clang/MSVC)
- Ninja (recommended generator used in examples)
- A BLAS implementation with CBLAS support available as a static library (for example: OpenBLAS, BLIS, Intel MKL, or Apple Accelerate), or allow CMake to fetch OpenBLAS automatically

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

## BLAS/CBLAS Dependency

`cml` uses CBLAS for linear algebra operations and links it into the `cml` target.

The build first tries to find a system BLAS with CBLAS headers and static libraries. If none is found, it fetches and builds OpenBLAS statically by default.

To disable auto-fetching and require a system BLAS:

```bash
cmake -Bbuild -GNinja . -DCML_FETCH_OPENBLAS=OFF
```

## Run Tests

After configuring and building, run:

```bash
ctest --test-dir build --output-on-failure
```

Testing is wired through CTest, with a basic framework setup using `FetchContent`.
