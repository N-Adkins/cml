# cml

`cml` is a C99-based machine learning library project.

The repository is in an early **work-in-progress** stage. Do not use! If you're seeing this, there's probably barely anything to use at all.

## Requirements

- CMake 3.21+
- A C compiler (GCC/Clang/MSVC)
- Ninja (recommended generator used in examples)

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

## Run Tests

After configuring and building, run:

```bash
ctest --test-dir build --output-on-failure
```

Testing is wired through CTest, with a basic framework setup using `FetchContent`.
