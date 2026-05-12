target_sources(cml PRIVATE
    backend/backend.c
    backend/cpu_backend.c
    context.c
    data.c
    linear.c
    loss.c
    nn.c
    optimizer.c
    tape.c
    tensor.c
    trainer.c
    memory/arena.c
)

if(CML_ENABLE_CUDA)
    target_sources(cml PRIVATE backend/cuda_backend.cu)
endif()
