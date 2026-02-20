#ifndef METAL_RENDER_H
#define METAL_RENDER_H

#include "gpu_render.h" // reuses GPUSceneParams + GPUPixelResult

// Launch the Metal compute render kernel.  Writes num_pixels results to host
// memory.  Returns true on success, false on Metal error.
// The interface is identical to gpu_render() so the caller (main.cpp) can
// switch between CUDA and Metal with a simple #ifdef.
bool metal_gpu_render(const GPUSceneParams &params, GPUPixelResult *host_results);

#endif // METAL_RENDER_H
