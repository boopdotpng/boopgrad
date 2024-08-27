import Metal
import ctypes
import random
from math import log

kernel_source = """
#include <metal_stdlib>
using namespace metal;
kernel void log_kernel(constant float *in  [[ buffer(0) ]],
                       device float *out [[ buffer(1) ]],
                       uint id [[ thread_position_in_grid ]]) {
    out[id] = log(in[id]);
}
"""

device = Metal.MTLCreateSystemDefaultDevice()
library = device.newLibraryWithSource_options_error_(kernel_source, None, None)[0]
kernel_function = library.newFunctionWithName_("log_kernel")


array_length = 1024
buffer_length = array_length * 4  # 4 bytes per float
input_buffer = device.newBufferWithLength_options_(buffer_length, Metal.MTLResourceStorageModeShared)
output_buffer = device.newBufferWithLength_options_(buffer_length, Metal.MTLResourceStorageModeShared)

input_list = [random.uniform(0.0, 1.0) for _ in range(array_length)]  
input_array = (ctypes.c_float * array_length).from_buffer(input_buffer.contents().as_buffer(buffer_length))  
input_array[:] = input_list  

commandQueue = device.newCommandQueue()
commandBuffer = commandQueue.commandBuffer()

pso = device.newComputePipelineStateWithFunction_error_(kernel_function, None)[0]
computeEncoder = commandBuffer.computeCommandEncoder()
computeEncoder.setComputePipelineState_(pso)
computeEncoder.setBuffer_offset_atIndex_(input_buffer, 0, 0)
computeEncoder.setBuffer_offset_atIndex_(output_buffer, 0, 1)

threadsPerThreadgroup = Metal.MTLSizeMake(1024, 1, 1)
threadgroupSize = Metal.MTLSizeMake(pso.maxTotalThreadsPerThreadgroup(), 1, 1)

computeEncoder.dispatchThreads_threadsPerThreadgroup_(threadsPerThreadgroup, threadgroupSize)
computeEncoder.endEncoding()

commandBuffer.commit()
commandBuffer.waitUntilCompleted()

output_data = (ctypes.c_float * array_length).from_buffer(output_buffer.contents().as_buffer(buffer_length))
output_list = list(output_data)

output_python = [log(x) for x in input_list]
assert all([abs(a - b) < 1e-5 for a, b in zip(output_list, output_python)]), "❌ Output does not match reference!"
print("✅ Reference matches output!")