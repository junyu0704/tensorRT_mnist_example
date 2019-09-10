
from random import randint
from PIL import Image
import numpy as np
import sys, os
import argparse

import time
# import tensorRT
import tensorrt as trt
# use python api tensorRT
import pycuda.driver as cuda
import pycuda.autoinit
import graphsurgeon as gs
import ctypes
import uff

# Path where clip plugin library will be built (check README.md)
CLIP_PLUGIN_LIBRARY = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    './build/libclipplugin.so'
)
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    './models/mnist.pb'
)

# There is a simple logger included in the TensorRT Python bindings.
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

## Create the ModelData
class ModelData(object):
    MODEL_FILE = "mnist.uff"
    INPUT_NAME = "input_1"
    INPUT_SHAPE = (1, 28, 28)
    OUTPUT_NAME = "dense_1/Softmax"
    RELU6_NAME = "ReLU6"
    DATA_TYPE = trt.float16

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()
def prepare_namespace_plugin_map():
    # In this sample, the only operation that is not supported by TensorRT
    # is tf.nn.relu6, so we create a new node which will tell UffParser which
    # plugin to run and with which arguments in place of tf.nn.relu6.


    # The "clipMin" and "clipMax" fields of this TensorFlow node will be parsed by createPlugin,
    # and used to create a CustomClipPlugin with the appropriate parameters.
    trt_relu6 = gs.create_plugin_node(name="trt_relu6", op="CustomClipPlugin", clipMin=0.0, clipMax=6.0)
    namespace_plugin_map = {
        ModelData.RELU6_NAME: trt_relu6
    }
    return namespace_plugin_map

def model_path_to_uff_path(model_path):
    uff_path = os.path.splitext(model_path)[0] + ".uff"
    return uff_path

# Converts the TensorFlow frozen graphdef to UFF format using the UFF converter
def model_to_uff(model_path):
    # Transform graph using graphsurgeon to map unsupported TensorFlow
    # operations to appropriate TensorRT custom layer plugins
    dynamic_graph = gs.DynamicGraph(model_path)
    dynamic_graph.collapse_namespaces(prepare_namespace_plugin_map())
    # Save resulting graph to UFF file
    output_uff_path = model_path_to_uff_path(model_path)
    uff.from_tensorflow(
        dynamic_graph.as_graph_def(),
        [ModelData.OUTPUT_NAME],
        output_filename=output_uff_path,
        text=True
    )
    return output_uff_path

def find_data(description="Runs a TensorRT Python sample", subfolder="", find_files=[]):
     # Standard command-line arguments for all samples.
    kDEFAULT_DATA_ROOT = os.path.join(os.sep, "usr", "src", "tensorrt", "data")
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--datadir", help="Location of the TensorRT sample data directory.", default=kDEFAULT_DATA_ROOT)
    args, unknown_args = parser.parse_known_args()

    # If data directory is not specified, use the default.
    data_root = args.datadir
    subfolder_path = os.path.join(data_root, subfolder)
    data_path = subfolder_path
    if not os.path.exists(subfolder_path):
        print("WARNING: " + subfolder_path + " does not exist. Trying " + data_root + " instead.")
        data_path = data_root

    # Find all requested files.
    for index, f in enumerate(find_files):
        find_files[index] = os.path.abspath(os.path.join(data_path, f))
        if not os.path.exists(find_files[index]):
            raise FileNotFoundError(find_files[index] + " does not exist. Please provide the correct data path with the -d option.")

    return data_path, find_files

def build_engine(model_path):
    # Create the builder, network, and parser
    # https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/infer/Core/Builder.html
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:

        #setting workspace szie
        builder.max_workspace_size = 1 << 30
        builder.debug_sync = True
        builder.fp16_mode = True
        # Parse the Uff Network
        # https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/parsers/Uff/pyUff.html
        uff_path = model_to_uff(model_path)
        parser.register_input(ModelData.INPUT_NAME, ModelData.INPUT_SHAPE)
        parser.register_output(ModelData.OUTPUT_NAME)
        parser.parse(uff_path, network)
        # Build and return an engine.
        # When the engine is built, TensorRT makes copies of the weights
        return builder.build_cuda_engine(network)

def load_normalized_test_data(data_path, pagelocked_buffer, num=randint(0, 9)):
    test_case_path = os.path.join(data_path, str(num) + ".pgm")
    # Flatten the image into a 1D array, normalize, and copy to pagelocked memory.
    # numpy.ravel() 1D array
    img = np.array(Image.open(test_case_path)).ravel()
    np.copyto(pagelocked_buffer, 1.0 - img / 255.0)
    return num

def inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

def init(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype) #h_input h_output
        device_mem = cuda.mem_alloc(host_mem.nbytes) #d_input d_output
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.

        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

def main():
    data_path, _ = find_data(description="using a UFF model file", subfolder="mnist")
    model_path = "./models"

    tStart = time.time()
    ctypes.CDLL(CLIP_PLUGIN_LIBRARY)
    with build_engine(MODEL_PATH) as engine:
        inputs, outputs, bindings, stream = init(engine)
        with open("sample.engine", "wb") as f:
            f.write(engine.serialize())
        with engine.create_execution_context() as context:

            num = load_normalized_test_data(data_path, pagelocked_buffer=inputs[0].host)
            tStart1 = time.time()
            [output] = inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            pred = np.argmax(output)
            print("Test Case: " + str(num))
            print("Prediction: " + str(pred))
            tEnd = time.time()
            print('time=', tEnd - tStart)
            print('time1=', tEnd - tStart1)

if __name__ == '__main__':
   main()
