#
# Copyright 2021 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Starting from Python 3.8 DLL search policy has changed.
# We need to add path to CUDA DLLs explicitly.
# import sys
# import os
# from torch import onnx
# from torch._C import ListType

# import torchvision
# from subprocess import PIPE, STDOUT, run

# if os.name == 'nt':
#     # Add CUDA_PATH env variable
#     cuda_path = os.environ["CUDA_PATH"]
#     if cuda_path:
#         os.add_dll_directory(cuda_path)
#     else:
#         print("CUDA_PATH environment variable is not set.", file=sys.stderr)
#         print("Can't set CUDA DLLs search path.", file=sys.stderr)
#         exit(1)

#     # Add PATH as well for minor CUDA releases
#     sys_path = os.environ["PATH"]
#     if sys_path:
#         paths = sys_path.split(';')
#         for path in paths:
#             if os.path.isdir(path):
#                 os.add_dll_directory(path)
#     else:
#         print("PATH environment variable is not set.", file=sys.stderr)
#         exit(1)

import os 
import torch
import tensorrt as trt
import pycuda
import pycuda.driver as cuda
import numpy as np

import PyNvCodec as nvc
# import PytorchNvCodec as pnvc

from torchvision import transforms


from line_profiler import LineProfiler
profile = LineProfiler()

resnet_categories = []

class PyTorchTensorHolder(pycuda.driver.PointerHolderBase):
    def __init__(self, tensor):
        super(PyTorchTensorHolder, self).__init__()
        self.tensor = tensor

    def get_pointer(self):
        return self.tensor.data_ptr()


class HostDeviceMem():
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TensorRTContext:
    def __init__(self, trt_nn_file: str, gpu_id: int) -> None:
        self.device = cuda.Device(gpu_id)
        self.cuda_context = self.device.retain_primary_context()
        self.push_cuda_ctx()
        self.stream = cuda.Stream()

        self.logger = trt.Logger(trt.Logger.INFO)
        self.runtime = trt.Runtime(self.logger)

        f = open(trt_nn_file, 'rb')
        self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.inputs, self.outputs, self.bindings = self.allocate_buffer()
        self.context = self.engine.create_execution_context()

    def __del__(self) -> None:
        self.pop_cuda_ctx()

    def push_cuda_ctx(self) -> None:
        self.cuda_context.push()

    def pop_cuda_ctx(self) -> None:
        self.cuda_context.pop()

    def allocate_buffer(self):
        bindings = []
        inputs = []
        outputs = []

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(
                binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings

    def run_inference(self, tensor_image) -> str:
        # Copy from PyTorch tensor to plain CUDA memory
        cuda.memcpy_dtod(self.bindings[0], PyTorchTensorHolder(tensor_image),
                         tensor_image.nelement() * tensor_image.element_size())

        # Run inference
        self.context.execute_async_v2(
            bindings=self.bindings, stream_handle=self.stream.handle)

        # Copy outputs from GPU
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)

        # Find most probable image type and return resnet categoy description
        [result] = [out.host for out in self.outputs]
        return resnet_categories[np.argmax(result)]


# Resnet expects images to be 3 channel planar RGB of 224x224 size at least.
target_w, target_h = 224, 224

def out(command):
    result = run(command, text=True,
                 shell=True, stdout=PIPE, stderr=STDOUT)
    return result.stdout


def Resnet50ExportToOnxx(nn_onxx: str, nn_trt: str) -> None:
    
    nn_onxx_exists = os.path.exists(nn_onxx) and os.path.getsize(nn_onxx)
    nn_trt_exists = os.path.exists(nn_trt) and os.path.getsize(nn_trt)
    
    if nn_onxx_exists and nn_trt_exists:
        print('Both ONXX and TRT files exist. Skipping the export.')
        return


    torch.manual_seed(0)
    resnet50 = torchvision.models.resnet50(pretrained=True)
    resnet50.eval()
    input_data = torch.randn(1, 3, target_h, target_w, dtype=torch.float32)

    input_names = ['input']
    output_names = ['output']

    print('Exporting resnet50 to onxx file...')
    torch.onnx.export(resnet50,
                      input_data,
                      nn_onxx,
                      input_names=input_names,
                      output_names=output_names,
                      verbose=False,
                      opset_version=9)

    print('Exporting resnet50 to trt file...')
    bash_cmd = ""
    bash_cmd += "CUDA_VISIBLE_DEVICES=0 trtexec --verbose --onnx="
    bash_cmd += nn_onxx
    bash_cmd += " --saveEngine="
    bash_cmd += nn_trt

    stdout = out(bash_cmd)
    print(stdout)

def infer_on_video(gpu_id: int, input_video: str, trt_nn_file: str):

    # Init TRT stuff
    # cuda.init()
    # trt_ctx = TensorRTContext(trt_nn_file, gpu_id)

    # Init HW decoder, convertor, resizer + tensor that video frames will be
    # exported to
    nvDec = nvc.PyNvDecoder(input_video, gpu_id)

    to_yuv = nvc.PySurfaceConverter(nvDec.Width(), nvDec.Height(),
                                    nvc.PixelFormat.NV12, nvc.PixelFormat.YUV420,
                                    gpu_id)

    to_dim = nvc.PySurfaceResizer(target_w, target_h, nvc.PixelFormat.YUV420,
                                  gpu_id)

    to_rgb = nvc.PySurfaceConverter(target_w, target_h,
                                    nvc.PixelFormat.YUV420, nvc.PixelFormat.RGB,
                                    gpu_id)

    to_pln = nvc.PySurfaceConverter(target_w, target_h, nvc.PixelFormat.RGB,
                                    nvc.PixelFormat.RGB_PLANAR, gpu_id)

    # Use most widespread bt601 and mpeg just for illustration purposes.
    cc_ctx = nvc.ColorspaceConversionContext(nvc.ColorSpace.BT_601,
                                             nvc.ColorRange.MPEG)

    # Decoding cycle + inference on video frames.

    @profile
    def decode():
        counter = 0
        while True:
            # Decode 1 compressed video frame to CUDA memory.
            nv12_surface = nvDec.DecodeSingleSurface()
            if nv12_surface.Empty():
                print('Cannot decode frame/ no more frames')
                break
            counter += 1

            # Convert from NV12 to YUV420.
            # This extra step is required because not all NV12 -> RGB conversions
            # implemented in NPP support all color spaces and ranges.
            yuv420 = to_yuv.Execute(nv12_surface, cc_ctx)
            if yuv420.Empty():
                print('Can not convert nv12 -> yuv420')
                break

            # Downscale YUV420.
            yuv_small = to_dim.Execute(yuv420)
            if yuv_small.Empty():
                print('Can not downscale yuv420 surface')
                break

            # Convert from YUV420 to interleaved RGB.
            rgb24_small = to_rgb.Execute(yuv_small, cc_ctx)
            if rgb24_small.Empty():
                print('Can not convert yuv420 -> rgb')
                break

            # Convert to planar RGB.
            rgb24_planar = to_pln.Execute(rgb24_small, cc_ctx)
            if rgb24_planar.Empty():
                print('Can not convert rgb -> rgb planar')
                break
            

            # Export to PyTorch tensor
            surf_plane = rgb24_planar.PlanePtr()
            print(surf_plane)
            # img_tensor = pnvc.makefromDevicePtrUint8(surf_plane.GpuMem(),
            #                                          surf_plane.Width(),
            #                                          surf_plane.Height(),
            #                                          surf_plane.Pitch(),
            #                                          surf_plane.ElemSize())
            # print(counter, type(img_tensor))
            # img_tensor.resize_(3, target_h, target_w)
            # img_tensor = img_tensor.type(dtype=torch.cuda.FloatTensor)
            # img_tensor = torch.divide(img_tensor, 255.0)

            # data_transforms = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                                                    std=[0.229, 0.224, 0.225])
            # surface_tensor = data_transforms(img_tensor)

            # Run inference
            # img_type = trt_ctx.run_inference(surface_tensor)

            # Output result
            # print('Image type: ', img_type)

        print(counter)

    decode()

if __name__ == "__main__":
    gpu_id = 0
    input_video = '2012.mp4'

    onnx_file = './resnet50.onxx'
    trt_file = './resnet50.trt'

    # Resnet50ExportToOnxx(onnx_file, trt_file)
    infer_on_video(gpu_id, input_video, trt_file)
    # profile.print_stats()