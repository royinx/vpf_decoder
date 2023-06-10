# third party library
import cupy as cp
import PyNvCodec as nvc
from loguru import logger

class VideoDecoder():
    def __init__(self):
        pass 

    def decode_video(self, input_video: str, gpu_id:int = 0):
        """Decode video into cupy array

        Args:
            gpu_id (int): _description_
            input_video (str): _description_

        Returns:
            _type_: _description_
        """
        nvDec = nvc.PyNvDecoder(input_video, gpu_id)
        max_frame = nvDec.Numframes()
        target_h = nvDec.Height()
        target_w = nvDec.Width()

        video = cp.zeros((max_frame, target_h, target_w, 3), dtype=cp.uint8)
        
        to_yuv = nvc.PySurfaceConverter(nvDec.Width(), nvDec.Height(),
                                        nvc.PixelFormat.NV12, nvc.PixelFormat.YUV420,
                                        gpu_id)
        
        to_dim = nvc.PySurfaceResizer(target_w, target_h, 
                                      nvc.PixelFormat.YUV420, 
                                      gpu_id)

        to_rgb = nvc.PySurfaceConverter(target_w, target_h,
                                        nvc.PixelFormat.YUV420, nvc.PixelFormat.RGB,
                                        gpu_id)

        # Use most widespread bt601 and mpeg just for illustration purposes.
        cc_ctx = nvc.ColorspaceConversionContext(nvc.ColorSpace.BT_601,
                                                nvc.ColorRange.MPEG)
        
        count = 0 
        while True:
            # Decode 1 compressed video frame to CUDA memory.
            nv12_surface = nvDec.DecodeSingleSurface()
            if nv12_surface.Empty():
                logger.debug('Cannot decode frame/ no more frames')
                break

            # Convert from NV12 to YUV420.
            # This extra step is required because not all NV12 -> RGB conversions
            # implemented in NPP support all color spaces and ranges.
            yuv420 = to_yuv.Execute(nv12_surface, cc_ctx)
            if yuv420.Empty():
                logger.error('Can not convert nv12 -> yuv420')
                break

            # Downscale YUV420.
            yuv_small = to_dim.Execute(yuv420)
            if yuv_small.Empty():
                logger.error('Can not downscale yuv420 surface')
                break

            # Convert from YUV420 to interleaved RGB. (1536 6144 3)
            rgb24_small = to_rgb.Execute(yuv_small, cc_ctx)
            if rgb24_small.Empty():
                logger.error('Can not convert yuv420 -> rgb')
                break

            out = self._get_cupy_array(rgb24_small)
            video[count] = out
            count += 1
        video = cp.ascontiguousarray(video[:count])
        return video

    def _get_cupy_array(self, rgb_frame: nvc.Surface):
        plane = rgb_frame.PlanePtr()
        # cuPy array zero copy non ownned
        H, W, pitch = (plane.Height(), plane.Width(), plane.Pitch())
        
        def get_memptr(rgb_frame):
            return rgb_frame.PlanePtr().GpuMem()
        cupy_mem = cp.cuda.UnownedMemory(get_memptr(rgb_frame), H * W * 1, rgb_frame)
        cupy_memptr = cp.cuda.MemoryPointer(cupy_mem, 0)

        cupy_frame = cp.ndarray( (H, W // 3, 3), cp.uint8, cupy_memptr, strides=(pitch, 3, 1))
        return cupy_frame

if __name__ == "__main__":
    video_decoder = VideoDecoder()
    path = "vpf_build/Tests/test.mp4"
    frames = video_decoder.decode_video(path, gpu_id = 0)
    print(frames.shape)