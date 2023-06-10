import cupy as cp
import PyNvCodec as nvc

from line_profiler import LineProfiler

profile = LineProfiler()


class Video_Capture():
    def __init__(self):
        self.dst_h = 640
        self.dst_w = 640
    
    @profile
    def decode_video(self, input_video: str, gpu_id:int = 0):
        """Decode video into cupy array

        Args:
            gpu_id (int): _description_
            input_video (str): _description_

        Returns:
            _type_: _description_
        """
        nvDec = nvc.PyNvDecoder(input_video, gpu_id)
        src_shape = (nvDec.Height(), nvDec.Width()) # H,W
        
        max_frame = nvDec.Numframes()
        video = cp.zeros((max_frame, self.dst_h, self.dst_w, 3), dtype=cp.uint8)
        target_shape, resize_scale, top_pad, left_pad = self._get_resize_shape(src_shape)
        target_h, target_w = target_shape
        
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
                print('Cannot decode frame/ no more frames')
                break

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

            # Convert from YUV420 to interleaved RGB. (1536 6144 3)
            rgb24_small = to_rgb.Execute(yuv_small, cc_ctx)
            if rgb24_small.Empty():
                print('Can not convert yuv420 -> rgb')
                break

            out = self._get_cupy_array(rgb24_small)
            # print(video[count, :, left_pad:left_pad + out.shape[1], :].shape, out.shape)
            if src_shape[0] / src_shape[1] > self.dst_h / self.dst_w: # H/W
                video[count, :, left_pad:left_pad + out.shape[1], :] = out
            else:
                video[count, top_pad:top_pad + out.shape[0], :, :] = out
            count += 1
        video = cp.ascontiguousarray(video[:count])
        rescale_info = (resize_scale, top_pad, left_pad)

        print(video.shape)
        return video, src_shape, rescale_info

    def _get_resize_shape(self, src_shape):
        src_h , src_w = src_shape
        dst_w = self.dst_w
        dst_h = self.dst_h
        
        resize_scale = 1
        left_pad = 0
        top_pad = 0
        if src_h / src_w > dst_h / dst_w:
            resize_scale = dst_h / src_h
            ker_h = dst_h
            ker_w = int(src_w * resize_scale)
            left_pad = int((dst_w - ker_w) / 2)
        else:
            resize_scale = dst_w / src_w
            ker_h = int(src_h * resize_scale)
            ker_w = dst_w
            top_pad = int((dst_h - ker_h) / 2)
        target_shape = (ker_h, ker_w)
        return  target_shape, resize_scale, top_pad, left_pad
        
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
    input_video = '2012.mp4'
    vc = Video_Capture()
    video, src_shape, rescale_info = vc.decode_video(input_video, 
                            gpu_id = 0 )
    
    print(video.shape)
    # import cv2
    # for idx,frame in enumerate(cp.asnumpy(video)):
    #     cv2.imwrite(f"temp/{idx}.jpg", frame[:,:,::-1])
    profile.print_stats()
    
    
    
# python3 test_nvc.py