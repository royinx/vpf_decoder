# ByteTrack Backbone

## Quick Start
### Environment setting
---
```bash
docker build -t vanjie_trt -f trt.dockerfile .
docker run --rm \
           --runtime=nvidia \
           --name=vanjie_trt \
           --privileged \
           -it \
           -v $PWD:/py/ \
           -w /py \
           vanjie_trt bash
```

---
#### Yolo v7 (TensorRT)

```bash 
# Download Bathc 1 onnx for model transform
wget -O /py/engine/yolov7_end2end_batch_1.onnx https://www.dropbox.com/s/k9uixjenfz6qu9e/yolov7_end2end_batch_1.onnx?dl=1
trtexec --onnx=/py/engine/yolov7_end2end_batch_1.onnx --saveEngine=/py/engine/end2end_batch_1.trt --fp16 --workspace=18432
```
<details><summary> Export manually </summary>

```bash 
docker build -t vanjie_onnx -f onnx_export.dockerfile .
docker run --rm \
           --runtime=nvidia \
           --name=vanjie_onnx \
           --privileged \
           -it \
           -v $PWD:/py/ \
           -w /py \
           vanjie_onnx bash
git clone https://github.com/WongKinYiu/yolov7
cd yolov7

# int8 (not recommend, not accurate)
python3 export.py --weights ./yolov7_best.pt --batch=1 --grid --include-nms --simplify --iou-thres 0.45 --conf-thres 0.5 --int8 --device 0
trtexec --onnx=/py/yolov7/yolov7_best.onnx --saveEngine=/py/engine/end2end_batch_1.trt --int8 --workspace=18432 --buildOnly

# fp16 (*** recommend)
python3 export.py --weights ./yolov7_best.pt --batch=1 --grid --include-nms --simplify --iou-thres 0.45 --conf-thres 0.5 --fp16 --device 0
trtexec --onnx=/py/yolov7/yolov7_best.onnx --saveEngine=/py/engine/end2end_batch_1.trt --fp16 --workspace=18432 --buildOnly

# fp32 (not recommend, slow)
python3 export.py --weights ./yolov7_best.pt --batch=1 --grid --include-nms --iou-thres 0.45 --conf-thres 0.5 --device 0 --img-size 640 640
trtexec --onnx=/py/yolov7/yolov7_best.onnx --saveEngine=/py/engine/end2end_batch_1.trt --workspace=18432 --buildOnly
```

#### Test
```bash
python3 test/pipe_yolov7_trt.py # python test case
trtexec --shapes=images:8x3x640x640 --loadEngine=end2end_batch_1.trt # TensorRT Engine tester
```

##### ONNX Runtime - end2end raw onnx ORT
```bash
python3 export.py --weights ./yolov7_best.pt --batch=1 --grid --end2end --simplify --iou-thres 0.45 --conf-thres 0.5 --device 0 --max-wh 1
mv ./yolov7_best.onnx /py/engine/end2end/fp32/runtime_batch_1.onnx
python3 test/pipe_yolov7_ort.py
```
</details>


---
#### OSnet (TensorRT)

```bash
trtexec --onnx=/py/engine/osnet_dynamic.onnx --saveEngine=/py/engine/osnet.engine --minShapes=images:1x3x256x128 --optShapes=images:128x3x256x128 --maxShapes=images:256x3x256x128 --workspace=4096
```
<details><summary> Export manually </summary>

```bash
# copy model weight from vanjie repo.
cp osnet_ain_x0_25_best.pth.tar osnet_ain_x0_25.pth.tar
git clone https://github.com/KaiyangZhou/deep-person-reid.git export_osnet
export PYTHONPATH=$PWD/export_osnet:$PYTHONPATH
# pip3 install pandas torch torchvision gdown tensorboard onnx onnxsim
```
##### modify export.py
> open export_osnet/tools/export.py
```python
# line 13
__model_types = [
    'resnet50', 'mlfn', 'hacnn', 'mobilenetv2_x1_0', 'mobilenetv2_x1_4',
    'osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5', 'osnet_x0_25',
    'osnet_ibn_x1_0', 'osnet_ain_x1_0', 
    'osnet_ain_x0_25']
# line 88 't0' -> 'images' 
input_shapes={'images': list(im.shape)} if dynamic else None)

# remove line 87
dynamic_input_shape=dynamic,
```

##### Export ONNX file (dynamic batch size)
```bash 
# transform onnx
python3 export_osnet/tools/export.py --dynamic --weight /py/engine/osnet_ain_x0_25.pth.tar --include onnx
mv /py/engine/osnet_ain_x0_25.pth.onnx /py/engine/osnet_dynamic.onnx

# transform TRT (* fp16 will cause too much losses, we recommend fp32 as Feature Extractor)
trtexec --onnx=/py/engine/osnet_dynamic.onnx --saveEngine=/py/engine/osnet.engine --minShapes=images:1x3x256x128 --optShapes=images:128x3x256x128 --maxShapes=images:256x3x256x128 --workspace=4096

trtexec --shapes=images:4x3x256x128 --loadEngine=/py/engine/osnet.engine

clear && clear && python3 test/pipe_osnet.py 
# python script setting - batch_size = 256 , dynamic = True 
```

---

##### For specific batch size (from dynamic to specific) (not recommend*, only batch = 1 works.)
```bash 
# tranform onnx (use the dynamic onnx*)
python3 ./tools/dynamic.py /py/engine/osnet_dynamic.onnx 1
# transform TRT
trtexec --onnx=/py/engine/osnet_dynamic_batch_1.onnx --saveEngine=/py/engine/osnet.engine --workspace=4096

# DON't do fp16 , too much losses
# batch - 1 
# trtexec --onnx=/py/engine/osnet_dynamic_batch_1.onnx --saveEngine=/py/engine/osnet.engine --fp16 --workspace=4096 --timingCacheFile=timing.cache
# batch - 256
# trtexec --onnx=/py/engine/osnet_dynamic_batch_256.onnx --saveEngine=/py/engine/osnet.engine --fp16 --workspace=4096 --timingCacheFile=timing.cache
```
</details>

---

### Run
```bash
### Engine pipeline
python3 application.py

### Independent Model pipeline
export PYTHONPATH=$PWD/:$PYTHONPATH
python3 test/pipe_yolov7_trt.py
python3 test/pipe_osnet.py
python3 test/pipe_dbscan.py
python3 test/pipe_bytetrack.py
### Extra - Yolo onnxruntime (remember export onnx version ONNX_NMS for onnxruntime)
pip3 install onnxruntime-gpu
python3 test/pipe_yolov7_ort.py
```
---

<details><summary> <b>Sprint List</b> </summary>

 - [x] Bottleneck Profiling 
    - Found out: Big Model to GPU - Yolov7(TensorRT), OSNet(TensorRT), DBSCAN(cuML.DBSCAN)
    - faster in CPU : Bytetrack, Digraph
 - [x] Docker Env Configuration
 - [x] Transform Yolo and OSNet into TRT 
 - [x] CUDA kernel for Preprocessing <br/>(Bilinear Interpolation, Transpose, Normalise), Different combination latency profiling, Fastest method is ~0.18ms/img
 - [x] Test the [Yolov7 engine](pipe_yolov7_trt.py)
 - [x] Test the [OSNet engine](pipe_osnet.py)
 - [x] Test the [ByteTrack engine](pipe_bytetrack.py)
 - [x] Test the [DBScan engine](pipe_dbscan.py)
 
 - [x] Transform pseudo clustering into cupy (cosine_distance in cupy) [checkout](https://github.com/royinx/GPU-tracker/blob/a5124591632503137d74135fc0a950de6cd96a9d/src/distance.py)
 - [x] profile cuML.DBSCAN vs pseudo clustering

```diff
CPU dbscan :  1x   (6.06770 s)
CPU pseudo : ~5x   (1.24645 s)
- GPU dbscan : 131x  (0.04617 s)
```
 - [x] use (Normalised vector + Euclidean) instead of (vector + cosine_distance) [cosine bug in cuML](https://github.com/rapidsai/cuml/issues/4938) <br/> eps_cosine = {0.25, 0.27} <br/>
 eps_euclidean = {0.705, 0.725}

 - [x] clean up temporary variables & most of the self.dict 
 - [x] application pipeline, 4 func left (cluster_occlusions, resolve_issues, save_pickle, load_previous_clusters)
 - [x] splitting 1min video into 10seconds videos
 <br/>-> split into 256 frames(~25*10frames) a batch (~7 batches) 
 <br/>-> save 2 seconds mean embeddings for continuation rematch
 - [x] Init multiple GPU Thread / Stream using cupy

 - [x] Removing cuML , use CUDA script instead. 
 - [x] Modifying Struct `Point` from 2D to n-dimensions (512)
 - [x] Test and profiling


 - [ ] Engine I/O: (Eugene will handle)
 <br/>In: add pull job janitor (Boto3) 
 <br/> out: Pack and upload the mean_embeddings.pkl for next video
 <br/> out: send request to Tracker Stats

---
</details>
