import av
import numpy as np
import torch

from transformers import VivitImageProcessor, VivitForVideoClassification
from huggingface_hub import hf_hub_download

np.random.seed(0)

def read_video_pyav(container, indices):  #解码视频 
    '''
    Decode the video with PyAV decoder.
    Args:
    container (av.container.input.InputContainer): PyAV container.
    indices (List[int]): List of frame indices to decode.
    Returns:
    result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])  #返回解码后的帧序列

def sample_frame_indices(clip_len, frame_sample_rate, seg_len): #从视频中采样一定数量的帧索引
    '''
    Sample a given number of frame indices from the video.
    Args:
    clip_len (int): Total number of frames to sample.  #采样的总帧数
    frame_sample_rate (int): Sample every n-th frame.  #采样的帧间隔

    seg_len (int): Maximum allowed index of sample's last frame.  #采样的最后一帧的最大索引
    Returns:
    indices (List[int]): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate) #被使用的总帧数
    end_idx = np.random.randint(converted_len, seg_len)  #随机选择一帧作为采样的最后一帧
    start_idx = end_idx - converted_len  #采样序列的开始帧
    indices = np.linspace(start_idx, end_idx, num=clip_len)  #生成采样帧的索引
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)  #确保采样的最后一帧索引不超过end_idx -1
    return indices  #返回采样后的帧 索引列表

##video clip consists of 300 frames (10 seconds at 30 FPS)
# file_path = hf_hub_download(
# repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
# )
file_path = "/home/cv/Project1/cxh/multi_model/vivit/dataset/eating_spaghetti.mp4"
file_path = "/home/cv/Project1/cxh/multi_model/vivit/dataset/MELD.Raw/train/train_splits/dia0_utt0.mp4"
container = av.open(file_path)   #读文件

#sample 32 frames
indices = sample_frame_indices(clip_len=32, frame_sample_rate=4, seg_len=container.streams.video[0].frames)
video = read_video_pyav(container=container, indices=indices)
print(video.shape)
image_processor = VivitImageProcessor.from_pretrained("/home/cv/Project1/cxh/multi_model/vivit/model/vivit-b-16x2-kinetics400",local_files_only=True)
model = VivitForVideoClassification.from_pretrained("/home/cv/Project1/cxh/multi_model/vivit/model/vivit-b-16x2-kinetics400",local_files_only=True)
#print(video[0].shape)
inputs1 = image_processor(list(video), return_tensors="pt")
#rint(inputs1["pixel_values"].shape,len(inputs1))
neew1 = torch.tensor([1])
inputs1["labels"]=neew1
inputs2 = image_processor(list(video), return_tensors="pt")
#print(inputs2["pixel_values"].shape,len(inputs2))
neew2 = torch.tensor([2])
inputs2["labels"]=neew2
#print(inputs)
input0={}
input0["0"] = inputs1
input0["1"] = inputs2
with torch.no_grad():
    outputs = model(**input0["1"])
    logits = outputs.logits

#model predicts one of the 400 Kinetics-400 classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])