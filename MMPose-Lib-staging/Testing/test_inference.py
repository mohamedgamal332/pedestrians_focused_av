import cv2
import numpy as np
from mmpose.apis.inference import init_model, inference_topdown, inference_bottomup, collect_multi_frames
from mmengine.config import Config

def test_topdown_inference():
    # Load config and model
    config = Config.fromfile('configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py')
    model = init_model(config, checkpoint='https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth')
    
    # Load a test image
    img = cv2.imread('demo/resources/sunglasses.jpg')
    if img is None:
        print("Please provide a valid image path")
        return
    
    # Run inference
    results = inference_topdown(model, img)
    print("\nTop-down inference results:", results)

def test_bottomup_inference():
    # Load config and model - using associative embedding instead of bottomup heatmap
    config = Config.fromfile('configs/body_2d_keypoint/associative_embedding/coco/ae_hrnet-w32_8xb24-300e_coco-512x512.py')
    model = init_model(config, checkpoint='https://download.openmmlab.com/mmpose/bottom_up/hrnet_w32_coco_512x512-bcb8c247_20200816.pth')
    
    # Load a test image
    img = cv2.imread('demo/resources/sunglasses.jpg')
    if img is None:
        print("Please provide a valid image path")
        return
    
    # Run inference
    results = inference_bottomup(model, img)
    print("\nBottom-up inference results:", results)

def test_multi_frame_collection():
    # Load a test video
    video = cv2.VideoCapture('demo/resources/demo.mp4')
    if not video.isOpened():
        print("Please provide a valid video path")
        return
    
    # Test frame collection
    frame_id = 10  # middle frame
    indices = [-2, -1, 0, 1, 2]  # collect 2 frames before and after
    
    # Test online mode
    print("\nTesting online mode:")
    frames = collect_multi_frames(video, frame_id, indices, online=True)
    print(f"Collected {len(frames)} frames in online mode")
    
    # Test offline mode
    print("\nTesting offline mode:")
    frames = collect_multi_frames(video, frame_id, indices, online=False)
    print(f"Collected {len(frames)} frames in offline mode")
    
    video.release()

if __name__ == '__main__':
    print("Testing top-down inference...")
    test_topdown_inference()
    
    print("\nTesting bottom-up inference...")
    test_bottomup_inference()
    
    print("\nTesting multi-frame collection...")
    test_multi_frame_collection() 