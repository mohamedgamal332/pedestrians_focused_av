import time
import cv2
import numpy as np
from mmpose.apis.inference import init_model, inference_topdown, inference_bottomup
from mmengine.config import Config
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB

def test_model_loading(config_path, checkpoint_url, num_runs=3):
    print(f"\n{'='*50}")
    print(f"Testing model loading with {num_runs} runs")
    print(f"{'='*50}")
    
    # Test without caching
    print("\n1. Without caching:")
    times = []
    for i in range(num_runs):
        start_mem = get_memory_usage()
        t0 = time.time()
        model = init_model(config_path, checkpoint_url, enable_cache=False)
        t1 = time.time()
        end_mem = get_memory_usage()
        times.append(t1 - t0)
        print(f"Run {i+1}: {t1-t0:.2f}s, Memory: {end_mem-start_mem:.2f}MB")
        del model
    
    print(f"Average time without caching: {sum(times)/len(times):.2f}s")
    
    # Test with caching
    print("\n2. With caching:")
    times = []
    for i in range(num_runs):
        start_mem = get_memory_usage()
        t0 = time.time()
        model = init_model(config_path, checkpoint_url, enable_cache=True)
        t1 = time.time()
        end_mem = get_memory_usage()
        times.append(t1 - t0)
        print(f"Run {i+1}: {t1-t0:.2f}s, Memory: {end_mem-start_mem:.2f}MB")
    
    print(f"Average time with caching: {sum(times)/len(times):.2f}s")
    
    # Test with quantization
    print("\n3. With quantization:")
    times = []
    for i in range(num_runs):
        start_mem = get_memory_usage()
        t0 = time.time()
        model = init_model(config_path, checkpoint_url, enable_cache=True, quantize=True)
        t1 = time.time()
        end_mem = get_memory_usage()
        times.append(t1 - t0)
        print(f"Run {i+1}: {t1-t0:.2f}s, Memory: {end_mem-start_mem:.2f}MB")
    
    print(f"Average time with quantization: {sum(times)/len(times):.2f}s")
    
    return model

def test_inference_speed(model, img_path, num_runs=5):
    print(f"\n{'='*50}")
    print(f"Testing inference speed with {num_runs} runs")
    print(f"{'='*50}")
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image: {img_path}")
        return
    
    times = []
    for i in range(num_runs):
        t0 = time.time()
        results = inference_topdown(model, img)
        t1 = time.time()
        times.append(t1 - t0)
        print(f"Run {i+1}: {t1-t0:.2f}s")
    
    print(f"Average inference time: {sum(times)/len(times):.2f}s")

def main():
    # Test top-down model
    print("\nTesting Top-down Model")
    config_path = 'configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
    checkpoint_url = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
    
    model = test_model_loading(config_path, checkpoint_url)
    test_inference_speed(model, 'demo/resources/sunglasses.jpg')
    
    # Test bottom-up model
    print("\nTesting Bottom-up Model")
    config_path = 'configs/body_2d_keypoint/associative_embedding/coco/ae_hrnet-w32_8xb24-300e_coco-512x512.py'
    checkpoint_url = 'https://download.openmmlab.com/mmpose/bottom_up/hrnet_w32_coco_512x512-bcb8c247_20200816.pth'
    
    model = test_model_loading(config_path, checkpoint_url)
    test_inference_speed(model, 'demo/resources/sunglasses.jpg')

if __name__ == '__main__':
    main() 