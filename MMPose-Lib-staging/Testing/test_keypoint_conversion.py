import cv2
import numpy as np
from mmpose.apis import init_model, inference_topdown
from mmpose.utils.visualization import visualize_pose

def test_keypoint_conversion():
    # Initialize the model
    config_file = 'configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
    checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
    
    model = init_model(config_file, checkpoint, device='cuda:0')
    
    # Load a test image
    img = cv2.imread('demo/resources/pose_demo.jpg')
    if img is None:
        print("Error: Could not load image")
        return
    
    # Run inference with original keypoints
    results_original = inference_topdown(model, img, simplify_keypoints=False)
    
    # Run inference with simplified keypoints
    results_simplified = inference_topdown(model, img, simplify_keypoints=True)
    
    # Visualize results
    img_original = visualize_pose(img.copy(), results_original[0])
    img_simplified = visualize_pose(img.copy(), results_simplified[0])
    
    # Save results
    cv2.imwrite('pose_original.jpg', img_original)
    cv2.imwrite('pose_simplified.jpg', img_simplified)
    
    # Print keypoint information
    print("\nOriginal keypoints:")
    print(f"Number of keypoints: {len(results_original[0].pred_instances.keypoints)}")
    print(f"Keypoint names: {results_original[0].metainfo['keypoint_names']}")
    
    print("\nSimplified keypoints:")
    print(f"Number of keypoints: {len(results_simplified[0].pred_instances.keypoints)}")
    print(f"Keypoint names: {results_simplified[0].metainfo['keypoint_names']}")
    
    # Print timing information
    print("\nTiming information:")
    print(f"Original inference time: {results_original[0].metainfo.get('inference_time', 'N/A')}")
    print(f"Simplified inference time: {results_simplified[0].metainfo.get('inference_time', 'N/A')}")

if __name__ == '__main__':
    test_keypoint_conversion() 