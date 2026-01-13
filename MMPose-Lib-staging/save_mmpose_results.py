import json
import os
import re
from pathlib import Path
import cv2
import numpy as np

from mmpose.evaluation.functional import nms
from mmpose.apis import inference_topdown, init_model as init_pose_estimator
from mmengine.registry import init_default_scope
from mmdet.apis import inference_detector, init_detector

scenes_dir = "C:\\Users\\samso\\MMPose-Lib\\Pedestrians"
output_dir = "./hrnet_ProcessedScenes"

characters_subdir   = "Charachters"
Video_output_subdir = "."

height = 1080
width = 1920

# Body parts are represented by their start point
body_parts = [  'RightArm'      , 'LeftArm',      # 0 , 1 
                'RightForeArm'  , 'LeftForeArm',  # 2 , 3
                'RightHand'     , 'LeftHand',     # 4 , 5
                'RightUpLeg'    , 'LeftUpLeg',    # 6 , 7
                'RightLeg'      , 'LeftLeg',      # 8 , 9
                'RightFoot'     , 'LeftFoot']     # 10, 11

parts_mmpose_indices = {
    'RightArm'      : 0,
    'LeftArm'       : 1,
    'RightForeArm'  : 2,
    'LeftForeArm'   : 3,
    'RightHand'     : 4,
    'LeftHand'      : 5,
    'RightUpLeg'    : 6,
    'LeftUpLeg'     : 7,
    'RightLeg'      : 8,
    'LeftLeg'       : 9,
    'RightFoot'     : 10,
    'LeftFoot'      : 11
}

bones = {
    'RightArm'      : ['RightArm'       , 'RightForeArm', 0.5], #[start, end, estimated length]
    'LeftArm'       : ['LeftArm'        , 'LeftForeArm' , 0.5],
    'RightForeArm'  : ['RightForeArm'   , 'RightHand'   , 0.5],
    'LeftForeArm'   : ['LeftForeArm'    , 'LeftHand'    , 0.5],
    'RightUpLeg'    : ['RightUpLeg'     , 'RightLeg'    , 0.5],
    'LeftUpLeg'     : ['LeftUpLeg'      , 'LeftLeg'     , 0.5],
    'RightLeg'      : ['RightLeg'       , 'RightFoot'   , 0.5],
    'LeftLeg'       : ['LeftLeg'        , 'LeftFoot'    , 0.5]
}

bones_idx = {i: [body_parts.index(s), body_parts.index(e), l] for i, (s, e, l) in bones.items()}

data_files_names = {"right":"CamRight.csv",
                    "left":"CamLeft.csv"}

images_dirs = {"right":"Right",
                "left":"Left"}

# MMPose Configuration

pose_config     = 'C:\\Users\\samso\\MMPose-Lib\\my_mmpose_configs\\pedestrian_finetune_hrnet_heatmap.py'
pose_checkpoint = 'C:\\Users\\samso\\MMPose-Lib\\work_dirs\\pedestrian_finetune_hrnet_heatmap\\best_coco_AP_epoch_15.pth'

det_config      = 'C:\\Users\\samso\\MMPose-Lib\\det-config.py'
det_checkpoint  = 'C:\\Users\\samso\\MMPose-Lib\\det-weights.pth'

device = 'cuda:0'

cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))

fps = 30
write_output = False
write_individual_frames = False

# Initialize MMPose

print("Initializing detector...")
detector = init_detector(
    det_config,
    det_checkpoint,
    device=device
)

print("Initializing pose estimator...")
pose_estimator = init_pose_estimator(
    pose_config,
    pose_checkpoint,
    device=device,
    cfg_options=cfg_options
)

# MMPose utilities

def get_pose_results(img, detector, pose_estimator):
    scope = detector.cfg.get('default_scope', 'mmdet')
    if scope is not None:
        init_default_scope(scope)
    detect_result = inference_detector(detector, img)
    pred_instance = detect_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == 0, pred_instance.scores > 0.3)]
    bboxes = bboxes[nms(bboxes, 0.3)][:, :4]
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    return pose_results

def extract_keypoints(pose_results):
    points = [[] for _ in range(len(pose_results))]
    for body_part in body_parts:
        part_index = parts_mmpose_indices[body_part]
        for i, pose in enumerate(pose_results):
            if pose.pred_instances.keypoints.shape[0] > 0:
                points[i].append(pose.pred_instances.keypoints[0][part_index][:2])
            else:
                points[i].append([-1, -1])
    return np.array(points, dtype=np.float32)

def extract_bboxes(pose_results):
    return np.array([p.pred_instances.bboxes[0] for p in pose_results])

scenes = [i for i in os.listdir(scenes_dir) if re.match(r"^Scene.+", i)]

characters = {}

for scene in scenes:
    characters_dir = os.path.join(scenes_dir, scene, characters_subdir)
    if not os.path.isdir(characters_dir):
        continue
    characters[scene] = os.listdir(characters_dir)

for scene in scenes:
    print(f"Processing scene: {scene}")
    right_images = Path(scenes_dir) / scene / images_dirs["right"]
    left_images = Path(scenes_dir) / scene / images_dirs["left"]
    if not right_images.exists() or not left_images.exists():
        print(f"Skipping scene {scene} due to missing images.")
        continue
    right_images = sorted(right_images.glob("frame*.png"))
    left_images = sorted(left_images.glob("frame*.png"))
    
    if len(right_images) != len(left_images):
        print(f"Skipping scene {scene} due to mismatched image counts.")
        continue
    
    right_output_path = Path(output_dir) / scene / images_dirs["right"]
    left_output_path = Path(output_dir) / scene / images_dirs["left"]
    
    right_output_path.mkdir(parents=True, exist_ok=True)
    left_output_path.mkdir(parents=True, exist_ok=True)
    
    right_results = []
    left_results = []
    
    for frame_idx, (right_img_path, left_img_path) in enumerate(zip(right_images, left_images)):
        if frame_idx % 20 == 0:
            print(f"Processing frame {frame_idx} in scene {scene}...")
        right_img = cv2.imread(str(right_img_path))
        left_img = cv2.imread(str(left_img_path))
        if right_img is None or left_img is None:
            print(f"Skipping frame {frame_idx} in scene {scene} due to image read error.")
            continue
        
        right_pose_results = get_pose_results(right_img, detector, pose_estimator)
        left_pose_results = get_pose_results(left_img, detector, pose_estimator)
        
        right_keypoints = extract_keypoints(right_pose_results)
        left_keypoints = extract_keypoints(left_pose_results)
        
        right_bboxes = extract_bboxes(right_pose_results)
        left_bboxes = extract_bboxes(left_pose_results)
        
        right_frame_result = {
            "frame_index": frame_idx,
            "keypoints": right_keypoints.tolist(),
            "bboxes": right_bboxes.tolist()
        }
        
        left_frame_result = {
            "frame_index": frame_idx,
            "keypoints": left_keypoints.tolist(),
            "bboxes": left_bboxes.tolist()
        }
        
        right_results.append(right_frame_result)
        left_results.append(left_frame_result)
        
        if write_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            right_out_writer = cv2.VideoWriter(
                str(Path(output_dir) / f"{scene}_right_output.mp4"),
                fourcc, fps, (width, height)
            )
            left_out_writer = cv2.VideoWriter(
                str(Path(output_dir) / f"{scene}_left_output.mp4"),
                fourcc, fps, (width, height)
            )
            
        if write_individual_frames or write_output:
            for i, keypoints in enumerate(right_keypoints):
                for i, keypoint in enumerate(keypoints):
                    cv2.circle(right_img, (int(keypoint[0]), int(keypoint[1])), 5, (0, 255, 0), -1)
            for i, keypoints in enumerate(left_keypoints):
                for i, keypoint in enumerate(keypoints):
                    cv2.circle(left_img, (int(keypoint[0]), int(keypoint[1])), 5, (0, 255, 0), -1)
                
            for i, bbox in enumerate(right_bboxes):
                cv2.rectangle(right_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            for i, bbox in enumerate(left_bboxes):
                cv2.rectangle(left_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            
            for i, keypoints in enumerate(right_keypoints):
                for bone, (start, end, _) in bones.items():
                    start_point = (int(keypoints[body_parts.index(start)][0]), int(keypoints[body_parts.index(start)][1]))
                    end_point = (int(keypoints[body_parts.index(end)][0]), int(keypoints[body_parts.index(end)][1]))
                    cv2.line(right_img, start_point, end_point, (0, 255, 255), 2)
                    
            for i, keypoints in enumerate(left_keypoints):
                for bone, (start, end, _) in bones.items():
                    start_point = (int(keypoints[body_parts.index(start)][0]), int(keypoints[body_parts.index(start)][1]))
                    end_point = (int(keypoints[body_parts.index(end)][0]), int(keypoints[body_parts.index(end)][1]))
                    cv2.line(left_img, start_point, end_point, (0, 255, 255), 2)
                
            if write_individual_frames:
                cv2.imwrite(str(right_output_path / f"frame_{frame_idx}.png"), right_img)
                cv2.imwrite(str(left_output_path / f"frame_{frame_idx}.png"), left_img)
            
            if write_output:
                right_out_writer.write(right_img)
                left_out_writer.write(left_img)
        
    if write_output:
        right_out_writer.release()
        left_out_writer.release()
            
    with open(right_output_path / f"{scene}_right.json", "w") as f:
        json.dump(right_results, f, indent=4)
    
    with open(left_output_path / f"{scene}_left.json", "w") as f:
        json.dump(left_results, f, indent=4)
        
if write_output:
    print(f"Output videos and JSON files saved in {output_dir}.")
    print(f"Right videos: {Path(output_dir) / f'{scene}_right_output.mp4'}")
    print(f"Left videos: {Path(output_dir) / f'{scene}_left_output.mp4'}")
    print(f"Right JSON: {Path(output_dir) / f'{scene}_right.json'}")
    print(f"Left JSON: {Path(output_dir) / f'{scene}_left.json'}")
    
else:
    print("No output video was created, as no frames were processed.")
    
print("Processing completed for all scenes.")