Hi I'm working on a project with the main objective of pedestrian localization and trajectory prediction relying on pose in a stereo camera setup.
The data is generated using carla. Two cameras on the same plane 1 m horizontally apart record synchronously. The pose is then detected using mmpose.

Now using triangulation, we can obtain the 3D pose of the skeleton, and we can predict future motion by interpolating from previous frames.
The problems: 
1- few pixels difference can cause tens of centimeters difference in the world position.
2- Pedestrian motion along with the vehicle motion makes using normal interpolation for trajectory / motion prediction is not reliable.
3- pedestrian my be hidden in the frame of missed during pose detection in one or both camera feeds for the span of several frames. being able to recognize a pedestrian over the span of several frames is needed.

Proposed solution: LSTM for calculating motion residuals
Now there are four components of this project:
1- The ability to match a pedestrian from the two feeds.
2- The ability to track the pedestrian across the consecutive frames.
3- The ability to project the motion into the future predicting the upcoming frames.
4- The ability to use the projected motion to correct any deviations in the triangulated skeleton that resulted from pixel deviations during pose detection process.

An input to the LSTM is from two sources:
1- From the previous layer: time encoded previous skeletal point positions in the two feeds, previous confidence, previous 3D pose, predicted 3D pose. (Hidden: estimated bone lengths kept symmetric)
2- New Input: new skeleton point positions, confidence scores, triangulated 3D pose.

First part is to compare the last LSTM results with the new results to identify a match. this can be decided via an mlp to get score comparing the two new feeds with the two old ones + an mlp to compare the two feeds against each other.

Now the output from the LSTM data and the new data of a match, should be a skeletal map that:
1- blends the predicted and triangulated skeletal nodes based on factors like confidence scores to minimize the error due to pose detection misses.
2- abide by the estimated bone lengths of the skeleton.
+ the predicted skeleton over several future frames.
3- reject the specific runs where there is a significant deviations from prediction / problems with the new data.

Now For how the residuals for predictions are handled, keep a range of possible magnitudes of change based on the pedestrian + the vehicle speeds, and try to keep an estimate of the magnitude of the joints within that range to be tha magnitude for the residuals of prediction. based on these limits on the 3D pose + the confidence values you can determine how much to trust the detection.

Expected statistics:
Triangulation accuracy per distance from camera per number of nodes with confidence above N (maybe 0.3) + overall accuracy.
The accuracy of the projected skeleton measured against the detected skeleton per distance from camera per number of nodes with confidence above N (maybe 0.3) + overall accuracy.
The improvement of the blended skeletal points compared to the GT against pure triangulation. + any other important statistics.

Things to take note of:
- Try to limit resources consumption to 10GB Ram + 4GB vRam.
- Try to include all the features in the baseline.
- The pedestrian ids in the evaluated skeletons aren't always correct.
- You are not tracking a pedestrian through out the whole recording. pedestrians enter and exit the frame, and may get occluded for some time. add a mechanism for discovery, deposal, and reidentification.
- Create a complete script with all the necessary functionalities including an implementation of the baseline. don't use the output from the baseline i provided. implement it so that you are processing frames fully as you go.


# The Baseline (baseline.py) output
```
======================================================================
STEREO PEDESTRIAN LOCALIZATION BASELINE
======================================================================

Configuration:
  Ground Truth Dir: Data-CarlaGT-Mini
  Detection Dir: Data-RTMPoseEvaluated-Mini
  Output Dir: Baseline_Output
  Min Confidence: 0.35

======================================================================
LOADING DATASET
======================================================================
  ✓ Loaded: S2 (1000 frames)

Total scenes loaded: 1

Updated camera parameters from intrinsics:
  Focal Length: 960.0
  Principal Point: (960.0, 540.0)
  Baseline: 1m

======================================================================
RUNNING BASELINE EVALUATION
======================================================================

──────────────────────────────────────────────────
Processing: S2
  Frames: 1000
  Frame range: 164512 - 165511
──────────────────────────────────────────────────
Frames: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:41<00:00, 24.06it/s]

  Scene Statistics:
    Avg detections (L/R): 3.1 / 2.9
    Avg raw matches: 2.18
    Avg confident matches: 2.13

======================================================================
EVALUATION SUMMARY
======================================================================

📊 MATCHING PERFORMANCE:
----------------------------------------
  Ground Truth Matches:  2241
  Predicted Matches:     2129
  Correct Matches:       1957
  Precision:             0.9192
  Recall:                0.8733
  F1 Score:              0.8957

📍 LOCALIZATION PERFORMANCE:
----------------------------------------
  Mean Position Error:   0.8181 m
  Std Position Error:    2.5944 m
  Median Position Error: 0.5257 m
  Mean Depth Error:      0.3959 m
  Mean Keypoint Error:   0.9118 m
  Samples Evaluated:     1957

🦴 BONE LENGTH ERRORS (relative):
----------------------------------------
  left_elbow_left_wrist         : 0.3854
  left_eye_left_ear             : 5.8098
  left_hip_left_knee            : 0.3213
  left_hip_right_hip            : 0.5879
  left_knee_left_ankle          : 0.3081
  left_shoulder_left_elbow      : 0.3863
  left_shoulder_left_hip        : 0.2053
  left_shoulder_right_shoulder  : 0.4641
  nose_left_eye                 : 8.8796
  nose_right_eye                : 12.0591
  right_elbow_right_wrist       : 0.3817
  right_eye_right_ear           : 6.4700
  right_hip_right_knee          : 0.3255
  right_knee_right_ankle        : 0.3106
  right_shoulder_right_elbow    : 0.3766
  right_shoulder_right_hip      : 0.2049

======================================================================

📏 METRICS BY DISTANCE:
----------------------------------------------------------------------
Distance        Count      Pos Error       Depth Error    
----------------------------------------------------------------------
0-10m           87         0.9012          0.4543         
10-20m          515        1.0350          0.5590         
20-30m          638        0.6623          0.2145         
30-50m          717        0.7909          0.4331         

🦴 BONE ERRORS BY DISTANCE (excluding head):
------------------------------------------------------------------------------------------
Bone                           0-10m        10-20m       20-30m       30-50m      
------------------------------------------------------------------------------------------
left_shoulder_right_shoulder   0.3076       0.2909       0.3579       0.7040      
left_hip_right_hip             0.3318       0.3469       0.4683       0.9084      
left_shoulder_left_hip         0.1435       0.1364       0.1764       0.2906      
right_shoulder_right_hip       0.1473       0.1430       0.1693       0.2891      
left_shoulder_left_elbow       0.2149       0.2200       0.3117       0.6036      
right_shoulder_right_elbow     0.2411       0.1901       0.2904       0.6202      
left_elbow_left_wrist          0.2611       0.2513       0.3679       0.5262      
right_elbow_right_wrist        0.2531       0.2337       0.3515       0.5500      
left_hip_left_knee             0.1877       0.1554       0.2710       0.5075      
right_hip_right_knee           0.1793       0.1491       0.2751       0.5163      
left_knee_left_ankle           0.1761       0.1962       0.2712       0.4399      
right_knee_right_ankle         0.1777       0.1919       0.2735       0.4456      

🎯 METRICS BY CONFIDENCE:
--------------------------------------------------
Confidence           Count      Pos Error      
--------------------------------------------------
0.3-0.6              46         5.7296         
0.6-0.9              1002       0.8341         
0.9-1.0              909        0.5519         

📈 CONFIDENCE VS ERROR CORRELATION:
--------------------------------------------------
  Conf 0.4-0.6: n=  46, mean_err=5.730m, median_err=1.314m
  Conf 0.6-0.8: n= 169, mean_err=1.456m, median_err=0.750m
  Conf 0.8-1.0: n=1742, mean_err=0.627m, median_err=0.520m

📊 PIPELINE STATISTICS:
--------------------------------------------------
  Total position samples: 1957
  Errors < 0.5m: 591 (30.2%)
  Errors < 1.0m: 1829 (93.5%)
  Errors > 2.0m: 47 (2.4%)

👥 PER-PEDESTRIAN CONSISTENCY:
--------------------------------------------------
  Pedestrians with 5+ samples: 58
  Consistent tracking (std < 0.5m): 35 (60.3%)

======================================================================
Saved 260 sequences to Baseline_Output/training_sequences.json

Saved evaluation summary to Baseline_Output/evaluation_summary.json

======================================================================
BASELINE COMPLETE
======================================================================

This baseline achieves:
  • Matching F1: 0.8957
  • Mean Position Error: 0.8181 m
  • Median Position Error: 0.5257 m
```

# Glaring problems:

## Sympromps: Fast training + error increases al lower loss

## Explanation: Most likely you are either removing most of the data, or a problem with the archetecture / loss functions.

- Try keeping as much of the detectable skeletons as possible
- Limit the changes you can make to the (3D skeleton not the 2D) by confidence of detection, normality of the triangulated pose. and try to arrive with the bone lengths in the 3D skeletons to the estimated bone lengths.

# Missing features.

- Missing the errors across the future predections of the 3D skeleton position.
- I need an estimate of the trajectory relative to the car and evaluate the estimation in the results.