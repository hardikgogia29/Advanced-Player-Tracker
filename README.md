# Advanced Sports Player Tracking System

## Overview
This project implements a robust multi-player tracking system for sports videos using computer vision techniques. It combines YOLO-based detection with appearance features, position prediction, and team identification to track players, referees, and goalkeepers across video frames.

## Key Features
- **Multi-object tracking** using YOLOv11 detection
- **Feature fusion** combining:
  - Color histograms (HSV space)
  - HOG appearance features
  - Position history
  - Bounding box aspect ratios
  - Team identification
- **Kalman filters** for position prediction
- **Hungarian algorithm** for optimal player-detection assignment
- **Team classification** based on uniform color clustering
- **Trajectory visualization** with frame-by-frame tracking

## Environment Requirements
  -Python 3.8+
  -PyTorch 2.0+
  -Ultralytics (for YOLO)
  -OpenCV 4.5+
  -Scipy
  -Numpy

## Installation
1. Clone repository:
git clone https://github.com/your-username/sports-player-tracking.git

2. Install dependencies:
pip install -r requirements.txt

3. Download pretrained YOLO model (best.pt) and place in project root:
https://drive.google.com/drive/folders/1tlpbcqMnr7P-fZLaJmMCeynPCLnQOBt2?usp=drive_link

4. Command Line Execution
python Final.py

## Configuration Options
| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_path` | Required | Path to YOLO model (.pt) |
| `max_disappeared` | 30 | Frames before removing missing players |
| `position_weight` | 0.3 | Position similarity weight |
| `appearance_weight` | 0.4 | Color/HOG similarity weight |
| `team_weight` | 0.2 | Team consistency weight |
| `size_weight` | 0.1 | Bounding box size weight |

# Approach
### Tracking Pipeline
1. **Detection**:
   - YOLO detects players/referees/goalkeepers per frame
   - Bounding box validation and filtering
2. **Feature Extraction**:
   - **Color Histograms**: HSV space with 50+32+32 bins
   - **HOG Features**: Standard 64×128 descriptor
   - **Position History**: Last 10 positions stored
   - **Team ID**: Uniform color clustering
3. **Assignment**:
   - Cost matrix combining position, appearance, team, and size metrics
   - Optimal assignment via Hungarian algorithm
4. **Prediction**:
   - Kalman filters for position prediction
   - Continuous position correction
5. **Lifecycle Management**:
   - New player registration
   - Disappeared player removal after `max_disappeared` frames

### Visualization
The system generates output videos with:
- Player bounding boxes (color-coded by team)
- Unique player IDs
- Confidence scores
- Position trajectories

## Results
- Tracking results saved as `_tracking_results.pkl`
- Annotated video output with player trajectories
- Console logging of processing progress

## Limitations
- Requires clear player visibility
- Performance depends on YOLO detection quality
- Team identification requires distinct uniform colors

