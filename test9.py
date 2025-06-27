import cv2
import numpy as np
from ultralytics import YOLO
import torch
from scipy.optimize import linear_sum_assignment
from collections import defaultdict, deque
import pickle
import os
import warnings
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import logging

# Suppress warnings and handle OpenMP issue
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Fix OpenMP issue
os.environ['OMP_NUM_THREADS'] = '1'  # Limit OpenMP threads

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PlayerFeatures:
    """Store comprehensive player features for re-identification"""
    color_histogram: np.ndarray
    hog_features: np.ndarray
    position_history: deque
    bbox_aspect_ratio: float
    team_id: int  # 0: team1, 1: team2, 2: referee, 3: goalkeeper
    confidence_score: float
    last_seen_frame: int
    
class AdvancedPlayerTracker:
    def __init__(self, model_path: str, max_disappeared: int = 30):
        """
        Initialize the advanced player tracking system
        
        Args:
            model_path: Path to YOLOv11 model
            max_disappeared: Maximum frames a player can be missing before removal
        """
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            self.model = YOLO(model_path)
            self.max_disappeared = max_disappeared
            
            # Tracking data structures
            self.next_id = 0
            self.players = {}  # id -> PlayerFeatures
            self.disappeared = defaultdict(int)
            
            # Feature extractors
            self.hog = cv2.HOGDescriptor()
            
            # Tracking parameters
            self.position_weight = 0.3
            self.appearance_weight = 0.4
            self.team_weight = 0.2
            self.size_weight = 0.1
            
            # Kalman filters for position prediction
            self.kalman_filters = {}
            
            # Team color analysis
            self.team_colors = {}
            self.color_initialized = False
            
            logger.info("Tracker initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing tracker: {e}")
            raise
        
    def extract_color_histogram(self, image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """Extract color histogram features from player region"""
        try:
            if image is None or image.size == 0:
                return np.zeros(114)  # Default histogram size (50+32+32)
            
            # Ensure image is valid
            if len(image.shape) != 3 or image.shape[2] != 3:
                return np.zeros(114)
            
            # Convert to HSV for better color representation
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Calculate histogram for each channel with error handling
            try:
                hist_h = cv2.calcHist([hsv], [0], mask, [50], [0, 180])
                hist_s = cv2.calcHist([hsv], [1], mask, [32], [0, 256])
                hist_v = cv2.calcHist([hsv], [2], mask, [32], [0, 256])
            except cv2.error:
                return np.zeros(114)
            
            # Normalize and concatenate with safety checks
            hist_h = cv2.normalize(hist_h, hist_h).flatten()
            hist_s = cv2.normalize(hist_s, hist_s).flatten()
            hist_v = cv2.normalize(hist_v, hist_v).flatten()
            
            # Ensure consistent size
            if len(hist_h) != 50 or len(hist_s) != 32 or len(hist_v) != 32:
                return np.zeros(114)
            
            return np.concatenate([hist_h, hist_s, hist_v])
            
        except Exception as e:
            logger.warning(f"Error extracting color histogram: {e}")
            return np.zeros(114)
    
    def extract_hog_features(self, image: np.ndarray) -> np.ndarray:
        """Extract HOG features for player appearance"""
        try:
            if image is None or image.size == 0:
                return np.zeros(3780)
            
            # Ensure minimum size for HOG
            if image.shape[0] < 64 or image.shape[1] < 32:
                return np.zeros(3780)
            
            # Resize to standard size for consistent features
            resized = cv2.resize(image, (64, 128))
            
            # Convert to grayscale safely
            if len(resized.shape) == 3:
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            else:
                gray = resized
            
            # Extract HOG features with error handling
            try:
                features = self.hog.compute(gray)
                if features is not None and len(features) > 0:
                    return features.flatten()
                else:
                    return np.zeros(3780)
            except cv2.error:
                return np.zeros(3780)
                
        except Exception as e:
            logger.warning(f"Error extracting HOG features: {e}")
            return np.zeros(3780)
    
    def determine_team_id(self, color_hist: np.ndarray, bbox: Tuple[int, int, int, int], 
                         class_name: str) -> int:
        """Determine team ID based on uniform color and class"""
        try:
            if class_name == 'referee':
                return 2
            elif class_name == 'goalkeeper':
                return 3
            
            # For players, use color clustering to determine team
            if not self.color_initialized and len(self.team_colors) < 2:
                # Store color histograms for team identification
                team_id = len(self.team_colors)
                self.team_colors[team_id] = color_hist.copy()
                return team_id
            
            if len(self.team_colors) >= 2:
                # Compare with existing team colors
                similarities = []
                for team_id, team_color in self.team_colors.items():
                    try:
                        similarity = cv2.compareHist(color_hist, team_color, cv2.HISTCMP_CORREL)
                        similarities.append((team_id, similarity))
                    except cv2.error:
                        similarities.append((team_id, 0.0))
                
                if similarities:
                    # Return team with highest similarity
                    return max(similarities, key=lambda x: x[1])[0]
            
            return 0  # Default team
            
        except Exception as e:
            logger.warning(f"Error determining team ID: {e}")
            return 0
    
    def create_kalman_filter(self, x: float, y: float) -> cv2.KalmanFilter:
        """Create Kalman filter for position prediction"""
        try:
            kf = cv2.KalmanFilter(4, 2)
            kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                            [0, 1, 0, 0]], dtype=np.float32)
            kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                           [0, 1, 0, 1],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]], dtype=np.float32)
            kf.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)
            kf.measurementNoiseCov = 0.1 * np.eye(2, dtype=np.float32)
            kf.errorCovPost = 0.1 * np.eye(4, dtype=np.float32)
            kf.statePost = np.array([x, y, 0, 0], dtype=np.float32)
            return kf
        except Exception as e:
            logger.warning(f"Error creating Kalman filter: {e}")
            return None
    
    def predict_position(self, player_id: int) -> Tuple[float, float]:
        """Predict next position using Kalman filter"""
        try:
            if player_id in self.kalman_filters and self.kalman_filters[player_id] is not None:
                prediction = self.kalman_filters[player_id].predict()
                # Fix the deprecation warning by extracting scalar values properly
                if prediction is not None and len(prediction) >= 2:
                    x_pred = prediction[0].item() if hasattr(prediction[0], 'item') else float(prediction[0])
                    y_pred = prediction[1].item() if hasattr(prediction[1], 'item') else float(prediction[1])
                    return x_pred, y_pred
            return 0.0, 0.0
        except Exception as e:
            logger.warning(f"Error predicting position for player {player_id}: {e}")
            return 0.0, 0.0
    
    def update_kalman_filter(self, player_id: int, x: float, y: float):
        """Update Kalman filter with new measurement"""
        try:
            if player_id in self.kalman_filters and self.kalman_filters[player_id] is not None:
                measurement = np.array([[x], [y]], dtype=np.float32)
                self.kalman_filters[player_id].correct(measurement)
        except Exception as e:
            logger.warning(f"Error updating Kalman filter for player {player_id}: {e}")
    
    def calculate_similarity(self, features1: PlayerFeatures, features2: Dict, 
                           predicted_pos: Tuple[float, float]) -> float:
        """Calculate comprehensive similarity between player features"""
        try:
            # Position similarity
            pos1 = features1.position_history[-1] if features1.position_history else (0, 0)
            pos2 = features2.get('center', (0, 0))
            
            # Use predicted position if available
            if predicted_pos != (0.0, 0.0):
                pos_dist = np.sqrt((predicted_pos[0] - pos2[0])**2 + (predicted_pos[1] - pos2[1])**2)
            else:
                pos_dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
            
            pos_similarity = 1.0 / (1.0 + pos_dist / 100.0)  # Normalize by typical distance
            
            # Appearance similarity (color histogram)
            try:
                color_similarity = cv2.compareHist(features1.color_histogram, 
                                                 features2.get('color_hist', np.zeros(114)), 
                                                 cv2.HISTCMP_CORREL)
                color_similarity = max(0, color_similarity)  # Ensure non-negative
            except (cv2.error, ValueError):
                color_similarity = 0.0
            
            # HOG feature similarity with safety checks
            try:
                hog1 = features1.hog_features.flatten()
                hog2 = features2.get('hog_features', np.zeros(3780)).flatten()
                
                if len(hog1) == len(hog2) and len(hog1) > 0:
                    corr_matrix = np.corrcoef(hog1, hog2)
                    if corr_matrix.shape == (2, 2) and not np.isnan(corr_matrix[0, 1]):
                        hog_similarity = max(0, corr_matrix[0, 1])
                    else:
                        hog_similarity = 0.0
                else:
                    hog_similarity = 0.0
            except (ValueError, RuntimeWarning):
                hog_similarity = 0.0
            
            # Team consistency
            team_similarity = 1.0 if features1.team_id == features2.get('team_id', -1) else 0.0
            
            # Size consistency
            bbox_ratio_2 = features2.get('bbox_aspect_ratio', 1.0)
            size_diff = abs(features1.bbox_aspect_ratio - bbox_ratio_2)
            size_similarity = 1.0 / (1.0 + size_diff)
            
            # Weighted combination
            total_similarity = (self.position_weight * pos_similarity +
                              self.appearance_weight * color_similarity +
                              self.appearance_weight * hog_similarity +
                              self.team_weight * team_similarity +
                              self.size_weight * size_similarity)
            
            return max(0.0, min(1.0, total_similarity))  # Clamp between 0 and 1
            
        except Exception as e:
            logger.warning(f"Error calculating similarity: {e}")
            return 0.0
    
    def process_detections(self, frame: np.ndarray, frame_idx: int) -> List[Dict]:
        """Process YOLO detections and extract features"""
        detections = []
        
        try:
            if frame is None or frame.size == 0:
                return detections
            
            # Run YOLO detection with error handling
            try:
                results = self.model(frame, conf=0.3, iou=0.5, verbose=False)
            except Exception as e:
                logger.warning(f"YOLO detection failed for frame {frame_idx}: {e}")
                return detections
            
            for result in results:
                if result.boxes is None:
                    continue
                    
                boxes = result.boxes
                for box in boxes:
                    try:
                        # Get bounding box coordinates with safety checks
                        coords = box.xyxy[0].cpu().numpy()
                        if len(coords) < 4:
                            continue
                            
                        x1, y1, x2, y2 = map(int, coords)
                        
                        # Validate bounding box
                        if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0:
                            continue
                        if x2 > frame.shape[1] or y2 > frame.shape[0]:
                            continue
                        
                        conf = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Get class name safely
                        if class_id >= len(self.model.names):
                            continue
                        class_name = self.model.names[class_id]
                        
                        # Filter for relevant classes
                        if class_name not in ['person', 'player', 'goalkeeper', 'referee']:
                            continue
                        
                        # Extract player region with bounds checking
                        y1_safe = max(0, min(y1, frame.shape[0]-1))
                        y2_safe = max(y1_safe+1, min(y2, frame.shape[0]))
                        x1_safe = max(0, min(x1, frame.shape[1]-1))
                        x2_safe = max(x1_safe+1, min(x2, frame.shape[1]))
                        
                        player_region = frame[y1_safe:y2_safe, x1_safe:x2_safe]
                        
                        if player_region.size == 0:
                            continue
                        
                        # Extract features
                        color_hist = self.extract_color_histogram(player_region)
                        hog_features = self.extract_hog_features(player_region)
                        
                        # Calculate center and aspect ratio
                        center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        bbox_aspect_ratio = (x2 - x1) / max(1, (y2 - y1))
                        
                        # Determine team
                        team_id = self.determine_team_id(color_hist, (x1, y1, x2, y2), class_name)
                        
                        detection = {
                            'bbox': (x1, y1, x2, y2),
                            'center': center,
                            'confidence': conf,
                            'class_name': class_name,
                            'color_hist': color_hist,
                            'hog_features': hog_features,
                            'bbox_aspect_ratio': bbox_aspect_ratio,
                            'team_id': team_id,
                            'frame_idx': frame_idx
                        }
                        detections.append(detection)
                        
                    except Exception as e:
                        logger.warning(f"Error processing detection in frame {frame_idx}: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Error processing detections for frame {frame_idx}: {e}")
        
        return detections
    
    def assign_ids(self, detections: List[Dict], frame_idx: int) -> List[Dict]:
        """Assign IDs to detections using Hungarian algorithm"""
        if not detections:
            return []
        
        try:
            # If no existing players, assign new IDs
            if not self.players:
                for detection in detections:
                    player_id = self.next_id
                    self.next_id += 1
                    
                    # Create player features
                    features = PlayerFeatures(
                        color_histogram=detection['color_hist'],
                        hog_features=detection['hog_features'],
                        position_history=deque([detection['center']], maxlen=10),
                        bbox_aspect_ratio=detection['bbox_aspect_ratio'],
                        team_id=detection['team_id'],
                        confidence_score=detection['confidence'],
                        last_seen_frame=frame_idx
                    )
                    
                    self.players[player_id] = features
                    
                    # Create Kalman filter
                    x, y = detection['center']
                    kf = self.create_kalman_filter(float(x), float(y))
                    if kf is not None:
                        self.kalman_filters[player_id] = kf
                    
                    detection['player_id'] = player_id
                
                return detections
            
            # Create cost matrix for assignment
            num_players = len(self.players)
            num_detections = len(detections)
            
            if num_players == 0 or num_detections == 0:
                return detections
            
            cost_matrix = np.ones((num_players, num_detections)) * 1.0
            player_ids = list(self.players.keys())
            
            for i, player_id in enumerate(player_ids):
                predicted_pos = self.predict_position(player_id)
                
                for j, detection in enumerate(detections):
                    similarity = self.calculate_similarity(
                        self.players[player_id], detection, predicted_pos
                    )
                    cost_matrix[i, j] = 1.0 - similarity  # Convert to cost
            
            # Apply Hungarian algorithm with error handling
            try:
                row_indices, col_indices = linear_sum_assignment(cost_matrix)
            except ValueError as e:
                logger.warning(f"Hungarian algorithm failed: {e}")
                # Fallback: assign based on minimum distance
                row_indices, col_indices = [], []
                for j, detection in enumerate(detections):
                    min_cost = float('inf')
                    best_i = -1
                    for i in range(num_players):
                        if cost_matrix[i, j] < min_cost:
                            min_cost = cost_matrix[i, j]
                            best_i = i
                    if best_i >= 0:
                        row_indices.append(best_i)
                        col_indices.append(j)
            
            assigned_detections = []
            assigned_players = set()
            assigned_cols = set()
            
            # Process assignments
            for row, col in zip(row_indices, col_indices):
                if row < len(player_ids) and col < len(detections) and cost_matrix[row, col] < 0.7:
                    player_id = player_ids[row]
                    detection = detections[col]
                    
                    # Update player features
                    self.players[player_id].position_history.append(detection['center'])
                    self.players[player_id].last_seen_frame = frame_idx
                    self.players[player_id].confidence_score = detection['confidence']
                    
                    # Update appearance gradually
                    alpha = 0.1  # Learning rate
                    try:
                        self.players[player_id].color_histogram = (
                            (1 - alpha) * self.players[player_id].color_histogram +
                            alpha * detection['color_hist']
                        )
                    except ValueError:
                        pass  # Skip update if shapes don't match
                    
                    # Update Kalman filter
                    x, y = detection['center']
                    self.update_kalman_filter(player_id, float(x), float(y))
                    
                    # Reset disappeared counter
                    if player_id in self.disappeared:
                        del self.disappeared[player_id]
                    
                    detection['player_id'] = player_id
                    assigned_detections.append(detection)
                    assigned_players.add(player_id)
                    assigned_cols.add(col)
            
            # Handle unassigned detections (new players)
            for j, detection in enumerate(detections):
                if j not in assigned_cols:
                    player_id = self.next_id
                    self.next_id += 1
                    
                    features = PlayerFeatures(
                        color_histogram=detection['color_hist'],
                        hog_features=detection['hog_features'],
                        position_history=deque([detection['center']], maxlen=10),
                        bbox_aspect_ratio=detection['bbox_aspect_ratio'],
                        team_id=detection['team_id'],
                        confidence_score=detection['confidence'],
                        last_seen_frame=frame_idx
                    )
                    
                    self.players[player_id] = features
                    
                    # Create Kalman filter
                    x, y = detection['center']
                    kf = self.create_kalman_filter(float(x), float(y))
                    if kf is not None:
                        self.kalman_filters[player_id] = kf
                    
                    detection['player_id'] = player_id
                    assigned_detections.append(detection)
            
            # Handle disappeared players
            for player_id in player_ids:
                if player_id not in assigned_players:
                    self.disappeared[player_id] += 1
                    
                    # Remove players that have been gone too long
                    if self.disappeared[player_id] > self.max_disappeared:
                        if player_id in self.players:
                            del self.players[player_id]
                        if player_id in self.disappeared:
                            del self.disappeared[player_id]
                        if player_id in self.kalman_filters:
                            del self.kalman_filters[player_id]
            
            return assigned_detections
        
        except Exception as e:
            logger.error(f"Error in ID assignment for frame {frame_idx}: {e}")
            return detections
    
    def track_video(self, video_path: str, output_path: str = None, 
                   save_results: bool = True) -> Dict:
        """Track players throughout the entire video"""
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Processing video: {total_frames} frames at {fps} FPS ({width}x{height})")
            
            # Initialize video writer if output path provided
            out = None
            if output_path:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    if not out.isOpened():
                        logger.warning(f"Could not create output video: {output_path}")
                        out = None
                except Exception as e:
                    logger.warning(f"Error creating video writer: {e}")
                    out = None
            
            tracking_results = {}
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                try:
                    # Process detections and assign IDs
                    detections = self.process_detections(frame, frame_idx)
                    tracked_detections = self.assign_ids(detections, frame_idx)
                    
                    # Store results
                    tracking_results[frame_idx] = tracked_detections
                    
                    # Draw tracking results on frame
                    if out is not None or output_path:
                        annotated_frame = self.draw_tracks(frame, tracked_detections)
                        
                        # Write frame if output specified and writer is valid
                        if out is not None:
                            out.write(annotated_frame)
                    
                    # Progress update
                    if frame_idx % 30 == 0 or frame_idx == total_frames - 1:
                        logger.info(f"Processed frame {frame_idx + 1}/{total_frames}")
                    
                except Exception as e:
                    logger.error(f"Error processing frame {frame_idx}: {e}")
                    # Store empty results for failed frame
                    tracking_results[frame_idx] = []
                
                frame_idx += 1
            
            cap.release()
            if out is not None:
                out.release()
            
            logger.info("Video processing completed")
            
            # Save tracking results
            if save_results:
                try:
                    results_path = video_path.replace('.mp4', '_tracking_results.pkl')
                    with open(results_path, 'wb') as f:
                        pickle.dump(tracking_results, f)
                    logger.info(f"Results saved to: {results_path}")
                except Exception as e:
                    logger.warning(f"Could not save results: {e}")
            
            return tracking_results
        
        except Exception as e:
            logger.error(f"Error in video tracking: {e}")
            raise
    
    def draw_tracks(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw tracking information on frame"""
        try:
            annotated_frame = frame.copy()
            
            # Define colors for different teams
            team_colors = {
                0: (0, 255, 0),    # Green for team 1
                1: (255, 0, 0),    # Blue for team 2
                2: (0, 255, 255),  # Yellow for referee
                3: (255, 0, 255)   # Magenta for goalkeeper
            }
            
            for detection in detections:
                try:
                    x1, y1, x2, y2 = detection['bbox']
                    player_id = detection['player_id']
                    team_id = detection.get('team_id', 0)
                    confidence = detection.get('confidence', 0.0)
                    
                    # Get color for team
                    color = team_colors.get(team_id, (255, 255, 255))
                    
                    # Ensure coordinates are within frame bounds
                    x1 = max(0, min(x1, frame.shape[1] - 1))
                    y1 = max(0, min(y1, frame.shape[0] - 1))
                    x2 = max(x1 + 1, min(x2, frame.shape[1]))
                    y2 = max(y1 + 1, min(y2, frame.shape[0]))
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw player ID and info
                    label = f"ID:{player_id} T:{team_id} ({confidence:.2f})"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    
                    # Ensure label position is within frame
                    label_y1 = max(label_size[1] + 10, y1)
                    label_x2 = min(x1 + label_size[0], frame.shape[1])
                    
                    cv2.rectangle(annotated_frame, (x1, label_y1 - label_size[1] - 10), 
                                 (label_x2, label_y1), color, -1)
                    cv2.putText(annotated_frame, label, (x1, label_y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    
                    # Draw trajectory
                    if player_id in self.players:
                        positions = list(self.players[player_id].position_history)
                        if len(positions) > 1:
                            for i in range(1, len(positions)):
                                pt1 = positions[i-1]
                                pt2 = positions[i]
                                # Ensure points are within frame bounds
                                pt1 = (max(0, min(pt1[0], frame.shape[1]-1)), 
                                      max(0, min(pt1[1], frame.shape[0]-1)))
                                pt2 = (max(0, min(pt2[0], frame.shape[1]-1)), 
                                      max(0, min(pt2[1], frame.shape[0]-1)))
                                cv2.line(annotated_frame, pt1, pt2, color, 2)
                
                except Exception as e:
                    logger.warning(f"Error drawing detection: {e}")
                    continue
            
            return annotated_frame
        
        except Exception as e:
            logger.error(f"Error drawing tracks: {e}")
            return frame

# Main execution function
def main():
    """Main function to run the tracking system"""
    # Configuration
    MODEL_PATH = "best.pt"  # Your YOLOv11 model path
    VIDEO_PATH = "15sec.mp4"  # Your input video
    OUTPUT_PATH = "tracked_output.mp4"  # Output video with tracking
    
    # Initialize tracker
    tracker = AdvancedPlayerTracker(
        model_path=MODEL_PATH,
        max_disappeared=30  # Allow 1 second at 30fps for re-identification
    )
    
    # Process video
    try:
        results = tracker.track_video(
            video_path=VIDEO_PATH,
            output_path=OUTPUT_PATH,
            save_results=True
        )
        
        print(f"Tracking completed successfully!")
        print(f"Tracked {len(results)} frames")
        print(f"Output saved to: {OUTPUT_PATH}")
        
        # Print summary statistics
        total_players = set()
        for frame_detections in results.values():
            for detection in frame_detections:
                total_players.add(detection['player_id'])
        
        print(f"Total unique players tracked: {len(total_players)}")
        
    except Exception as e:
        logger.error(f"Error during tracking: {e}")
        raise

if __name__ == "__main__":
    main()