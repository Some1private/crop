#!/usr/bin/env python3

import cv2
import mediapipe as mp
import numpy as np
from moviepy.editor import *
import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import threading
from queue import Queue
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SubjectTracker:
    def __init__(self, confidence_threshold=0.7, tracker_type="CSRT"):
        """Initialize the subject tracker with MediaPipe face detection and OpenCV tracking."""
        self.confidence_threshold = confidence_threshold
        self.tracker_type = tracker_type
        
        # Initialize MediaPipe face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=confidence_threshold
        )
        
        self.tracker = None
        self.tracking_failed = False
    
    def create_tracker(self):
        """Create an OpenCV tracker based on the specified type."""
        if self.tracker_type == "CSRT":
            return cv2.TrackerCSRT_create()
        elif self.tracker_type == "KCF":
            return cv2.TrackerKCF_create()
        else:
            raise ValueError(f"Unsupported tracker type: {self.tracker_type}")
    
    def detect_face(self, frame):
        """Detect face using MediaPipe."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        if not results.detections:
            return None
        
        # Get the first detected face
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        
        # Convert relative coordinates to absolute
        h, w, _ = frame.shape
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        
        # Expand the box to include more of the upper body
        y = max(0, y - height//2)
        height = int(height * 1.5)
        
        return (x, y, width, height)
    
    def initialize_tracking(self, frame):
        """Initialize tracking with face detection."""
        bbox = self.detect_face(frame)
        if bbox is None:
            return False
        
        self.tracker = self.create_tracker()
        return self.tracker.init(frame, bbox)
    
    def update_tracking(self, frame):
        """Update tracking and return the new bounding box."""
        if self.tracker is None or self.tracking_failed:
            success = self.initialize_tracking(frame)
            if not success:
                return None
            self.tracking_failed = False
        
        success, bbox = self.tracker.update(frame)
        
        if not success:
            self.tracking_failed = True
            return self.detect_face(frame)
        
        return bbox

class VideoReframer:
    def __init__(self, config):
        """Initialize the video reframer with configuration."""
        self.config = config
        self.tracker = SubjectTracker(
            confidence_threshold=config.get('detection_confidence', 0.7),
            tracker_type=config.get('tracking_algorithm', 'CSRT')
        )
        self.zoom_factor = config.get('zoom_factor', 1.1)
        self.padding_color = config.get('padding_color', [0, 0, 0])
        self.padding_blur = config.get('padding_blur', 5)
    
    def calculate_crop_window(self, frame_width, frame_height, bbox):
        """Calculate the crop window to maintain 9:16 aspect ratio."""
        if bbox is None:
            return None
        
        x, y, w, h = [int(v) for v in bbox]
        center_x = x + w//2
        center_y = y + h//2
        
        # Calculate target width for 9:16 aspect ratio
        target_width = int(frame_height * 9/16)
        
        # Apply zoom factor
        target_width = int(target_width / self.zoom_factor)
        
        # Calculate crop coordinates
        crop_x = max(0, min(center_x - target_width//2, frame_width - target_width))
        
        return (crop_x, 0, target_width, frame_height)
    
    def apply_padding(self, frame, target_height):
        """Apply padding to the frame if needed."""
        h, w = frame.shape[:2]
        if h >= target_height:
            return frame
        
        pad_top = (target_height - h) // 2
        pad_bottom = target_height - h - pad_top
        
        if self.padding_blur > 0:
            # Create blurred padding
            padded = cv2.copyMakeBorder(
                frame, pad_top, pad_bottom, 0, 0,
                cv2.BORDER_REPLICATE
            )
            blurred = cv2.GaussianBlur(padded, (self.padding_blur*2+1, self.padding_blur*2+1), 0)
            return blurred
        else:
            # Use solid color padding
            return cv2.copyMakeBorder(
                frame, pad_top, pad_bottom, 0, 0,
                cv2.BORDER_CONSTANT,
                value=self.padding_color
            )
    
    def process_frame(self, frame):
        """Process a single frame."""
        bbox = self.tracker.update_tracking(frame)
        if bbox is None:
            return frame
        
        crop_window = self.calculate_crop_window(frame.shape[1], frame.shape[0], bbox)
        if crop_window is None:
            return frame
        
        x, y, w, h = crop_window
        cropped = frame[:, x:x+w]
        return self.apply_padding(cropped, frame.shape[0])
    
    def process_video(self, input_path, output_path):
        """Process the entire video."""
        try:
            # Load video and get properties
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError("Could not open input video")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate output dimensions (9:16 aspect ratio)
            output_width = int(height * 9/16)
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (output_width, height)
            )
            
            # Process frames
            pbar = tqdm(total=frame_count, desc="Processing frames")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame = self.process_frame(frame)
                out.write(processed_frame)
                pbar.update(1)
            
            pbar.close()
            cap.release()
            out.release()
            
            # Copy audio from input to output
            self._copy_audio(input_path, output_path)
            
            logger.info(f"Video processing completed: {output_path}")
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise
    
    def _copy_audio(self, input_path, output_path):
        """Copy audio from input video to output video."""
        try:
            # Load the input video with audio
            video = VideoFileClip(input_path)
            
            if video.audio is not None:
                # Load the output video and add audio
                output_video = VideoFileClip(output_path)
                final_video = output_video.set_audio(video.audio)
                
                # Write the final video with audio
                temp_output = output_path + ".temp.mp4"
                final_video.write_videofile(
                    temp_output,
                    codec='libx264',
                    audio_codec='aac'
                )
                
                # Clean up
                video.close()
                output_video.close()
                final_video.close()
                
                # Replace the original output with the version with audio
                import os
                os.replace(temp_output, output_path)
            
        except Exception as e:
            logger.error(f"Error copying audio: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Reframe landscape videos to portrait format")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--config", default=None, help="Configuration file path")
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize and run video reframer
    reframer = VideoReframer(config)
    reframer.process_video(args.input, args.output)

if __name__ == "__main__":
    main()
