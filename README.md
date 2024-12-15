# Video Reframer

A robust system for automatically reframing landscape videos into portrait (9:16) format while dynamically tracking and centering the main subject.

## Features

- Automatic face detection and tracking using MediaPipe and OpenCV
- Dynamic 9:16 aspect ratio reframing
- Smooth subject tracking with fallback detection
- Configurable parameters for detection, tracking, and visual settings
- Audio preservation in output videos
- Progress tracking and error logging
- Support for various video formats

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Basic usage:
```bash
python reframe_video.py --input input.mp4 --output output.mp4
```

With configuration file:
```bash
python reframe_video.py --input input.mp4 --output output.mp4 --config config.json
```

## Configuration

The system can be configured using a JSON configuration file. Example configuration:

```json
{
    "detection_confidence": 0.7,
    "tracking_algorithm": "CSRT",
    "zoom_factor": 1.1,
    "padding_color": [0, 0, 0],
    "padding_blur": 5
}
```

### Configuration Options

- `detection_confidence` (float): Confidence threshold for face detection (0.0-1.0)
- `tracking_algorithm` (string): OpenCV tracking algorithm ("CSRT" or "KCF")
- `zoom_factor` (float): Zoom factor for the subject (> 1.0 zooms in)
- `padding_color` (list): RGB values for padding color [R, G, B]
- `padding_blur` (int): Blur radius for padding (0 for solid color)

## Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- NumPy
- MoviePy
- tqdm

## Error Handling

The system includes comprehensive error handling and logging:
- Input video validation
- Face detection and tracking status
- Processing progress updates
- Audio copying status

Logs are output to the console with timestamps and severity levels.

## Performance

The system is optimized for:
- Efficient frame processing
- Smooth tracking transitions
- Memory management for large videos
- Audio preservation

## License

MIT License
