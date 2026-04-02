# Real-Time Crowd Analysis System using YOLOv8

A real-time computer vision project for detecting, tracking, and analyzing people in video streams using **YOLOv8**, **OpenCV**, and **Python**.

## Overview

This project performs:

- Real-time **person detection** from video
- Simple **ID tracking** across frames using centroid distance
- Calculation of:
  - person presence duration
  - entry and exit times
  - active people count over time
  - peak crowd periods
  - empty time intervals
- Export of analytics results to:
  - `summary.csv`
  - `crowd_time.csv`
  - `peak_period.txt`
- Visualization of crowd density over time using **Matplotlib**

## Features

- Detects only people using **YOLOv8**
- Assigns IDs to detected people
- Tracks movement frame by frame
- Measures how long each person stays in the scene
- Finds the busiest time period
- Detects empty periods with no people
- Displays live analytics on video frames
- Saves structured reports for further analysis

## Technologies Used

- Python
- OpenCV
- Ultralytics YOLOv8
- Matplotlib
- CSV
- Math

## Project Workflow

1. Load input video
2. Run YOLOv8 person detection on each frame
3. Match detections with previous tracked positions using centroid distance
4. Update IDs and timestamps
5. Compute crowd statistics over time
6. Save output files
7. Plot crowd density graph

## Output Files

### `summary.csv`
Contains:
- Person ID
- Start time
- End time
- Duration in seconds

### `crowd_time.csv`
Contains:
- Time in seconds
- Number of active people

### `peak_period.txt`
Contains:
- Peak period index
- Peak average crowd count

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

Install dependencies:

```bash
pip install opencv-python ultralytics matplotlib
```

## Usage

1. Put your video file path inside the code:

```python
video_path = r"C:\Users\yourname\path\to\video.mp4"
```

2. Run the script:

```bash
python main.py
```

3. Press `q` to stop the video window.

## Example Use Cases

- Crowd monitoring
- Store or mall traffic analysis
- Public area analytics
- Entrance/exit observation
- Basic smart surveillance systems

## Limitations

- Tracking is based on simple centroid distance, so it may fail in crowded or complex scenes
- No re-identification after long occlusion
- Not optimized yet for production deployment

## Future Improvements

- Integrate **DeepSORT** or **ByteTrack**
- Add support for live camera streams
- Build a web dashboard for analytics
- Export more advanced statistics
- Improve tracking robustness in dense crowds

## Author

**Yousef Reda Hassan**

If you like this project, feel free to star the repository ⭐
