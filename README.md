## About

A simple shape tracker written in python as an interview assignment. Currently only two shapes (rectangles and circles) are being tracked and their trajectories plotted in a separate window.

## Requirements
- Python3
- Poetry

## Usage
```
poetry install
poetry run python app/main.py
```

Once the video ends, pressing any key confirms the termination of program.

Trajectories will be saved in `app/results` folder in `.png` format.

### Using custom vido samples
Simply put your video to the `app/samples` folder and run the script as:
```
poetry run python app/main.py -s your_sample_video.mp4
```

If circles are not being detected, modify parameters of HoughCircles function in `tracker.py` file.

Note: There is a check in the code for `.mp4` format. Disable it if you want to use different video formats.
