## About

A simple shape tracker written in python as an interview assignment. Currently only two shapes (circles and squares) are being tracked and their trajectories plotted in a separate window.

## Requirements
- Python3
- Poetry

## Usage
```
poetry install
poetry run python main.py
```

### Using custom vido samples
Simply put your video to the `samples` folder and run the script as:
```
poetry run python main.py -s your_sample_video.mp4
```

Note: There is a check in the code for `.mp4` format. Disable it if you want to use different video formats.
