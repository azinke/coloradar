# Coloradar

The `coloradar` package makes it easy to access and render entries of the Coloradar dataset.
From the dataset, only the Lidar, single-chip radar, and cascade radar are covered for now in the
current version of the package.

One of the interesting aspects of the Coloradar dataset is the availability of raw ADC samples of
the radar sensors. Thus, it's possible to apply the desired radar signal processing techniques.

With the current version of the `coloradar` package, the supported features as presented in the table below


| Sensor                | 3D PCL | 3d Heatmap | Bird Eye View | Heatmap (from raw ADC) | PCL (from raw ADC) |
|-----------------------|:------:|:----------:|:-------------:|:----------------------:|:------------------:|
| LiDaR                 |   x    |     -      |       x       |           -            |          -         |
| Single chip radar     |   x    |     x      |       x       |           x            |          x         |
| Cascaded radar        |   -    |     x      |       x       |           x            |          x         |

[x] : Implemented

[-]: N/A or Not Implemented

PCL: Pointcloud

For the radar sensors, the first two columns of the table above indicate post-processed data made available
in the `coloradar` dataset.


## Environment setup

Create a Python virtual environment and install the required packages as follows:

```bash
python -m venv venv
source ./venv/bin/activate

python -m pip install -r requirements.txt
```

## Coloradar dataset

The coloradar dataset can be downloaded from [coloradar dataset](https://arpg.github.io/coloradar/).

The dataset is composed of many subsets that can be downloaded separately. But the folder structure of
each subset is the same.

## Setup and configuration

The structure of this repository is as follows:

```txt
.
├── core
├── dataset
│   └── dataset.json
├── __init__.py
├── main.py
├── README.md
└── requirements.txt
```

Before using this tool, the dataset must be downloaded and unzipped in the `dataset` folder as well as
the calibration information (named `calib` on the coloradar dataset website) of the different sensors.

Then, the folder structure would look like this:

```txt
.
├── core
├── dataset
│   ├── 12_21_2020_ec_hallways_run0
│   ├── 12_21_2020_ec_hallways_run4
│   ├── 2_22_2021_longboard_run1
│   ├── 2_23_2021_edgar_army_run5
│   ├── 2_24_2021_aspen_run0
│   ├── 2_24_2021_aspen_run9
│   ├── 2_28_2021_outdoors_run0
│   ├── calib
│   └── dataset.json
├── __init__.py
├── main.py
├── README.md
└── requirements.txt
```

In the snippet above, one can notice that multiple subsets of the dataset have been downloaded. The key
point to configure the supported subset of the dataset and how to access them is the
`dataset/dataset.json` file.

The only part that requires your attention is the `folders` key of the `datastore` entry.

```json
    "folders": [
      {
        "codename": "hallway0",
        "path": "12_21_2020_ec_hallways_run0"
      },
      {
        "codename": "hallway4",
        "path": "12_21_2020_ec_hallways_run4"
      },
      {
        "codename": "hallway1",
        "path": "12_21_2020_ec_hallways_run1"
      },
      {
        "codename": "outdoor0",
        "path": "2_28_2021_outdoors_run0"
      },
      {
        "codename": "aspen0",
        "path": "2_24_2021_aspen_run0"
      },
      {
        "codename": "aspen9",
        "path": "2_24_2021_aspen_run9"
      },
      {
        "codename": "edgar5",
        "path": "2_23_2021_edgar_army_run5"
      },
      {
        "codename": "longboard1",
        "path": "2_22_2021_longboard_run1"
      }
    ]
```
 In the example shown above, each subset of the dataset to handle, is registered along with a short
 `codename` to access it. The subsets already registered can be accessed with their corresponding
 codename. New subsets of the Coloradar dataset can be added similarly. The codenames can
 even be updated to suit the naming convention that you would prefer.

With that, you're all set to play around with the Coloradar dataset.

## Usage

**IMPORTANT NOTE**:
- _If you've set up a virtual environment, don't forget to enable it first_
- _Renderings are done based on a web-based backend. So, a web browser tab will automatically be launched for all renderings._

The easiest way to have an overview of all the available options to interact with the dataset
is the help command.


```bash
python coloradar.py -h
```

However, find below the cheat sheet of this CLI tool


1. Overview

```bash
# Print the 'dataset.json' configuration file
python coloradar.py -o
python coloradar.py --overview
```

Either one of these commands pretty-prints the entire `dataset/dataset.json` file to allow a quick
overview of the current configuration in use.

Since each subset of the dataset receives a codename to interact with it, you can request
the list of currently registered subsets of the dataset and their codenames as follows:

```bash
# Get the list of registered dataset and their codenames
python coloradar.py -o --codename
python coloradar.py --overview --codename
```

2. Lidar sensor

```bash
# Render lidar 3D pointcloud
python coloradar.py --dataset <codename> -i <frame-index> --lidar

# Render lidar pointcloud bird eye view
python coloradar.py --dataset <codename> -i <frame-index> --lidar -bev
python coloradar.py --dataset <codename> -i <frame-index> --lidar --bird-eye-view
```

See examples below:

```bash
# Render lidar 3D pointcloud
python coloradar.py --dataset hallway0 -i 130 --lidar

# Render lidar pointcloud bird eye view
python coloradar.py --dataset hallway0 -i 175 --lidar -bev
```


3. Radar sensor

The CLI options to access the data from either the single-chip radar sensor or cascaded
radar are quite similar.

The shorthand used is:
- `scradar`: single-chip radar
- `ccradar`: cascaded chip radar

Therefore, we have the following commands


```bash
# Render single chip radar 3D pointcloud
python coloradar.py --dataset <codename> -i <frame-index> --scradar

# Render single chip radar birds' eye view from 3D pointcloud
# Resolution in meter/pixel  | Eg.: 0.1 -> 10cm / pixel
python coloradar.py --dataset <codename> -i <frame-index> --scradar -bev --resolution <resolution>

# Render single chip radar  3D heatmap
python coloradar.py --dataset <codename> -i <frame-index> --scradar --heatmap


#
# Processing of raw ADC samples from single chip radar
#

# Render single chip radar  3D heatmap  from raw ADC
python coloradar.py --dataset <codename> -i <frame-index> --scradar --raw [--no-sidelobe]

# Render single chip radar  2D heatmap  from raw ADC samples
python coloradar.py --dataset <codename> -i <frame-index> --scradar --raw --heatmap-2d

# Render single chip radar  3D pointcloud  from raw ADC samples
python coloradar.py --dataset <codename> -i <frame-index> --scradar --raw -pcl

# Render single chip radar pointcloud bird eye view  from raw ADC samples
python coloradar.py --dataset <codename> -i <frame-index> --scradar --raw -pcl -bev
```

```bash
# Render cascaded chip radar 3D pointcloud
python coloradar.py --dataset <codename> -i <frame-index> --ccradar

# Render cascaded chip radar birds' eye view from 3D pointcloud
# Resolution in meter/pixel  | Eg.: 0.1 -> 10cm / pixel
python coloradar.py --dataset <codename> -i <frame-index> --ccradar -bev --resolution <resolution>

# Render cascaded chip radar  3D heatmap
python coloradar.py --dataset <codename> -i <frame-index> --ccradar --heatmap


#
# Processing of raw ADC samples from cascaded radar sensor
#

# Render cascaded chip radar  3D heatmap  from raw ADC samples
python coloradar.py --dataset <codename> -i <frame-index> --ccradar --raw [--no-sidelobe]

# Render cascaded chip radar  2D heatmap  from raw ADC samples
python coloradar.py --dataset <codename> -i <frame-index> --ccradar --raw --heatmap-2d

# Render cascaded chip radar  3D pointcloud  from raw ADC samples
python coloradar.py --dataset <codename> -i <frame-index> --ccradar --raw -pcl

# Render cascaded chip radar pointcloud bird eye view  from raw ADC samples
python coloradar.py --dataset <codename> -i <frame-index> --ccradar --raw -pcl -bev
```

See examples below:

```bash
# Render single chip radar 3D pointcloud
python coloradar.py --dataset hallway0 -i 130 --scradar --raw

# Render cascaded chip radar pointcloud bird eye view from raw ADC samples
python coloradar.py --dataset hallway0 -i 175 --ccradar --raw -pcl -bev
```

4. Batched processing and save outputs

You can note that the index option `-i` is no longer needed. The path given for
the `save-to` option could be a non-existing one. The path will automatically be
created in that case.

```bash
# Render and save all lidar bird eye view of a given subset of the dataset
python coloradar.py --dataset <codename> --lidar -bev --save-to <output-directory>

# Render and save all single chip radar plointcloud bird eye view of a given subset of the dataset
python coloradar.py --dataset <codename> --scradar --raw -pcl -bev --save-to <output-directory>

# Render and save all cascaded chip radar plointcloud bird eye view of a given subset of the dataset
python coloradar.py --dataset <codename> --ccradar --raw -pcl -bev --save-to <output-directory>
```

5. Save post-processed files as `.csv` or `.bin` files

The placeholder `<ext>` could be either `csv` or `bin`. Binary files are saved as float32 values.

- `csv`: Comma Separated Value
- `bin`: Binary

```bash
# Save all cascaded chip radar plointcloud of a given subset of the dataset as "csv" or "bin" files
python coloradar.py --dataset <codename> --ccradar --raw -pcl --save-as <ext> --save-to <output-directory>

# Example for saving post-processed pointcloud as bin files
python coloradar.py --dataset outdoor0 --ccradar --raw -pcl --save-as bin --save-to output
```

The binary files generated can be read as follows:

```python
import numpy as np

# [0]: Azimuth
# [1]: Range
# [2]: Elevation
# [3]: Velocity
# [4]: Intensity of reflection in dB or SNR
data = np.fromfile(fileptah, dtype=np.float32, count=-1).reshape(-1, 5)
```

6. Animation

```bash
# Create a video out of the images present in the input folder provided
python coloradar.py --dataset <codename> --animate <path-to-image-folder>
```

The generated video is saved in the same folder as the images.
