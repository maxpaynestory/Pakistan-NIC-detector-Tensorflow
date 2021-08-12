# Pakistan NIC detector Tensorflow

## Setup

- Install python 3.6 and pip
- Download trained model file
  https://drive.google.com/uc?id=12BD7NyDZHxDVC5H8nHDH_Qw0SXDnbw7e&export=download and extract on root
- Download tensorflow models from https://github.com/tensorflow/models/archive/refs/tags/v1.13.0.zip
- Install tensorflow models by going inside folder research and run command

```bash
python setup.py install
```

- Install pip dependencies by running

```bash
pip install -r requirements.txt
```

- Open script detectusingwebcam.py and adjust your PATH_TO_CKPT according to extracted zip file

## Run

On repository root run

```
python detectusingwebcam.py
```

You should be able to see a window showing your camera feed. If you show a Pakistani NIC in front of camera, the video will show you a bounding box around detected photo.

![Screenshot of running app](/screenshot.png "Screenshot of running app")
