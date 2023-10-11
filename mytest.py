import argparse
from pathlib import Path
import sys
import os
import cv2
import img_operation as imo
import os_operation as oso
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
import numpy as np

path='/home/liyuxuan/vscode/pywork_linux/others/yolov5/fordetect/pencil.jpg'
data_path='/home/liyuxuan/vscode/pywork_linux/res/face2/labels/val'

print(imo.diagonal_to_reccont([10,20],[30,40]))