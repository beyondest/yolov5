import argparse
from pathlib import Path
import sys
import os
import cv2
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


path='/home/liyuxuan/vscode/pywork_linux/others/yolov5/fordetect/pencil.jpg'
img=cv2.imread(path)

s=img.shape[2:]
print(s)
print(img.shape)
