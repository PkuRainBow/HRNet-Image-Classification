import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.distributed as dist

import numpy as np
from PIL import Image, ImageFilter
import time
import csv
import sys
import os
import base64
import cv2
import math
import os.path as op
import random
import io

csv.field_size_limit(sys.maxsize)

'''
def img_from_base64(imagestring, color=True):
    jpgbytestring = base64.b64decode(imagestring)
    nparr = np.frombuffer(jpgbytestring, np.uint8)
    try:
        if color:
            r = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return r
        else:
            r = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            return r
    except:
        return None
'''

def img_from_base64(imagestring, color=True):
    img_str = base64.b64decode(imagestring)
    try:
        if color:
            r = Image.open(io.BytesIO(img_str)).convert('RGB')
            return r
        else:
            r = Image.open(io.BytesIO(img_str)).convert('L')
            return r
    except:
        return None

def generate_lineidx(filein, idxout):
    assert not os.path.isfile(idxout)
    with open(filein, 'r') as tsvin, open(idxout, 'w') as tsvout:
        fsize = os.fstat(tsvin.fileno()).st_size
        fpos = 0
        while fpos != fsize:
            tsvout.write(str(fpos)+"\n")
            tsvin.readline()
            fpos = tsvin.tell()

class TSVFile(object):
    def __init__(self, tsv_file):
        self.tsv_file = tsv_file
        self.lineidx = op.splitext(tsv_file)[0] + '.lineidx'
        self._fp = None
        self._lineidx = None

    def num_rows(self):
        self._ensure_lineidx_loaded()
        return len(self._lineidx)

    def seek(self, idx):
        self._ensure_tsv_opened()
        self._ensure_lineidx_loaded()
        pos = self._lineidx[idx]
        self._fp.seek(pos)
        return [s.strip() for s in self._fp.readline().split('\t')]

    def seek_list(self, idxs, q):
        assert isinstance(idxs, list)
        self._ensure_tsv_opened()
        self._ensure_lineidx_loaded()
        for idx in idxs:
            pos = self._lineidx[idx]
            self._fp.seek(pos)
            q.put([s.strip() for s in self._fp.readline().split('\t')])

    def close(self):
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def _ensure_lineidx_loaded(self):
        # if not op.isfile(self.lineidx) and not op.islink(self.lineidx):
        #     generate_lineidx(self.tsv_file, self.lineidx)
        if self._lineidx is None:
            with open(self.lineidx, 'r') as fp:
                self._lineidx = [int(i.strip()) for i in fp.readlines()]

    def _ensure_tsv_opened(self):
        if self._fp is None:
            self._fp = open(self.tsv_file, 'r')

class TSVInstance(Dataset):
    def __init__(self, tsv_file, transform=None):
        self.tsv = TSVFile(tsv_file + '.tsv')
        self.transform = transform

    def __getitem__(self, index):
        row = self.tsv.seek(index)
        img = img_from_base64(row[-1])
        
        if self.transform is not None:
            trans_img = self.transform(img)
        idx = int(row[1])
        label = torch.from_numpy(np.array(idx, dtype=np.int))

        return trans_img, label

    def __len__(self):
        return self.tsv.num_rows()

