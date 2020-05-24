from __future__ import absolute_import, division, print_function
import os
import hashlib
import zipfile
from six.moves import urllib

def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d

import os
def get_files(mydir):
    res = []
    for root, dirs, files in os.walk(mydir, followlinks=True):
        for f in files:
            if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg") or f.endswith(".JPG"):
                res.append(os.path.join(root, f))
    return res


