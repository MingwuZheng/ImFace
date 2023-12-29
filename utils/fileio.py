import os, cv2, plyfile
import json


def write_dict(path, dict):
    assert os.path.exists(os.path.dirname(path)), '{} not exist!'.format(path)
    with open(path, 'w') as f:
        f.write(json.dumps(dict))


def read_dict(path, key2int=True):
    assert os.path.exists(path), '{} not exist!'.format(path)
    with open(path, 'r') as f:
        res = json.load(f)
    if key2int:
        res_int = {}
        for k in res.keys():
            res_int[int(k)] = res[k]
        return res_int
    return res
