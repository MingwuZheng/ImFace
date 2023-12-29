import os, sys
root_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(root_dir)
from data_preprocess.preprocess import data_preprocess

data_preprocess('demo/facescape_demo.obj', 'demo/demo.npy')