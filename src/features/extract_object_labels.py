import chainercv
import chainer
import numpy as np
import os
import glob
from chainer.datasets import ImageDataset
from chainer.iterators import SerialIterator
from chainer.iterators import MultiprocessIterator
import orjson
import argparse
# from tqdm import tqdm
import pandas as pd
import time

from chainercv.links import SSD512
from chainercv.datasets import voc_bbox_label_names

FRAME_DIR = '/home/otani_mayu/3TDisk/Data/Charades/Charades_v1_rgb/'
EXPORT_DIR = './data/interim/object_detection/charade/'
chainer.config.cv_resize_backend = "cv2"

def cvrt_json(bboxes, labels, scores):
    res = []
    # loop over frames
    for bbox, label, score in zip(bboxes, labels, scores):
        frame = []
        # loop over objects
        for b, l, s in zip(bbox, label, score):
            obj = {
                'label': int(l),
                'bbox':b.tolist(),
                'score':float(s)
            }
            frame.append(obj)
        res.append(frame)
    
    json_str = str(orjson.dumps(res), "utf-8")
    
    return json_str

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--device', '-d', type=str, default='-1',
                        help='Device specifier. Either ChainerX device '
                        'specifier or an integer. If non-negative integer, '
                        'CuPy arrays with specified device id are used. If '
                        'negative integer, NumPy arrays are used')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--split', type=str, default='train')
    
    args = parser.parse_args()
    device = chainer.get_device(args.device)
    device = chainer.get_device(device)

    model = SSD512(pretrained_model='voc0712')
    model.to_device(device)
    device.use()
    
    df = pd.read_csv('data/raw/Charades_v1_{}.csv'.format(args.split))
    videos = df.id.unique().tolist()
#     videos = os.listdir(FRAME_DIR)
    videos.sort()
    
    videos = videos[args.start:args.end]
    
    for v_id in videos:
        f_name = EXPORT_DIR + v_id + '.json'
        
        if os.path.exists(f_name):
            continue
            
        start = time.time()
        
        frames = os.listdir(FRAME_DIR+v_id)
        frames.sort()
        frames = frames[::6] # 4fps: 0-frame, 6-frame, 12, 18, ...
        dataset = ImageDataset(frames, root=FRAME_DIR+v_id)
        itr = SerialIterator(dataset,
                                   48,
                                   shuffle=False,
                                   repeat=False,
                                  )
        v_bboxes = []
        v_labels = []
        v_scores = []
        
        for batch in itr:
            bboxes, labels, scores = model.predict(batch)
            v_bboxes += bboxes
            v_labels += labels
            v_scores += scores

        json_str = cvrt_json(v_bboxes, v_labels, v_scores)
        
        with open(f_name, 'w') as f:
            f.write(json_str)
        
        print('%s: %.2f (%03i fr)'%(v_id, time.time()-start, len(frames)))