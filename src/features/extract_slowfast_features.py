import os
import random
from io import BytesIO
import torch
import torch.utils.data

import slowfast.datasets.decoder as decoder
import slowfast.datasets.transform as transform
import slowfast.datasets.video_container as container
import slowfast.utils.logging as logging
import slowfast.datasets.kinetics as kinetics
import slowfast.utils.checkpoint as cu
from tqdm import tqdm
import math

from slowfast.models import model_builder
from slowfast.models.video_model_builder import SlowFastModel
import numpy as np
from slowfast.config.defaults import get_cfg

import h5py

logger = logging.get_logger(__name__)

class SlowFastExtractor(SlowFastModel):
    def forward(self, x):
        x = self.s1(x)
        x = self.s1_fuse(x)
        x = self.s2(x)
        x = self.s2_fuse(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s3_fuse(x)
        x = self.s4(x)
        x = self.s4_fuse(x)
        x = self.s5(x)
        x = [self.head.pathway0_avgpool(x[0]), self.head.pathway1_avgpool(x[1])]
        x = torch.cat(x, 1)
        x = x.permute(0,2,3,4,1)
        x = x.mean([1,2,3])
        return x

class Charade(kinetics.Kinetics):
    """
    Kinetics video loader. Construct the Kinetics video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=10, san_check=False):
        """
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        self.cfg = cfg
        self.mode = mode
        self._num_retries = num_retries
        self._construct_loader(san_check)
        
    
    def _construct_loader(self, san_check):
        """
        Construct the video loader.
        """
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(self.mode)
        )
        assert os.path.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )
        
        # configurations
        target_fps = 30
        sampling_rate = self.cfg.DATA.SAMPLING_RATE
        num_frames = self.cfg.DATA.NUM_FRAMES
        
        videos = []
        durations = []
        with open(path_to_file) as f:
            for line in f:
                vid_id, dur = line.rstrip().split(', ')
                videos.append(vid_id)
                durations.append(float(dur))

        self._load_indices = []
        
        clip_duration = 2 * 32 / 30
        
        for video, dur in tqdm(zip(videos, durations), total=len(videos)):
            video_path = f"{self.cfg.DATA.PATH_PREFIX}/{video}.mp4"
            video_container = container.get_video_container(
                    video_path
            )
            fps = float(video_container.streams.video[0].average_rate)
            total_frames = video_container.streams.video[0].frames
            num_clips = int(total_frames / fps / clip_duration + 1)

            for ci in range(num_clips + 1):
                self._load_indices.append((video_path, num_clips, ci))
            
            if san_check:
                if len(self._load_indices) > 100:
                    break
                
        logger.info(
            "Constructing kinetics dataloader (size: {}) from {}".format(
                len(self._load_indices), path_to_file
            )
        )

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        video, num_clips, ci = self._load_indices[index]
        min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
        
        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for _ in range(self._num_retries):
            video_container = None
            try:
                video_container = container.get_video_container(
                    video,
                    multi_thread_decode=True,
                )
            except Exception as e:
                logger.info(
                    "Failed to load video from {} with error {}".format(
                        video, e
                    )
                )
            
            if video_container is None:
                raise RuntimeError(f'could not construct video_container: {video}')

            # Decode video. Meta info is used to perform selective decoding.
            frames = decoder.decode(
                video_container,
                self.cfg.DATA.SAMPLING_RATE,
                self.cfg.DATA.NUM_FRAMES,
                ci,
                num_clips,
                video_meta=None,
                target_fps=30,
            )

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames is None:
                raise RuntimeError('output frames are empty')

            # Perform color normalization.
            frames = frames.float()
            frames = frames / 255.0
            frames = frames - torch.tensor(self.cfg.DATA.MEAN)
            frames = frames / torch.tensor(self.cfg.DATA.STD)
            
            # T H W C -> C T H W.
            frames = frames.permute(3, 0, 1, 2)
            # Perform data augmentation.
            frames = self.spatial_sampling(
                frames,
                spatial_idx=1,
                min_scale=min_scale,
                max_scale=max_scale,
                crop_size=crop_size,
            )

            frames = self.pack_pathway_output(frames)
            return frames, video, num_clips, ci
        else:
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(
                    self._num_retries
                )
            )

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._load_indices)
    
    def spatial_sampling(
        self,
        frames,
        spatial_idx=-1,
        min_scale=256,
        max_scale=320,
        crop_size=224,
    ):
        """
        Perform spatial sampling on the given video frames. If spatial_idx is
        -1, perform random scale, random crop, and random flip on the given
        frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
        with the given spatial_idx.
        Args:
            frames (tensor): frames of images sampled from the video. The
                dimension is `num frames` x `height` x `width` x `channel`.
            spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
                or 2, perform left, center, right crop if width is larger than
                height, and perform top, center, buttom crop if height is larger
                than width.
            min_scale (int): the minimal size of scaling.
            max_scale (int): the maximal size of scaling.
            crop_size (int): the size of height and width used to crop the
                frames.
        Returns:
            frames (tensor): spatially sampled frames.
        """
        assert spatial_idx in [-1, 0, 1, 2]
        if spatial_idx == -1:
            frames = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames = transform.random_crop(frames, crop_size)
            frames = transform.horizontal_flip(0.5, frames)
        else:
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
#             frames = transform.random_short_side_scale_jitter(
#                 frames, min_scale, max_scale
#             )
#             print(frames.shape)
            frames = short_side_scale_padding(frames, crop_size)
        return frames
    
def short_side_scale_padding(images, size):
    height = images.shape[2]
    width = images.shape[3]
    if (width <= height and width == size) or (
        height <= width and height == size
    ):
        return images
    new_width = size
    new_height = size
    if width < height:
        new_width = int(math.floor((float(width) / height) * size))
        pad_w = (size - new_width) // 2
        pad = (pad_w, size-pad_w-new_width, 0, 0)
    else:
        new_height = int(math.floor((float(height) / width) * size))
        pad_h = (size - new_height) // 2
        pad = (0, 0, pad_h, size-pad_h-new_height)

    images = torch.nn.functional.interpolate(
        images,
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False,
    )
    return torch.nn.functional.pad(
        images,
        pad=pad,
    )

def cvrt_npy2h5(cfg, split):
    feat = np.load(f'data/interim/slowfast/{split}.npy')
    
    dataset = Charade(cfg, split)
    cur_video_id = None
    N = len(dataset)
    
    with h5py.File(f"data/processed/slowfast/{split}.h5", "w") as hf:    
        for i in tqdm(range(N)):
            video_path, num_clip, ci = dataset._load_indices[i]
            video_id = os.path.basename(video_path).split('.')[0]

            if video_id != cur_video_id:
                chunk = feat[i:i+num_clip+1]
                hf.create_dataset(video_id, data=chunk)
                cur_video_id = video_id

def extract_feat(cfg, split):
    """
    Args:
        cfg (CfgNode): configs.
    """
    device = torch.device("cuda:0")
    model = SlowFastExtractor(cfg)
    
    cu.load_checkpoint(
        cfg.TEST.CHECKPOINT_FILE_PATH,
        model,
        cfg.NUM_GPUS > 1,
        None,
        inflation=False,
        convert_from_caffe2=cfg.TEST.CHECKPOINT_TYPE == "caffe2",
    )
        
    model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()

    dataset = Charade(cfg, split)
#     sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=12,
        shuffle=False,
        sampler=None,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=False,
    )
    
    feat = np.zeros((len(dataset), 2304))
    i = 0
    
    with torch.no_grad():
        for inputs, video, _, _ in tqdm(loader):
            y = model(inputs)
            y = y.cpu().numpy()
            feat[i:i+len(y)] = y
            i += len(y)

    return feat

def main():
    cfg = get_cfg()
    cfg.merge_from_file('data/external/slowfast_cfg/slowfast_8x8_r50.yml')
    
    for split in ['test', 'train']:
        feat = extract_feat(cfg, split)
        np.save(f'data/interim/slowfast/{split}.npy', feat)

        cvrt_npy2h5(cfg, split)
    
if __name__ == "__main__":
    main()