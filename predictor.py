import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer

sky = cv2.imread("/content/sky.jpg")
ocean = cv2.VideoCapture("/content/oce.mp4")

import numpy as np
from PIL import Image


def add_channel(image):
    try:
        b_channel, g_channel, r_channel = cv2.split(image)
    except:
        return image

    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255  # creating a dummy alpha channel image.

    return cv2.merge((b_channel, g_channel, r_channel, alpha_channel))


def mask(segmentation, segments_info, target_image, effect_image, category_id, opacity):
    category_id = next(filter(lambda x: x['category_id'] == category_id, segments_info))['id']
    effect_h, effect_w, _ = effect_image.shape
    print(effect_w, effect_h)
    target_h, target_w, _ = target_image.shape
    crop_effect = effect_image[0:target_h, 0:target_w]
    crop_effect = add_channel(crop_effect)
    prediction = segmentation.cpu().numpy()
    target_prediction = np.array([
        [prediction[j][i] == category_id for i in range(len(prediction[j]))]
        for j in range(len(prediction))
    ])
    crop_effect[~target_prediction, :] = [0, 0, 0, 0]
    target_image = add_channel(target_image)
    dst = cv2.addWeighted(crop_effect, opacity, target_image, 1, 0)
    return dst


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

    
    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            print(frame, type(frame))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if "panoptic_seg" in predictions:
                # panoptic_seg, segments_info = predictions["panoptic_seg"]
                # vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                #     frame, panoptic_seg.to(self.cpu_device), segments_info
                # )
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                success, ocean_frame = ocean.read()
                res = mask(panoptic_seg, segments_info, frame, ocean_frame, 21, 0.8)
                res = mask(panoptic_seg, segments_info, res, sky, 40, 0.8)
                # res = mask(panoptic_seg, segments_info, res, graffiti, 50, 0.5)
                img = cv2.cvtColor(res, cv2.COLOR_BGRA2BGR)
                return np.array(img)
            # Converts Matplotlib RGB format to OpenCV BGR format
        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor(frame))

