# Hackathon2020

- `pip install -r requierements.txt`
- restart after it
- `gcc --version`
for google colab:
- !pip install youtube-dl
- !pip uninstall -y opencv-python-headless opencv-contrib-python
- !apt install python3-opencv  # the one pre-installed have some issues
- !youtube-dl https://www.youtube.com/watch?v=ll8TgCZ0plk -f 22 -o video.mp4
- !ffmpeg -i video.mp4 -ss 00:16:35 -t 00:00:15 -c:v copy video-clip.mp4

!git clone https://github.com/facebookresearch/detectron2
!python demo.py --config-file detectron2/configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml --video-input video-clip.mp4 --confidence-threshold 0.6 --output video-output.mkv \
  --opts MODEL.WEIGHTS detectron2://COCO-PanopticSegmentation/panoptic_fpn_R_101_3x/139514519/model_final_cafdb1.pkl