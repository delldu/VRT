import video_former

# video_former.video_zoom_predict("videos/zoom.mp4", "output/zoom.mp4")

# video_former.video_deblur_predict("videos/deblur.mp4", "output/deblur.mp4")

video_former.video_denoise_predict("videos/denoise.mp4", 10, "output/denoise.mp4")
