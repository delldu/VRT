import video_former

# video_former.video_zoom_client("TAI", "videos/zoom.mp4", "output/zoom.mp4")
# video_former.video_zoom_server("TAI")

video_former.video_deblur_client("TAI", "videos/deblur.mp4", "output/deblur.mp4")
video_former.video_deblur_server("TAI")

# video_former.video_denoise_client("TAI", "videos/denoise.mp4", 10, "output/denoise.mp4")
# video_former.video_denoise_server("TAI")
