---
title: VisionQuery
emoji: 🛸
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: agpl-3.0
short_description: Ask questions about drone photos; answers are counted from detector boxes
---

# VisionQuery

Ask a question about a drone photo ("how many cars are in the left half?") and get an answer counted from detector boxes you can
see, with a calibrated range.

Source code, paper details and local setup: see the GitHub repository linked from this Space.

Runs on the free CPU tier. The default model (YOLO26s) takes about 0.26 s per new photo on a laptop CPU (Intel Core i5-12450H);
the shared Space CPU is slower. The sliced YOLO26m-Tile and the Ensemble take tens of seconds per new photo on CPU. Follow-up
questions about the same photo are answered from a cache.
