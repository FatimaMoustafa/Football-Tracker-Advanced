# Football Tracker - Quick Reference Cheat Sheet

## Installation (First Time)
```bash
# 1. Install FFmpeg
# Windows: choco install ffmpeg
# Mac: brew install ffmpeg  
# Linux: sudo apt-get install ffmpeg

# 2. Verify
ffmpeg -version

# 3. Install Python packages
pip install -r requirements_tracking_support.txt
```

## Most Common Commands

### Run Everything (Recommended)
```bash
python run_complete_demo.py --full --input input_videos/video.mp4
# Output: demo_output/evaluation/evaluation_report.html
```

### Just Process Video (Memory Safe)
```bash
python run_chunked_pipeline.py --input video.mp4 --output output_videos
# Handles 45+ minutes without crash
```

### Just Evaluate Quality
```bash
python run_evaluation.py --input video.mp4 --output eval_results
# Generate metrics + HTML report
```

### Just Calibrate Homography
```bash
python run_homography_picker.py --input frame.jpg --output config/homography_matrix.json
# Click 4 corners, save matrix
```

## Output Files

| Command | Output File | Usage |
|---------|------------|-------|
| Chunked Pipeline | `output_videos/output_video_chunked.mp4` | Annotated video |
| Evaluation | `evaluation_results/evaluation_report.html` | Open in browser ⭐ |
| Evaluation | `evaluation_results/evaluation_metrics.json` | Raw data |
| Homography | `config/homography_matrix.json` | Reuse for same camera |

## Key Metrics (in HTML Report)

| Metric | Good | Bad | What to Do |
|--------|------|-----|-----------|
| ID Switches | < 10 | > 50 | Retune tracking |
| Fragmentation | < 0.3 | > 0.7 | Adjust thresholds |
| Speed Violations | < 5 | > 20 | Check camera/occlusion |
| Consistency | > 0.7 | < 0.5 | Check lighting/angle |
| **Overall Rating** | **A-B** | **C-F** | **Evaluate report** |

## Troubleshooting Quick Fixes

| Problem | Fix |
|---------|-----|
| FFmpeg not found | `choco install ffmpeg` then restart terminal |
| Out of Memory | Use `run_chunked_pipeline.py` (that's what it's for!) |
| Homography picker won't open | `pip install opencv-python --upgrade` |
| Model not found | Copy model to `models/best.pt` or use `--model path` |
| Video too slow | Use GPU: ensure CUDA is installed |

## File Structure

```
✓ Video Chunking:     pipeline/video_chunking.py
✓ Evaluation:         evaluation/eval_harness.py  
✓ Homography:         tools/homography_picker.py
✓ Entry Points:       run_*.py scripts
✓ Config:            config/pipeline_config.json
✓ Docs:              SETUP_AND_USAGE.md
```

## Memory Usage

```
Before (Original):
  10min:  ~200MB
  45min:  ~900MB → CRASH ✗

After (Chunking):
  10min:  ~150MB
  45min:  ~150MB ✓ NO CRASH
```

## Performance

```
10 minutes:   2-3 min (GPU) / 8-10 min (CPU)
45 minutes:  12-15 min (GPU) / 45-60 min (CPU)
Memory:      Stable at ~150MB
```

## Python API Quick Start

```python
# Process video in chunks
from run_chunked_pipeline import ChunkedPipeline
pipeline = ChunkedPipeline()
stats = pipeline.process_video_chunked('video.mp4', 'output')

# Evaluate quality
from run_evaluation import EvaluationHarness
harness = EvaluationHarness()
metrics = harness.evaluate_video('video.mp4', 'eval_results')

# Calibrate homography
from tools.homography_picker import interactive_homography_picker
data = interactive_homography_picker('frame.jpg', 'homography.json')
```

## Workflow Summary

```
1. Install FFmpeg ← CRITICAL
2. Run: python run_complete_demo.py --full --input video.mp4
3. Wait for processing (12-15 min for 45min video)
4. Open: demo_output/evaluation/evaluation_report.html
5. Review metrics
6. ✓ Done!
```

## Getting Help

1. **Setup Issues**: Read `SETUP_AND_USAGE.md`
2. **Configuration**: Check `CONFIG_GUIDE.md`
3. **Code Issues**: Enable debug: `export DEBUG=1`
4. **Evaluation Report**: See HTML report for detailed metrics

---

**TL;DR**: 
```bash
pip install -r requirements_tracking_support.txt
python run_complete_demo.py --full --input video.mp4
# Open: demo_output/evaluation/evaluation_report.html
```
