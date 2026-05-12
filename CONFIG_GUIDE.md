# Configuration Examples for Tracking Support System

## Default Configuration
The default configuration uses:
- Chunk Duration: 60 seconds
- Court Width: 68 meters (football field)
- Court Length: 23.32 meters
- Model: YOLOv8 (best.pt)
- Device: CUDA GPU (auto-detects)

## Customization Guide

### 1. Adjust Chunk Duration
**File**: `config/pipeline_config.json`

```json
{
  "video_processing": {
    "chunk_duration_seconds": 120  // Increase for longer chunks (trade memory for speed)
  }
}
```

**Guidance**:
- 60s: Default, good balance
- 120s: Faster processing, slightly more memory
- 30s: Slower, lower memory usage

### 2. Court Dimensions
For different court sizes:

```json
{
  "homography": {
    "court_width_m": 68,      // Football: 68m, Hockey: 30m
    "court_length_m": 23.32   // Football: 23.32m, Hockey: 61m
  }
}
```

### 3. Evaluation Thresholds
Fine-tune detection parameters:

```json
{
  "evaluation": {
    "id_switch_distance_threshold_px": 50,           // Lower = stricter
    "speed_plausibility_threshold_px_per_frame": 200, // Lower = stricter
    "min_track_length_frames": 5                       // Minimum track duration
  }
}
```

### 4. Device Selection
Force specific device:

```json
{
  "model": {
    "device": "cuda"   // Options: cuda, cpu, mps (Mac)
  }
}
```

## Environment Variables (Optional)

```bash
# Override model path
export TRACKER_MODEL=models/custom_model.pt

# Override device
export TRACKER_DEVICE=cuda

# Override chunk duration (seconds)
export CHUNK_DURATION=60

# Enable debug logging
export DEBUG=1
```

## Performance Tuning

### For Maximum Speed (Large GPU)
```json
{
  "video_processing": {
    "chunk_duration_seconds": 120
  },
  "model": {
    "device": "cuda"
  }
}
```

### For Low Memory (Small GPU/CPU)
```json
{
  "video_processing": {
    "chunk_duration_seconds": 30
  },
  "evaluation": {
    "min_track_length_frames": 3
  }
}
```

### For Maximum Accuracy
```json
{
  "evaluation": {
    "id_switch_distance_threshold_px": 30,
    "speed_plausibility_threshold_px_per_frame": 100,
    "min_track_length_frames": 10
  }
}
```

## Integration with Existing Pipeline

### Using in Your Code
```python
from pipeline.video_chunking import VideoChunker
import json

# Load configuration
with open('config/pipeline_config.json', 'r') as f:
    config = json.load(f)

# Use in pipeline
chunker = VideoChunker(
    chunk_duration=config['video_processing']['chunk_duration_seconds']
)
```

### Using in Scripts
```bash
# Pass configuration via command line
python run_chunked_pipeline.py \
    --input video.mp4 \
    --chunk-duration 120 \
    --model models/best.pt
```

## Troubleshooting Configuration

### Issue: OOM Errors
→ Reduce `chunk_duration_seconds` in config

### Issue: Slow Processing
→ Increase `chunk_duration_seconds` or ensure GPU is enabled

### Issue: ID Switches Too High
→ Decrease `id_switch_distance_threshold_px`

### Issue: Too Many False Positives
→ Increase `min_track_length_frames`

## Advanced: Custom Homography

Pre-calibrated homography for same camera angle:

```json
{
  "homography": {
    "matrix_file": "config/homography_matrix.json",
    "use_cached": true
  }
}
```

The matrix file should contain (from homography picker):
```json
{
  "homography_matrix": [[...], [...], [...]],
  "pixel_vertices": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
  "court_width_m": 68,
  "court_length_m": 23.32
}
```

---

See `SETUP_AND_USAGE.md` for complete usage guide.
