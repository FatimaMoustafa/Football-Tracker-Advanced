## Chunk Boundary ID Continuity Validation

### Overview

The **Chunk Boundary Validator** is a regression test tool that verifies player ID continuity across chunk boundaries in the chunked processing pipeline. It automatically detects where chunks merge and validates that player IDs are properly linked from one chunk to the next.

### Features

#### Automatic Boundary Detection
- Detects chunk boundaries by identifying frames with high simultaneous ID changes
- Uses configurable threshold (default: 5+ players disappearing/arriving simultaneously)
- Works without prior knowledge of chunk boundaries

#### Continuity Analysis
For each detected boundary, the validator measures:
- **Preservation Rate**: Percentage of player IDs that continue across the boundary
- **Position Jump**: Average and max pixel distance for preserved IDs (should be smooth)
- **Quality Rating**: GOOD/FAIR/POOR based on preservation metrics

#### ID Retention Metrics
- Tracks how many IDs form long continuous tracks (≥5 seconds at 24fps)
- Computes ID retention rate (fraction of frames from long tracks)
- Identifies fragmentation caused by boundary issues

#### Overall Scoring
- **Continuity Score** (0-1): Higher is better
  - ≥ 0.85: Excellent (ID linking working well)
  - ≥ 0.65: Good (acceptable minor losses)
  - ≥ 0.40: Fair (recommend tuning)
  - < 0.40: Poor (significant issues)

### Integration with Evaluation Pipeline

The validator automatically runs during evaluation when using chunked processing:

```bash
python run_evaluation.py --input video.mp4 --output results/
```

When chunked processing is enabled (`--no-chunked` not specified), the validator:
1. Runs after track evaluation
2. Generates boundary continuity metrics JSON
3. Includes results in HTML evaluation report
4. Prints detailed console report

### Output Files

When validation runs, it produces:
- **boundary_continuity_metrics.json**: Detailed metrics in JSON format
- Console report with boundary-by-boundary analysis
- HTML integration showing continuity score and recommendations

### Metrics Explained

#### Boundary Frame Metrics
```json
{
  "boundary_frame": 720,           // Frame number where boundary was detected
  "preserved_ids": 18,              // Player IDs that continued
  "total_prev_ids": 20,            // Total IDs in previous chunk
  "total_curr_ids": 19,            // Total IDs in current chunk
  "preservation_rate": 0.9,        // Fraction preserved (18/20)
  "avg_position_jump": 12.5,       // Average pixel movement for preserved IDs
  "max_position_jump": 45.2,       // Maximum jump (should be < 200px)
  "quality": "GOOD"                // Quality rating
}
```

#### Summary Metrics
```json
{
  "continuity_score": 0.82,        // Overall score across all boundaries
  "num_boundaries": 3,             // Number of chunk boundaries detected
  "id_retention_rate": 0.87,       // Fraction of frames with long tracks
  "status": "PASS",                // PASS/WARN based on score
  "recommendation": "..."          // Actionable feedback
}
```

### Direct Usage (Standalone)

Use the validator outside the evaluation pipeline:

```python
from evaluation.chunk_boundary_validator import validate_chunk_boundary_continuity
import pickle

# Load merged tracks from pipeline
with open('output_videos/merged_tracks.pkl', 'rb') as f:
    merged_tracks = pickle.load(f)

# Run validation
metrics = validate_chunk_boundary_continuity(
    merged_tracks,
    output_json='boundary_validation.json'
)

print(f"Continuity Score: {metrics['continuity_score']:.3f}")
print(f"Status: {metrics['status']}")
```

### Troubleshooting

#### Low Continuity Score (< 0.65)

**Issue**: Many player IDs are not preserved across chunk boundaries

**Root Causes**:
1. Camera motion estimation errors causing position drift
2. `_link_chunk_boundary_ids()` max_dist threshold too small
3. Tracker instability near chunk boundaries

**Solutions**:
1. Increase `max_dist` parameter in `_link_chunk_boundary_ids()` (default 80px, try 120px)
2. Verify camera movement compensation is working properly
3. Review individual boundary frames for tracking anomalies
4. Consider reducing chunk duration for more stable tracking

#### High Position Jumps at Boundary

**Issue**: Preserved IDs show large pixel movements across boundary (> 200px)

**Causes**:
1. Camera movement not properly compensated
2. Perspective transformation offset
3. Mismatch in position coordinates between chunks

**Solutions**:
1. Check `camera_movement_estimator.py` optical flow quality
2. Verify homography matrix accuracy in `view_transformer.py`
3. Ensure position and position_adjusted fields are computed correctly

#### No Boundaries Detected

**Scenario**: Validator reports 0 boundaries (single chunk file)

**Meaning**: Either processing was single-chunk, or chunk linking was so successful that no major ID resets occurred at boundaries (good sign!)

### Performance Notes

- Validator adds minimal overhead (~1% to evaluation time)
- Runs automatically only when using chunked processing
- Can be disabled by running evaluation with `--no-chunked` flag
- JSON export adds < 1MB to results

### Integration with CI/CD

Suggested validation checks for automated pipelines:

```python
from evaluation.chunk_boundary_validator import validate_chunk_boundary_continuity

# ... run pipeline ...

metrics = validate_chunk_boundary_continuity(merged_tracks)

# Fail if continuity too low
assert metrics['continuity_score'] >= 0.70, \
    f"Chunk continuity failed: {metrics['continuity_score']:.3f}"

# Warn if medium quality
if metrics['continuity_score'] < 0.85:
    print(f"⚠️ Continuity warning: {metrics['recommendation']}")
```

### Implementation Details

The validator uses:
- **Boundary Detection**: Simultaneous ID disappearance/arrival pattern matching
- **Preservation Analysis**: Positional distance thresholding (50px default for matching)
- **Continuity Scoring**: Weighted average of preservation rates + consistency
- **Report Generation**: Console logging + JSON serialization + HTML embedding

The algorithm is robust to:
- Multi-chunk videos with multiple boundaries
- Single-chunk scenarios (reports as PASS with 1.0 score)
- Scenarios with partial ID matching
- Noisy position data and small jitter

---

**Related Code**:
- `run_chunked_pipeline.py::_link_chunk_boundary_ids()` - Cross-chunk ID linking
- `run_chunked_pipeline.py::_smooth_chunk_track_ids()` - Intra-chunk smoothing  
- `run_evaluation.py::evaluate_video()` - Evaluation entry point
- `evaluation/eval_harness.py` - Main tracking quality evaluation
