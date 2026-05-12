# ✅ INSTALLATION & VERIFICATION CHECKLIST

## Pre-Flight Check

### 1. FFmpeg Installation ⚠️ CRITICAL
```powershell
# Check if FFmpeg is installed
ffmpeg -version

# If not installed:
choco install ffmpeg

# Restart terminal after installation
# Then verify again:
ffmpeg -version
ffprobe -version
```
**Status**: [ ] FFmpeg installed and verified

### 2. Python Dependencies
```powershell
# Install requirements
pip install -r requirements_tracking_support.txt

# Verify installations
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import ultralytics; print('YOLOv8: OK')"
```
**Status**: [ ] All dependencies installed

### 3. Model File Check
```powershell
# Verify model exists
dir models/best.pt

# If missing: Download or place model in models/best.pt
```
**Status**: [ ] Model file present at models/best.pt

### 4. Input Video Check
```powershell
# Create test videos directory if needed
mkdir input_videos

# Add your test videos:
# - 10_minutes.mp4
# - 45_minutes.mp4
```
**Status**: [ ] Test video(s) ready in input_videos/

## Quick Verification Test

### Test 1: Homography Picker
```powershell
# Extract first frame from a test video
# Then run homography picker
python run_homography_picker.py ^
    --input input_videos/test_frame.jpg ^
    --output config/homography_matrix.json
```
**Expected**: Homography matrix saved
**Status**: [ ] Working

### Test 2: Quick Evaluation
```powershell
# Fast evaluation (5-10 min for 10min video)
python run_evaluation.py ^
    --input input_videos/10_min_sample.mp4 ^
    --output test_eval

# Check outputs
dir test_eval/
```
**Expected**: 
- evaluation_metrics.json ✓
- evaluation_report.txt ✓
- evaluation_report.html ✓

**Status**: [ ] Working

### Test 3: Chunked Processing
```powershell
# Process with chunking (safe on large videos)
python run_chunked_pipeline.py ^
    --input input_videos/45_minutes.mp4 ^
    --output test_processing ^
    --chunk-duration 60
```
**Expected**: 
- output_video_chunked.mp4 ✓
- stats/merged_stats.json ✓
- No crashes ✓

**Status**: [ ] Working

### Test 4: Complete Workflow
```powershell
# Full workflow
python run_complete_demo.py --full ^
    --input input_videos/test_video.mp4 ^
    --output test_workflow
```
**Expected**:
- Homography configured ✓
- Video processed ✓
- Evaluation generated ✓
- All reports created ✓

**Status**: [ ] Working

## Monitoring During Processing

### Things to Check:
```
✓ Memory stays constant (~150MB)
✓ CPU/GPU utilization is normal
✓ Progress logs appear every few seconds
✓ No red error messages
✓ Video chunks created and cleaned up
✓ Output video generated
```

### Timing Expectations:
- 10 min video: 2-3 min (GPU) / 8-10 min (CPU)
- 45 min video: 12-15 min (GPU) / 45-60 min (CPU)

## Output Verification

### After Complete Workflow:
```powershell
# Check output structure
tree test_workflow

# Files that MUST exist:
dir test_workflow\evaluation\evaluation_report.html

# Open and review
start test_workflow\evaluation\evaluation_report.html
```

### HTML Report Should Show:
- [x] Overall Quality Rating (A-F)
- [x] Summary Statistics
- [x] ID Switches count
- [x] Fragmentation score
- [x] Speed plausibility violations
- [x] Component scores with percentages
- [x] Visual styling and tables

## Troubleshooting During Testing

### If FFmpeg Error:
```
Error: "FFmpeg not found"
Solution: 
  1. choco install ffmpeg
  2. Restart PowerShell
  3. Verify: ffmpeg -version
  4. Try again
```

### If Out of Memory:
```
Error: "Out of Memory" or crash
This means chunking isn't working
Solution:
  1. Check FFmpeg is installed
  2. Verify chunk splitting in output
  3. Reduce chunk-duration to 30
```

### If Model Not Found:
```
Error: "Model file not found"
Solution:
  1. Verify: models/best.pt exists
  2. Or use: --model path/to/model.pt
```

### If No Window Opens (Homography):
```
Error: "No display" or window won't open
Solution:
  1. pip install opencv-python --upgrade
  2. Or use pre-saved homography.json
```

## Final Verification Checklist

- [ ] FFmpeg installed and verified
- [ ] Python packages installed
- [ ] Model file present
- [ ] Test video(s) available
- [ ] Homography picker test passed
- [ ] Quick evaluation test passed
- [ ] Chunked processing test passed
- [ ] Complete workflow test passed
- [ ] HTML report generated
- [ ] No errors in any test
- [ ] Memory stayed constant
- [ ] Processing time reasonable
- [ ] Output files created
- [ ] Video processed without crash

## ✅ All Systems GO!

If all boxes are checked, you're ready to:
```powershell
# Process your 10-minute video
python run_complete_demo.py --full ^
    --input input_videos/10_minutes.mp4 ^
    --output results_10min

# Process your 45-minute video
python run_complete_demo.py --full ^
    --input input_videos/45_minutes.mp4 ^
    --output results_45min

# Report to team
# ✓ No crashes
# ✓ Memory stable at ~150MB
# ✓ Quality metrics: [see reports]
```

## Quick Help

| Issue | Solution | Doc |
|-------|----------|-----|
| Setup | `QUICK_REFERENCE.md` | 1 page |
| FFmpeg | `SETUP_AND_USAGE.md` | Section: FFmpeg |
| Commands | `QUICK_REFERENCE.md` | 2 pages |
| Config | `CONFIG_GUIDE.md` | Full guide |
| Details | `SETUP_AND_USAGE.md` | Comprehensive |

---

**Next Step**: Run Test 1 above (Homography Picker) to verify everything works!
