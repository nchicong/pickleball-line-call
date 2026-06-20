# Pickleball Line Call — Android App

## Overview
Android app phát hiện bóng pickleball **out** trong thời gian thực. Phone đặt ở cuối sân hoặc giữa sân, quan sát 1 góc sân. Khi bóng out → nháy flash + beep.

## Tech Stack
| Component | Technology |
|---|---|
| Language | **Kotlin** (100%, no Java) |
| UI | **Jetpack Compose** |
| Camera | **CameraX** |
| ML | **TensorFlow Lite** (YOLOv8-nano INT8 quantized) |
| Build | **Gradle KTS** |
| Min SDK | **26** (Android 8) |
| Target SDK | **34** |

## Architecture

### Module structure
```
app/
├── MainActivity.kt              # Entry point, navigation
├── ui/
│   ├── calibration/
│   │   ├── CalibrationScreen.kt   # Chạm 4 góc sân
│   │   └── CalibrationViewModel.kt
│   └── play/
│       ├── PlayScreen.kt          # Real-time detection UI
│       └── PlayViewModel.kt
├── camera/
│   └── CameraManager.kt          # CameraX lifecycle + frame analysis
├── vision/
│   ├── LineDetector.kt           # OpenCV HoughLinesP
│   ├── BallDetector.kt           # Interface
│   ├── TFLiteBallDetector.kt     # YOLOv8-nano ball detection
│   ├── BounceDetector.kt         # Kalman filter + trajectory analysis
│   └── OutJudge.kt               # Bounce vs line boundaries
├── calibration/
│   └── CourtCalibrator.kt        # Perspective transform + court model
├── alert/
│   └── FlashAlert.kt             # Camera torch control
└── model/
    ├── CourtGeometry.kt          # Court dimensions, line coordinates
    └── DetectionResult.kt        # Frame analysis result data class
```

### Class Diagram (key relationships)
```
CourtCalibrator → CourtGeometry
       ↓
LineDetector (xác nhận calibration)
       
BallDetector (interface)
       ↑
TFLiteBallDetector → DetectionResult (bounding box)

BounceDetector (Kalman filter)
  ← DetectionResult (từ ball detector)
  → BounceEvent(x, y, confidence)

OutJudge
  ← BounceEvent + CourtGeometry (line boundaries)
  → OutCall(boolean)

PlayViewModel
  ← Frame → LineDetector + BallDetector + BounceDetector + OutJudge
  → UI state + FlashAlert trigger
```

## Data Flow (1 frame)
```
CameraX ImageProxy
  ↓
Convert to Bitmap (ARGB_8888)
  ↓
CourtCalibrator.warpToTopDown()
  ↓
LineDetector.detect(bmp) → List<Line>
  &  (verify calibration still valid)
TFLiteBallDetector.detect(bmp) → List<DetectionResult>
  ↓
BounceDetector.update(ballPositions)
  → detect bounce event → (x, y)
  ↓
OutJudge.judge(x, y, courtGeometry)
  → Out if outside any line boundary
  ↓
PlayViewModel → FlashAlert.flash() + update UI
```

## Bounce Detection Strategy
- **Input**: Ball bounding box centers over sequential frames
- **Method**: 2D Kalman filter tracking position + velocity
- **Bounce signal**: Sudden change in Y-velocity direction (up → down bounce means ball going up after bounce; but in camera view, the bounce shows as a "pause" or direction change in the trajectory)
- **Reality**: In a 2D top-down warped view, the bounce is when the ball appears to stop moving (velocity ~= 0) as it contacts the ground, then continues. But actually in top-down view, bounce might not be visible.
- **Revised approach**: Use **side/projected view** near line regions. Detect when ball disappears (occluded by ground) or shows motion aberration near line areas.
- **Simpler**: Use **confidence threshold** — when ball is detected near a line and has consistent trajectory, the lowest Y position in trajectory path is the bounce point.

### Practical bounce detection for Phase 1
1. Track ball over last N frames (trajectory buffer)
2. Fit trajectory curve
3. At each frame, project ball position onto court geometry
4. **Bounce = frame where ball position is closest to court surface plane** (lowest vertical component in warped space)
5. If ball consistently detected near line region → check if downstream from trajectory goes "past" line → out

## Device-Specific Notes

### Samsung S23+ (SD 8 Gen 2)
- **CameraX profile**: Use `CameraX Extensions` (HDR, stabilization)
- **TFLite delegate**: **GPU delegate** (OpenGL ES 3.1+) or **NNAPI delegate**
- **Expected FPS**: ~30-45fps detection loop
- **Slow-mo**: Can use 960fps capture later (Phase 2+)
- **Camera**: 4K@60fps input, keep 720p for analysis

### Detection at runtime
```kotlin
val delegate = when {
    hasGpuDelegate() -> GpuDelegate()
    hasNnapi(version = 2) -> NnApiDelegate()
    else -> null // CPU fallback
}
```

## Court Calibration

### User flow
1. User places phone at desired position
2. App shows camera preview with instruction overlay
3. User taps **4 corners of visible court area** in order:
   ```
   1 → 2 (far side, left to right)
   3 → 4 (near side, left to right)
   ```
4. `CourtCalibrator.computePerspectiveTransform()` maps image coords → top-down court
5. Using standard pickleball court dimensions (20ft × 44ft, kitchen 7ft from net):
   - Baseline: y=0
   - Kitchen line: y=7ft (from net side)
   - Centerline: x=10ft (midpoint)
   - Sidelines: x=0 and x=20ft
   - Net: y=44ft (or y=0 depending on orientation)
6. Saved to `SharedPreferences` as 4 corner points + transform matrix

### Visual feedback
- After calibration, overlay ALL court lines on preview
- Red dashed lines for boundaries, green for in-play area
- Allow recalibration with a button

## TFLite Model

### Model specs (to be created/trained later)
| Property | Value |
|---|---|
| Model | YOLOv8-nano |
| Input | 320×320×3 (RGB, INT8 quantized) |
| Output | 84×8400 tensor (box + class scores) |
| Classes | 1: `ball` |
| Quantization | INT8 (full integer) |
| Size target | ~3-5MB |
| FPS (S23+ GPU) | ~40 |
| FPS (S23+ NNAPI) | ~35 |

### Training data needs
- 500-1000 images of pickleball balls on courts
- Varied lighting: sunny, overcast, evening
- Varied angles: from baseline, from 45°, from sideline
- Ball at different distances (near: 40-60px, mid: 15-30px, far: 5-10px)
- Augment with rotation, brightness, contrast, blur

### Integration
```kotlin
// TFLiteBallDetector.kt
class TFLiteBallDetector(context: Context, useGpu: Boolean) {
    private val interpreter: Interpreter
    
    fun detect(bitmap: Bitmap): List<DetectionResult> {
        // 1. Preprocess: resize to 320x320, convert to INT8 byte array
        // 2. Run inference
        // 3. Postprocess: NMS (IoU threshold 0.5, confidence 0.3)
        // 4. Map boxes back to original image coordinates
        // 5. Return list of DetectionResult(x, y, w, h, confidence)
    }
}
```

## Out Decision Logic

After calibration, court lines form polygons:
- **In-bounds area**: The quadrant(s) being monitored
- **Out**: Ball bounce outside the polygon

```
Top-down view (phone at baseline looking toward net, monitoring one half):

                NET
  ═══════════════════════════════════
  ┌────────────────┬────────────────┐
  │                │                │
  │   FAR COURT    │   FAR COURT    │  ← far side (may be partially visible)
  │   (opponent)   │   (opponent)   │
  │                │                │
  ├────────────────┴────────────────┤  ← Far Kitchen Line (7ft from net)
  │                                 │
  │            NET CORRIDOR         │  ← Net area
  │                                 │
  ├────────────────┬────────────────┤  ← Near Kitchen Line
  │                │                │
  │   KITCHEN      │   KITCHEN      │  ← 7ft non-volley zone
  │   (Left)       │   (Right)      │
  │                │                │
  ├────────────────┴────────────────┤
  │                                 │
  │          PLAY AREA              │  ← 15ft (behind kitchen → baseline)
  │                                 │
  ├────────────────┬────────────────┤  ← Baseline
  │                │                │
  │   BASELINE     │   BASELINE     │
  │   (Left)       │   (Right)      │
  │                │                │
  └────────────────┴────────────────┘
           CENTER LINE

  OUT if bounce falls outside the line boundaries of the monitored half
  Ball on line = IN (per real rules)
```

### Edge cases
- Ball on line: counted as IN (like real rules)
- Ball in kitchen on serve: considered OUT (per pickleball rules)
- Confidence threshold: <0.3 → skip frame (no decision)
- Must have 3+ consistent detections before calling out (debounce)

## Implementation Order (Phase 1)

| Step | Task | Depends on |
|---|---|---|
| 1 | Scaffold: Gradle KTS, deps, theme, navigation | — |
| 2 | CameraX preview + frame analysis loop | Step 1 |
| 3 | CourtCalibrator + CalibrationScreen UI | Step 2 |
| 4 | LineDetector (OpenCV HoughLinesP) | Step 3 |
| 5 | CourtGeometry + line overlay on preview | Step 4 |
| 6 | TFLiteBallDetector (model integration) | Step 2 |
| 7 | BounceDetector (Kalman filter) | Step 6 |
| 8 | OutJudge (bounce vs court geometry) | Step 5 + 7 |
| 9 | FlashAlert + audio beep | Step 8 |
| 10 | PlayScreen UI (live status, stats) | Step 9 |
| 11 | S23+ optimization (GPU delegate, 720p) | Step 10 |

## Code Conventions
- **No comments in code** unless documenting public API
- File names: PascalCase matching class name
- ViewModel per screen: `*ViewModel.kt`
- All camera/image processing off main thread (`Dispatchers.Default`)
- UI updates via `StateFlow` + `collectAsState()`
- Use `sealed class` for UI state enums
- Data classes for all model objects
- No `lateinit var` in ViewModels; use `MutableStateFlow` with initial value

## Dependencies (latest stable as of 2026-06)
```kotlin
// build.gradle.kts (app)
dependencies {
    // AndroidX
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")
    
    // Compose
    implementation(platform("androidx.compose:compose-bom:2024.12.01"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.navigation:navigation-compose:2.8.5")
    
    // CameraX
    implementation("androidx.camera:camera-core:1.4.1")
    implementation("androidx.camera:camera-camera2:1.4.1")
    implementation("androidx.camera:camera-lifecycle:1.4.1")
    implementation("androidx.camera:camera-view:1.4.1")
    
    // OpenCV
    implementation("org.opencv:opencv-contrib:4.9.0")
    
    // TensorFlow Lite
    implementation("org.tensorflow:tensorflow-lite:2.16.1")
    implementation("org.tensorflow:tensorflow-lite-support:0.4.4")
    implementation("org.tensorflow:tensorflow-lite-gpu:2.16.1")
    implementation("org.tensorflow:tensorflow-lite-api:2.16.1")
    
    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
}
```

## Testing Strategy
- Unit tests: ViewModel logic, OutJudge, BounceDetector
- Instrumented tests: CameraX + frame pipeline (on real device)
- Manual testing: Samsung S23+ on actual pickleball court
- Ball detection accuracy: record 10 rallies, measure precision/recall
