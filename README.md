# FrodoBot Topological Navigation

Visual topological navigation for the FrodoBots Earth Rover using **CosPlace** (place recognition) + **LightGlue** (keypoint matching).

## How It Works

```
OFFLINE:
  Corridor Video → Extract Frames → CosPlace Features → Topological Graph

ONLINE:
  Camera Frame → CosPlace Retrieval → Current Node
                      ↓
                BFS Shortest Path → Next Waypoint
                      ↓
                LightGlue Match → Relative Direction (left/right/ahead)
                      ↓
                Motor Control → POST /control
```

- **CosPlace**: Fast coarse localization ("which part of the corridor am I in?")
- **LightGlue**: Precise spatial alignment at junctions ("is my target left or right?")
- Both pretrained, zero training needed

## Usage

### 1. Extract frames from corridor video
```bash
python scripts/01_extract_frames.py corridor_video.mp4 --fps 2
```

### 2. Build topological graph
```bash
python scripts/02_build_graph.py --frames data/frames --output data/graph.pkl
```

### 3. Navigate to target
```bash
# Start SDK first
cd earth-rovers-sdk && hypercorn main:app --reload

# Run navigation
python scripts/03_navigate.py --target target.jpg --graph data/graph.pkl
```

## Project Structure

```
frodobot-topnav/
  topnav/
    feature_extractor.py    # CosPlace descriptor extraction
    graph_builder.py        # Topological graph construction + BFS
    direction_estimator.py  # LightGlue direction estimation
    navigator.py            # Full navigation controller
  scripts/
    01_extract_frames.py    # Video → clean frames
    02_build_graph.py       # Frames → CosPlace features → graph
    03_navigate.py          # Live navigation to target image
  data/                     # Frames, graphs, features
  config/
```

## Performance

| Component | Latency | Device |
|-----------|---------|--------|
| CosPlace retrieval | ~15ms | GPU |
| LightGlue matching | ~30ms | GPU |
| BFS path planning | <1ms | CPU |
| Total control loop | ~50ms | Mixed |
