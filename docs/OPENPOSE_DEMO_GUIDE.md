# OpenPose Demo Handleiding - Oelala Project

## Overzicht

Deze handleiding beschrijft hoe je the OpenPose demo scripts kunt use to pose estimation te testen and te valideren in het Oelala project.

## Vereisten

### Environment Setup
```bash
# Activeer Python 3.10 environment
source /home/flip/openpose_py310/bin/activate

# Stel library paths in
export LD_LIBRARY_PATH=/home/flip/openpose_py310/lib/python3.10/site-packages:$LD_LIBRARY_PATH
export PYTHONPATH=/home/flip/openpose_py310/lib/python3.10/site-packages:$PYTHONPATH
```

### Controleer Installatie
```bash
# test OpenPose import
Python -c "import openpose.pyopenpose as on; print('OpenPose import: OK')"

# Controleer CUDA
Python -c "import openpose.pyopenpose as on; print(f'GPU available: {on.get_gpu_number() > 0}')"

# Controleer modellen
ls -la /home/flip/oelala/openpose/models/pose/
```

## Demo Scripts

### 1. Basis Demo (`demo_openpose.py`)

**Doel**: Basis pose estimation demonstratie with webcam of test afbeelding.

**use**:
```bash
cd /home/flip/oelala
Python demo_openpose.py
```

**which het doet**:
- Laadt OpenPose with BODY_25 model
- Detecteert poses in real-time (webcam) of statische afbeelding
- Toont keypoints and verbindingen visueel
- Print keypoint coördinaten to console

**Parameters** (aanpasbaar in script):
- `model_folder`: Pad to OpenPose modellen
- `net_resolution`: Netwerk resolutie (lager = sneller)
- `number_people_max`: Max aantal personen
- `render_pose`: Rendering to/out

### 2. Real Image test (`test_real_image.py`)

**Doel**: test pose estimation on echte afbeeldingen.

**use**:
```bash
cd /home/flip/oelala
Python test_real_image.py
```

**which het doet**:
- Laadt test afbeelding (sample.jpg)
- Voert pose estimation out
- Slaat resultaten on as keypoints.npy
- Genereert geannoteerde output afbeelding
- Print Detailed keypoint informatie

**Vereiste bestanden**:
- `sample.jpg`: test afbeelding in dezelfde directory

### 3. validation Script (`test_openpose.py`)

**Doel**: extensive validation of OpenPose installatie.

**use**:
```bash
cd /home/flip/oelala
Python test_openpose.py
```

**which het doet**:
- Controleert all OpenPose components
- test model loading
- Valideert Python bindings
- Checks GPU/CPU availability
- test basis pose estimation functionality

## Uitvoer Voorbeelden

### Console Output (demo_openpose.py)
```
OpenPose Python Wrapper initialized
GPU available: True
Model loaded: BODY_25
Processing frame...
Person 0 keypoints:
  Nose: (320.5, 180.2) confidence: 0.95
  Neck: (320.1, 220.8) confidence: 0.92
  RShoulder: (280.3, 225.1) confidence: 0.88
  ...
FPS: 15.2
```

### Keypoint Data structure
```Python
# Shape: (num_people, 25, 3)
keypoints = [
    [
        [320.5, 180.2, 0.95],  # Nose
        [320.1, 220.8, 0.92],  # Neck
        [280.3, 225.1, 0.88],  # RShoulder
        # ... 22 meer keypoints
    ]
]
```

## Troubleshooting

### Common Issues

#### 1. Import Error
```
ModuleNotFoundError: No module named 'openpose'
```
**Oplossing**:
```bash
# Controleer library path
export LD_LIBRARY_PATH=/home/flip/openpose_py310/lib/python3.10/site-packages:$LD_LIBRARY_PATH

# Herinstalleer libraries
cp /home/flip/oelala/openpose/build/Python/openpose/pyopenpose.cpython-310-x86_64-linux-gnu.so /home/flip/openpose_py310/lib/python3.10/site-packages/openpose/
```

#### 2. CUDA Error
```
CUDA driver version is insufficient
```
**Oplossing**:
```bash
# Controleer CUDA versie
nvidia-smi
nvcc --version

# Force CPU mode in script
params["num_gpu_start"] = 0
```

#### 3. Model Not Found
```
Error: Model file not found
```
**Oplossing**:
```bash
# Controleer model directory
ls -la /home/flip/oelala/openpose/models/pose/

# Download modellen indien nodig
cd /home/flip/oelala/openpose/models
./getModels.sh
```

#### 4. Memory Error
```
CUDA out of memory
```
**Oplossing**:
```Python
# Verminder resolutie
params["net_resolution"] = "256x144"

# Beperk aantal personen
params["number_people_max"] = 1
```

## Performance Tips

### for Snelheid
- use lagere `net_resolution`: "320x176" i.p.v. "656x368"
- Stel `number_people_max` in on verwacht aantal personen
- Zet `render_pose` on 0 for headless processing
- use GPU if available

### for Nauwkeurigheid
- use hogere `net_resolution`: "656x368"
- Houd `number_people_max` ruim (standaard -1 = ongelimiteerd)
- use BODY_25 model for meeste keypoints

## Uitbreiding Demo Scripts

### Eigen Afbeelding Testen
```Python
# in test_real_image.py
image_path = "jouw_afbeelding.jpg"
image = cv2.imread(image_path)

# Verwerk afbeelding
datum = on.Datum()
datum.cvInputData = image
# ... rest of processing
```

### Batch Processing
```Python
# Meerdere afbeeldingen
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = []

for path in image_paths:
    image = cv2.imread(path)
    # Process image
    # Save results
    results.append(keypoints)
```

### Video Processing
```Python
# Video frame processing
cap = cv2.VideoCapture("video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame
    datum = on.Datum()
    datum.cvInputData = frame
    # ... processing ...
    # Save frame keypoints

cap.release()
```

## Integratie Voorbeelden

### with Avatar Generatie
```Python
# use pose keypoints for avatar positioning
def create_avatar_with_pose(reference_image, pose_keypoints):
    # Extract key body points
    nose = pose_keypoints[0][:2]  # x, y
    left_shoulder = pose_keypoints[5][:2]
    right_shoulder = pose_keypoints[2][:2]

    # generate avatar in pose
    avatar = generate_avatar(reference_image, pose_keypoints)

    return avatar
```

### with Video Pipeline
```Python
# Continue pose tracking about video
previous_keypoints = None

for frame in video_frames:
    current_keypoints = process_frame(frame)

    if previous_keypoints is not None:
        # Smooth pose transitions
        smoothed_keypoints = smooth_poses(previous_keypoints, current_keypoints)

    previous_keypoints = current_keypoints
```

## Bestanden structure

```
/home/flip/oelala/
├── demo_openpose.py          # Basis demo
├── test_real_image.py        # Afbeelding test
├── test_openpose.py          # validation script
├── sample.jpg               # test afbeelding
├── openpose/                # OpenPose source
│   ├── models/             # Pre-trained modellen
│   └── build/              # Build artifacts
└── docs/
    └── OPENPOSE_TECHNICAL_GUIDE.md
```

## Volgende Stappen

1. **test all demo scripts** to functionality te bevestigen
2. **Experimenteer with parameters** for optimale performance
3. **Integreer with video processing** for real-time toepassingen
4. **Combineer with avatar generatie** for complete pipeline

---

*Deze handleiding is specifiek for the Oelala OpenPose implementation*
*Laatste update: September 7, 2025*
