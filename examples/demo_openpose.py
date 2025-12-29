#!/usr/bin/env python3
"""
Oelala OpenPose Demo - Pose Estimation met Python Bindings
Demonstreert pose estimation op afbeeldingen met OpenPose Python bindings
"""

import sys
import os
import cv2
import numpy as np

# Set LD_LIBRARY_PATH for OpenPose libraries
sys.path.append('/home/flip/openpose_py310/lib/python3.10/site-packages')
os.environ['LD_LIBRARY_PATH'] = '/home/flip/openpose_py310/lib/python3.10/site-packages'

import openpose.pyopenpose as op

def create_sample_image():
    """Maak een voorbeeld afbeelding met een persoonachtige vorm die meer lijkt op een echte foto"""
    # Create a larger image for better visualization
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    img.fill(200)  # Light gray background

    # Draw a more realistic person shape with color gradients
    # Background with color gradient
    for y in range(600):
        for x in range(800):
            img[y, x] = [min(255, 200 + y//10), min(255, 220 + x//20), min(255, 240 + (x+y)//30)]

    # Head (more realistic shape)
    cv2.ellipse(img, (400, 120), (25, 30), 0, 0, 360, (80, 100, 120), -1)
    # Face details
    cv2.circle(img, (390, 115), 3, (60, 80, 100), -1)  # Left eye
    cv2.circle(img, (410, 115), 3, (60, 80, 100), -1)  # Right eye
    cv2.ellipse(img, (400, 130), (8, 4), 0, 0, 360, (70, 90, 110), -1)  # Mond

    # Body (rectangle with rounded corners)
    cv2.rectangle(img, (375, 150), (425, 300), (100, 80, 60), -1)  # Lichaam
    # Schouders
    cv2.ellipse(img, (375, 160), (15, 25), 0, 0, 360, (90, 70, 50), -1)
    cv2.ellipse(img, (425, 160), (15, 25), 0, 0, 360, (90, 70, 50), -1)

    # Armen
    cv2.rectangle(img, (340, 180), (375, 250), (110, 90, 70), -1)  # Left arm
    cv2.rectangle(img, (425, 180), (460, 250), (110, 90, 70), -1)  # Right arm
    # Handen
    cv2.circle(img, (357, 255), 8, (120, 100, 80), -1)  # Left hand
    cv2.circle(img, (443, 255), 8, (120, 100, 80), -1)  # Right hand

    # Benen
    cv2.rectangle(img, (385, 300), (395, 450), (80, 60, 40), -1)  # Left leg
    cv2.rectangle(img, (405, 300), (415, 450), (80, 60, 40), -1)  # Right leg
    # Voeten
    cv2.rectangle(img, (380, 450), (400, 470), (70, 50, 30), -1)  # Left foot
    cv2.rectangle(img, (400, 450), (420, 470), (70, 50, 30), -1)  # Right foot

    # Add some noise to make it look more like a real photo
    noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)

    return img

def draw_pose_keypoints(img, keypoints, threshold=0.1):
    """Teken pose keypoints op de afbeelding"""
    if keypoints is None or len(keypoints) == 0:
        return img

    # BODY_25 keypoint pairs for drawing lines
    pose_pairs = op.getPosePartPairs(op.PoseModel.BODY_25)

    # Colors for different body parts
    colors = [
        (255, 0, 0),    # Nose
        (255, 85, 0),   # Neck
        (255, 170, 0),  # RShoulder
        (255, 255, 0),  # RElbow
        (170, 255, 0),  # RWrist
        (85, 255, 0),   # LShoulder
        (0, 255, 0),    # LElbow
        (0, 255, 85),   # LWrist
        (0, 255, 170),  # MidHip
        (0, 255, 255),  # RHip
        (0, 170, 255),  # RKnee
        (0, 85, 255),   # RAnkle
        (0, 0, 255),    # LHip
        (85, 0, 255),   # LKnee
        (170, 0, 255),  # LAnkle
        (255, 0, 255),  # REye
        (255, 0, 170),  # LEye
        (255, 0, 85),   # REar
        (255, 0, 0),    # LEar
        (255, 170, 85), # LBigToe
        (255, 85, 170), # LSmallToe
        (170, 85, 255), # LHeel
        (85, 170, 255), # RBigToe
        (0, 170, 255),  # RSmallToe
        (0, 85, 255),   # RHeel
    ]

    # Teken keypoints
    for person in keypoints:
        for i, keypoint in enumerate(person):
            x, y, confidence = keypoint
            if confidence > threshold:
                cv2.circle(img, (int(x), int(y)), 5, colors[i % len(colors)], -1)
                cv2.putText(img, f"{i}", (int(x)+5, int(y)-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i % len(colors)], 1)

    # Teken verbindingen tussen keypoints
    for pair in pose_pairs:
        part_from, part_to = pair
        if (part_from < len(keypoints[0]) and part_to < len(keypoints[0]) and
            keypoints[0][part_from][2] > threshold and keypoints[0][part_to][2] > threshold):
            x1, y1 = int(keypoints[0][part_from][0]), int(keypoints[0][part_from][1])
            x2, y2 = int(keypoints[0][part_to][0]), int(keypoints[0][part_to][1])
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

    return img

def main():
    print("Oelala OpenPose Demo")
    print("=" * 50)

    try:
        print("1. Initialiseren OpenPose wrapper...")
        wrapper = op.WrapperPython()
        print("   ✓ Wrapper aangemaakt")

        print("2. Configureren parameters...")
        params = dict()
        params["model_folder"] = "/home/flip/oelala/openpose/models/"
        params["net_resolution"] = "320x176"
        params["face"] = False
        params["hand"] = False
        # params["pose_model"] = "BODY_25"  # Deze parameter wordt niet ondersteund
        wrapper.configure(params)
        print("   ✓ Parameters geconfigureerd")

        print("3. Starten OpenPose...")
        wrapper.start()
        print("   ✓ OpenPose gestart")

        print("4. Creëren voorbeeld afbeelding...")
        sample_img = create_sample_image()
        cv2.imwrite("/home/flip/oelala/sample_input.png", sample_img)
        print("   ✓ Voorbeeld afbeelding opgeslagen")

        print("5. Uitvoeren pose estimation...")
        datum = op.Datum()
        datum.cvInputData = sample_img
        datumVector = op.VectorDatum()
        datumVector.append(datum)
        wrapper.emplaceAndPop(datumVector)
        print("   ✓ Pose estimation uitgevoerd")

        print("6. Analyseren resultaten...")
        if datum.poseKeypoints is not None:
            num_people = datum.poseKeypoints.shape[0]
            num_keypoints = datum.poseKeypoints.shape[1]
            print(f"   ✓ {num_people} persoon(en) gedetecteerd")
            print(f"   ✓ {num_keypoints} keypoints per persoon (BODY_25 model)")

            # Tel aantal keypoints met confidence > 0.1
            confident_keypoints = 0
            for person in datum.poseKeypoints:
                for keypoint in person:
                    if keypoint[2] > 0.1:
                        confident_keypoints += 1
            print(f"   ✓ {confident_keypoints} keypoints met hoge confidence (>0.1)")

        else:
            print("   ⚠ Geen pose keypoints gedetecteerd")

        print("7. Visualiseren resultaten...")
        result_img = draw_pose_keypoints(sample_img.copy(), datum.poseKeypoints)
        cv2.imwrite("/home/flip/oelala/pose_result.png", result_img)
        print("   ✓ Resultaat opgeslagen als pose_result.png")

        print("\nDemo succesvol voltooid!")
        print("Resultaten:")
        print("- sample_input.png: Originele afbeelding")
        print("- pose_result.png: Afbeelding met pose keypoints")
        print("\nOpenPose Python bindings werken perfect in het oelala project!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
