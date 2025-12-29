#!/usr/bin/env python3
"""
Oelala OpenPose Real Image Test
Test OpenPose met een meer realistische gegenereerde afbeelding
"""

import sys
import os
import cv2
import numpy as np

# Stel LD_LIBRARY_PATH in voor OpenPose libraries
sys.path.append('/home/flip/openpose_py310/lib/python3.10/site-packages')
os.environ['LD_LIBRARY_PATH'] = '/home/flip/openpose_py310/lib/python3.10/site-packages'

import openpose.pyopenpose as op

def create_realistic_person():
    """Maak een meer realistische persoon afbeelding"""
    # Maak een afbeelding met natuurlijke achtergrond
    img = np.zeros((600, 800, 3), dtype=np.uint8)

    # Maak een natuurlijk kleurverloop achtergrond (gras/blauw)
    for y in range(600):
        for x in range(800):
            if y > 400:  # Gras onderkant
                r = min(255, max(0, 30 + np.random.randint(-10, 10)))
                g = min(255, max(0, 80 + np.random.randint(-10, 10)))
                b = min(255, max(0, 30 + np.random.randint(-10, 10)))
                img[y, x] = [r, g, b]
            else:  # Lucht bovenkant
                r = min(255, max(0, 100 + y//6 + np.random.randint(-15, 15)))
                g = min(255, max(0, 150 + y//4 + np.random.randint(-15, 15)))
                b = min(255, max(0, 200 + y//3 + np.random.randint(-15, 15)))
                img[y, x] = [r, g, b]

    # Voeg wat ruis toe aan de achtergrond
    noise = np.random.normal(0, 8, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)

    # Teken een persoon met huidskleur
    skin_tone = [70, 120, 180]  # Lichte huidskleur

    # Hoofd
    cv2.ellipse(img, (400, 120), (28, 35), 0, 0, 360, skin_tone, -1)
    # Hals
    cv2.ellipse(img, (400, 155), (12, 15), 0, 0, 360, skin_tone, -1)

    # Lichaam (T-shirt vorm)
    body_points = np.array([
        [370, 170], [430, 170], [440, 250], [420, 300], [380, 300], [360, 250]
    ], np.int32)
    cv2.fillPoly(img, [body_points], [200, 100, 80])  # Lichtblauw T-shirt

    # Armen
    # Linker arm
    cv2.ellipse(img, (355, 200), (18, 45), -20, 0, 360, skin_tone, -1)
    cv2.circle(img, (350, 245), 12, skin_tone, -1)  # Linker hand

    # Rechter arm
    cv2.ellipse(img, (445, 200), (18, 45), 20, 0, 360, skin_tone, -1)
    cv2.circle(img, (450, 245), 12, skin_tone, -1)  # Rechter hand

    # Broek
    cv2.rectangle(img, (385, 300), (415, 450), [100, 80, 60], -1)  # Donkerblauwe broek

    # Benen
    cv2.rectangle(img, (390, 450), (400, 520), skin_tone, -1)  # Linker been
    cv2.rectangle(img, (410, 450), (420, 520), skin_tone, -1)  # Rechter been

    # Schoenen
    cv2.rectangle(img, (385, 520), (405, 540), [20, 20, 20], -1)  # Linker schoen
    cv2.rectangle(img, (405, 520), (425, 540), [20, 20, 20], -1)  # Rechter schoen

    # Gezicht details
    cv2.circle(img, (390, 110), 2, [0, 0, 0], -1)  # Linker oog
    cv2.circle(img, (410, 110), 2, [0, 0, 0], -1)  # Rechter oog
    cv2.ellipse(img, (400, 125), (6, 3), 0, 0, 360, [50, 50, 50], -1)  # Mond

    # Haar
    cv2.ellipse(img, (400, 95), (30, 20), 0, 0, 360, [30, 20, 10], -1)

    # Voeg wat schaduwen toe voor diepte
    shadow = img.copy()
    shadow = cv2.GaussianBlur(shadow, (21, 21), 0)
    shadow = cv2.addWeighted(shadow, 0.3, img, 0.7, 0)
    img = shadow

    return img

def main():
    print("Oelala OpenPose Real Image Test")
    print("=" * 50)

    try:
        print("1. Initialiseren OpenPose...")
        wrapper = op.WrapperPython()
        params = dict()
        params["model_folder"] = "/home/flip/oelala/openpose/models/"
        params["net_resolution"] = "320x176"
        params["face"] = False
        params["hand"] = False
        wrapper.configure(params)
        wrapper.start()
        print("   ✓ OpenPose klaar")

        print("2. Creëren realistische persoon afbeelding...")
        person_img = create_realistic_person()
        cv2.imwrite("/home/flip/oelala/real_person.png", person_img)
        print("   ✓ Afbeelding opgeslagen")

        print("3. Uitvoeren pose estimation...")
        datum = op.Datum()
        datum.cvInputData = person_img
        datumVector = op.VectorDatum()
        datumVector.append(datum)
        wrapper.emplaceAndPop(datumVector)
        print("   ✓ Pose estimation uitgevoerd")

        print("4. Analyseren resultaten...")
        if datum.poseKeypoints is not None and len(datum.poseKeypoints) > 0:
            num_people = datum.poseKeypoints.shape[0]
            num_keypoints = datum.poseKeypoints.shape[1]
            print(f"   ✓ {num_people} persoon(en) gedetecteerd")
            print(f"   ✓ {num_keypoints} keypoints per persoon")

            # Tel keypoints met confidence > 0.1
            confident_keypoints = 0
            for person in datum.poseKeypoints:
                for keypoint in person:
                    if keypoint[2] > 0.1:
                        confident_keypoints += 1
            print(f"   ✓ {confident_keypoints} keypoints met hoge confidence")

            # Visualiseer keypoints
            result_img = person_img.copy()
            for i, keypoint in enumerate(datum.poseKeypoints[0]):
                x, y, confidence = keypoint
                if confidence > 0.1:
                    cv2.circle(result_img, (int(x), int(y)), 5, (0, 255, 0), -1)
                    cv2.putText(result_img, f"{i}", (int(x)+5, int(y)-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imwrite("/home/flip/oelala/real_person_pose.png", result_img)
            print("   ✓ Resultaat gevisualiseerd")

        else:
            print("   ⚠ Geen pose keypoints gedetecteerd")
            # Sla toch de originele afbeelding op
            cv2.imwrite("/home/flip/oelala/real_person_pose.png", person_img)

        print("\nTest voltooid!")
        print("Bestanden:")
        print("- real_person.png: Originele afbeelding")
        print("- real_person_pose.png: Met pose keypoints (indien gedetecteerd)")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
