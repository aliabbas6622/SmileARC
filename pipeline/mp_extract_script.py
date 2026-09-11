import os
import time
import numpy as np
from PIL import Image
from scipy.spatial.distance import euclidean
import mediapipe as mp

DATA_ROOT = "/marimo/data/smileArc"
CLASSES = ["consonant", "not available", "reverse- non consonant", "straight- non consonant"]
MODEL_PATH = "/marimo/face_landmarker.task"
CACHE_PATH = "/marimo/mediapipe_features.npz"

# Landmark indices
UPPER_LIP_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LOWER_LIP_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
UPPER_LIP_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LOWER_LIP_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
UPPER_LIP_CENTER = 13
LOWER_LIP_CENTER = 14
LIP_CENTER = 0
NOSE_TIP = 4
NOSE_BOTTOM = 2
PHILTRUM_SIDE = 97
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
LEFT_CHEEK = 234
RIGHT_CHEEK = 454


def _fit_parabola_deviation(points_2d):
    if len(points_2d) < 3:
        return 0.0
    pts = np.array(points_2d)
    x, y = pts[:, 0], pts[:, 1]
    chord_y = np.interp(x, [x[0], x[-1]], [y[0], y[-1]])
    try:
        coeffs = np.polyfit(x, y, 2)
        fitted_y = np.polyval(coeffs, x)
        deviation = np.abs(fitted_y - chord_y).max()
        return deviation
    except Exception:
        return 0.0


def extract_lip_features(landmarks_list, img_w=1, img_h=1):
    lm = np.array([[l.x * img_w, l.y * img_h, l.z] for l in landmarks_list])
    eye_dist = euclidean(lm[LEFT_EYE_OUTER], lm[RIGHT_EYE_OUTER])
    if eye_dist < 1e-6:
        eye_dist = 1.0
    face_w = euclidean(lm[LEFT_CHEEK], lm[RIGHT_CHEEK])
    if face_w < 1e-6:
        face_w = 1.0

    left_corner = lm[MOUTH_LEFT]
    right_corner = lm[MOUTH_RIGHT]
    upper_center = lm[UPPER_LIP_CENTER]
    lower_center = lm[LOWER_LIP_CENTER]
    nose_tip = lm[NOSE_TIP]

    upper_pts = lm[UPPER_LIP_OUTER][:, :2]
    lower_pts = lm[LOWER_LIP_OUTER][:, :2]
    upper_curv = _fit_parabola_deviation(upper_pts) / eye_dist
    lower_curv = _fit_parabola_deviation(lower_pts) / eye_dist
    curv_asym = upper_curv - lower_curv

    mouth_w = euclidean(left_corner, right_corner)
    mouth_h = euclidean(upper_center, lower_center)
    aspect_ratio = mouth_w / (mouth_h + 1e-6)
    mouth_w_norm = mouth_w / eye_dist

    upper_outer_avg = np.mean([euclidean(lm[i], lm[j]) for i, j in zip(UPPER_LIP_OUTER, UPPER_LIP_OUTER[1:])])
    upper_inner_avg = np.mean([euclidean(lm[i], lm[j]) for i, j in zip(UPPER_LIP_INNER, UPPER_LIP_INNER[1:])])
    lower_outer_avg = np.mean([euclidean(lm[i], lm[j]) for i, j in zip(LOWER_LIP_OUTER, LOWER_LIP_OUTER[1:])])
    lower_inner_avg = np.mean([euclidean(lm[i], lm[j]) for i, j in zip(LOWER_LIP_INNER, LOWER_LIP_INNER[1:])])
    upper_teeth_gap = (upper_outer_avg - upper_inner_avg) / eye_dist
    lower_teeth_gap = (lower_outer_avg - lower_inner_avg) / eye_dist
    open_ratio = mouth_h / (euclidean(upper_center, lm[0]) + euclidean(lower_center, lm[0]) + 1e-6)

    center_x = (left_corner[0] + right_corner[0]) / 2
    center_y = (left_corner[1] + right_corner[1]) / 2
    left_ox = (left_corner[0] - center_x) / eye_dist
    left_oy = (left_corner[1] - center_y) / eye_dist
    right_ox = (right_corner[0] - center_x) / eye_dist
    right_oy = (right_corner[1] - center_y) / eye_dist
    corner_rise = -(left_oy + right_oy) / 2

    n_upper = min(len(UPPER_LIP_OUTER), len(UPPER_LIP_INNER))
    upper_thickness = np.mean([euclidean(lm[UPPER_LIP_OUTER[i]], lm[UPPER_LIP_INNER[i]]) for i in range(n_upper)]) / eye_dist
    n_lower = min(len(LOWER_LIP_OUTER), len(LOWER_LIP_INNER))
    lower_thickness = np.mean([euclidean(lm[LOWER_LIP_OUTER[i]], lm[LOWER_LIP_INNER[i]]) for i in range(n_lower)]) / eye_dist

    nose_to_lip = euclidean(nose_tip, upper_center) / eye_dist
    philtrum_w = euclidean(lm[NOSE_BOTTOM], lm[PHILTRUM_SIDE]) / eye_dist
    nose_angle = np.arctan2(upper_center[1] - nose_tip[1], upper_center[0] - nose_tip[0])

    mouth_symmetry = np.abs(left_oy - right_oy)
    face_ratio = mouth_w / face_w

    features = np.array([
        upper_curv, lower_curv, curv_asym,
        mouth_w_norm, aspect_ratio,
        upper_teeth_gap, lower_teeth_gap, open_ratio,
        left_ox, left_oy, right_ox, right_oy, corner_rise,
        upper_thickness, lower_thickness,
        nose_to_lip, philtrum_w, nose_angle,
        mouth_symmetry, face_ratio,
    ], dtype=np.float32)
    return features


def main():
    if os.path.exists(CACHE_PATH):
        print("Cache already exists:", CACHE_PATH)
        return

    samples = []
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    for cls in CLASSES:
        cls_dir = os.path.join(DATA_ROOT, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                samples.append((os.path.join(cls_dir, fname), class_to_idx[cls]))
    print("Total samples:", len(samples))

    base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.1,
        min_tracking_confidence=0.1,
    )

    features_list = []
    labels_list = []
    paths_list = []
    failed = 0

    t0 = time.time()
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        for i, (path, label) in enumerate(samples):
            try:
                mp_image = mp.Image.create_from_file(path)
                results = landmarker.detect(mp_image)
                if results.face_landmarks:
                    landmarks = results.face_landmarks[0]
                    img = np.array(Image.open(path).convert("RGB"))
                    feats = extract_lip_features(landmarks, img.shape[1], img.shape[0])
                else:
                    feats = np.zeros(20, dtype=np.float32)
                    failed += 1
            except Exception as e:
                feats = np.zeros(20, dtype=np.float32)
                failed += 1

            features_list.append(feats)
            labels_list.append(label)
            paths_list.append(path)

            if (i + 1) % 200 == 0:
                elapsed = time.time() - t0
                print("  %d/%d processed (%.1fs, %d failed)" % (i + 1, len(samples), elapsed, failed), flush=True)

    features_arr = np.array(features_list, dtype=np.float32)
    labels_arr = np.array(labels_list, dtype=np.int64)
    paths_arr = np.array(paths_list)

    np.savez(CACHE_PATH, features=features_arr, labels=labels_arr, paths=paths_arr)
    elapsed = time.time() - t0
    print("Extracted features for %d images in %.1fs" % (len(samples), elapsed), flush=True)
    print("  Failed face detection: %d/%d" % (failed, len(samples)), flush=True)
    print("  Features shape:", features_arr.shape, flush=True)
    print("  Cached to:", CACHE_PATH, flush=True)


if __name__ == "__main__":
    main()