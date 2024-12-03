import numpy as np
import cv2
from datetime import datetime
import math

def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

def find_center_of_quadrilateral(A, B, C, D):
    mid_AC = midpoint(A, C)
    mid_BD = midpoint(B, D)
    center = midpoint(mid_AC, mid_BD)
    return center

def distance_eye(facekps):
    eye_left = facekps[0]
    eye_right = facekps[1]
    return abs(eye_left[0] - eye_right[0])

def calculate_straight_score(landmarks, bbox):
    left_eye, right_eye, nose, left_mouth, right_mouth = landmarks
    
    bbox_width = bbox[0]
    bbox_height = bbox[1]
    bbox_size = max(bbox_width, bbox_height)
    
    center_eyes = [(left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2]
    
    diff_eye = abs(left_eye[0] - right_eye[0]) / bbox_size
    diff_mouth = abs(left_mouth[0] - right_mouth[0]) / bbox_size
    distance_eyes = math.sqrt((left_eye[0] - right_eye[0])**2 + (left_eye[1] - right_eye[1])**2) / bbox_size
    
    angle_eyes = math.atan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0])
    angle_nose = math.atan2(nose[1] - center_eyes[1], nose[0] - center_eyes[0])
    angle_deviation = abs(angle_eyes - angle_nose)
    
    symmetry = 1 - (diff_eye + diff_mouth) / distance_eyes
    
    w1, w2 = 0.8, 0.2 
    straight_score = w1 * symmetry - w2 * angle_deviation
    
    return straight_score + 1

arcface_dst = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32
)

def estimate_norm(lmk, image_size=112):
    # Điều chỉnh tỷ lệ và dịch chuyển
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio

    # Điều chỉnh điểm đích
    dst = arcface_dst * ratio
    dst[:, 0] += diff_x

    # Tính toán ma trận affine bằng OpenCV
    M, _ = cv2.estimateAffinePartial2D(lmk, dst)
    return M

def face_align(img, landmark, image_size=112):

    M = estimate_norm(landmark, image_size)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped

def is_image_blurry(image_path):
    if type(image_path) == str:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = image_path

    if image is None:
        raise ValueError("Could not open or read the image.")
    
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    variance = laplacian.var()
    return variance

def is_image_blurry1(image_path):
    if type(image_path) == str:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = image_path

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    height, width = gray.shape
    area = height * width
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_score = laplacian.var() / np.sqrt(area) * 1000

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_score = (np.mean(np.abs(sobelx)) + np.mean(np.abs(sobely))) / np.sqrt(area) * 1000

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    high_freq_score = np.mean(magnitude_spectrum) / np.sqrt(area) * 10

    local_var = cv2.blur(gray, (3,3))
    var_score = np.var(local_var) / np.sqrt(area) * 100

    combined_score = (laplacian_score + sobel_score + high_freq_score + var_score) / 4
    return combined_score

def get_time():
    return datetime.now().strftime('%H-%M-%S')

def get_day():
    return datetime.now().strftime('%Y%m%d')

def calculate_straight_score(landmarks, bbox):
    left_eye, right_eye, nose, left_mouth, right_mouth = landmarks
    
    bbox_width = bbox[0]
    bbox_height = bbox[1]
    bbox_size = max(bbox_width, bbox_height)
    
    center_eyes = [(left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2]
    
    diff_eye = abs(left_eye[0] - right_eye[0]) / bbox_size
    diff_mouth = abs(left_mouth[0] - right_mouth[0]) / bbox_size
    distance_eyes = math.sqrt((left_eye[0] - right_eye[0])**2 + (left_eye[1] - right_eye[1])**2) / bbox_size
    
    angle_eyes = math.atan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0])
    angle_nose = math.atan2(nose[1] - center_eyes[1], nose[0] - center_eyes[0])
    angle_deviation = abs(angle_eyes - angle_nose)
    
    symmetry = 1 - (diff_eye + diff_mouth) / distance_eyes
    
    w1, w2 = 0.8, 0.2 
    straight_score = w1 * symmetry - w2 * angle_deviation
    
    return straight_score + 1
