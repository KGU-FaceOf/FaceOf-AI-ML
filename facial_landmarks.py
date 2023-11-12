# 필요한 패키지 가져오기


from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import sys

# data = sys.argv[1]
# print(data)

def save_landmarks_coordinates(image, detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 흑백 이미지에서 얼굴 감지
    rects = detector(gray, 1)

    # 얼굴 감지 반복
    for (i, rect) in enumerate(rects):
        # 얼굴 영역의 특징점 결정 후 (x, y) 좌표를 NumPy 배열로 변환
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # 얼굴 특징점 (x, y) 좌표를 따로 배열에 저장
        landmarks_array = []
        for (x, y) in shape:
            landmarks_array.append((x, y))

        # 저장된 특징점 좌표 배열 출력
        print("얼굴 특징점 좌표:")
        for idx, landmark in enumerate(landmarks_array):
            print(f"인덱스 {idx}: {landmark}")

# dlib의 얼굴 검출기(HOG 기반) 초기화 및 얼굴 특징점 예측기 생성

print("start")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'D:\_FaceOf\facial-landmarks-master\shape_predictor_68_face_landmarks.dat')

print("model")
# 입력 이미지 불러오고 크기 조정 및 흑백으로 변환
image = cv2.imread(r'https://images.unsplash.com/photo-1438761681033-6461ffad8d80?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D')
# image = cv2.imread(r'D:\_FaceOf\facial-landmarks-master\images\choi1.jpg')
image = imutils.resize(image, width=500)

print("image")
# 얼굴 특징점 좌표를 추출하고 저장하는 함수 호출
save_landmarks_coordinates(image, detector, predictor)
