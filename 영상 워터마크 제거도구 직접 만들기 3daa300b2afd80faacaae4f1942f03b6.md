
# 동영상 워터마크 제거, 파이썬으로 직접 할 수 있습니다

유료 프로그램 없이 파이썬과 무료 오픈소스 라이브러리를 사용하면 충분히 워터마크를 제거하거나 가릴 수 있습니다. 코딩 난이도는 어떤 방식을 선택하느냐에 따라 나뉩니다.

## 1. 워터마크 제거의 3가지 접근법

- 화면 잘라내기(Crop): 가장 기초적인 방법입니다. 워터마크가 영상 구석에 있다면 그 부분만 잘라내어 저장합니다. 코드가 매우 단순합니다.
- 주변 픽셀로 덮어쓰기(Inpainting): 중간 난이도입니다. 워터마크 위치를 마스크로 지정하면, 프로그램이 주변 색상을 계산해 자연스럽게 지워진 것처럼 덮어씌웁니다.
- AI 모델 사용: 가장 어렵지만 결과물이 좋습니다. 영상 복원 딥러닝 모델을 사용해 지워진 영역을 정교하게 채워 넣습니다.

## 2. 파이썬 OpenCV를 활용한 기본 코드 (주변 픽셀로 덮어쓰기)

가장 일반적으로 접근할 수 있는 OpenCV 라이브러리의 Inpainting 기능을 활용한 뼈대 코드입니다. 시작 전 터미널에서 `pip install opencv-python` 명령어로 라이브러리를 설치해야 합니다.

Python

# 

```
import cv2
import numpy as np

# 동영상 파일 불러오기
cap = cv2.VideoCapture("input_video.mp4")

# 동영상 저장 설정 (해상도는 원본에 맞게 수정 필요)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter("output_video.mp4", fourcc, 30.0, (1920, 1080))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. 마스크 생성 (워터마크가 있는 위치를 하얗게 칠한 빈 도화지)
    # 실제 워터마크의 x, y 좌표와 크기에 맞게 사각형 영역을 설정해야 합니다.
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, (1500, 900), (1900, 1050), 255, -1)

    # 2. 인페인팅 적용 (워터마크 지우기)
    # cv2.INPAINT_TELEA 알고리즘을 사용해 주변 픽셀로 자연스럽게 채웁니다.
    result = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)

    out.write(result)

cap.release()
out.release()
cv2.destroyAllWindows()
```

## 3. 직접 코드를 짤 때의 한계점

위 코드를 실행하면 워터마크가 있던 자리가 살짝 뿌옇게 뭉개지는 느낌(블러 처리)으로 남게 됩니다. 완벽하게 티가 나지 않게 지우려면 고도화된 AI 모델을 다뤄야 합니다. 또한, 영상 속에서 워터마크가 계속 움직인다면 매 프레임마다 위치를 추적해 마스크 좌표를 바꿔주는 복잡한 로직이 추가로 필요합니다.

코딩이 조금 번거롭게 느껴진다면, 명령어 한 줄로 동영상을 편집할 수 있는 무료 오픈소스 도구인 FFmpeg의 delogo 필터를 사용하는 것도 훌륭한 대안입니다.