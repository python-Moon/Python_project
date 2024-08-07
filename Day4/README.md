## 이미지 데이터의 특성과 처리

1. 이미지 데이터의 특성
    - 이미지는 픽셀로 구성된 행렬 형태의 데이터
    - 각 픽셀에 색상 정보를 담고있고, RGB 값이 표현
2. 이미지 데이터 처리
    - openCV 라이브러리 활용
    - **이미지 불러오기**: 이미지를 컴퓨터에서 읽어와서 처리
    - **이미지 전처리**: 이미지를 모델에 입력하기 전에 크기를 조정하거나 정규화
    - **데이터 증강**: 모델 성능 향상을 위해 이미지를 회전, 반전, 이동, 색상변경 등 변형
3. 이미지 데이터의 특징 추출
    - **특징 맵(Feature Map)**: 이미지의 주요 특성을 강조하는 작은 부분 이미지
    - **필터(Filter)**: 특정한 특징을 찾기위해 이미지를 스캔하는 작은 윈도우
    - **스트라이드(Stride)**: 필터가 이미지를 스캔할 때 이동하는 간격

## CNN(Convolutional Neural Network)

![Untitled (4)](https://github.com/user-attachments/assets/e049bd1d-922d-4040-9858-a08ff3976aaa)


1. **합성곱층 (Convolutional Layer)**: 이미지의 특징을 추출

![Untitled (5)](https://github.com/user-attachments/assets/a13eaafa-7f2c-413f-8751-a8d68bab24a5)

    
2. **풀링층 (Pooling Layer)**: 특징 맵의 크기를 줄이고 중요한 정보를 강조
    - 맥스 풀링 (Max Pooling): 특정 영역에서 가장 큰 값을 추출하여 특징을 강조
    - 평균 풀링 (Average Pooling): 특정 영역의 평균 값을 계산하여 정보를 압축
3. **완전 연결층 (Fully Connected Layer)**: 추출된 특징을 바탕으로 최종 예측을 수행
