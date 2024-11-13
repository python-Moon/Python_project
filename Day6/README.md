## 데이터 증강 (Data Augmentation)

- 기존의 학습 데이터를 변형 및 조작하여 데이터의 다양성을 증가시키는 것
- 모델의 일반화 성능을 향상시키고, 과적합을 방지
- 데이터 부족으로 인한 문제 완화
- 실제 데이터 수집 및 라벨링 비용을 줄이고 모델의 성능 향상

### 이미지 데이터

- 회전, 이동, 크기 조정 등의 기하학적 변환
- 색상 조정, 반전, 노이즈 추가 등의 이미지 처리
- 이미지 잘라내기, 확대/축소 등의 처리

### 텍스트 데이터

- 동의어 교체, 단어 순서 변경 등의 텍스트 변형
- 랜덤 추가/삭제, 노이즈 추가 등의 텍스트 조작
- 단어 재배치, 문장 재구성 등의 텍스트 리포맷팅

### 오디오 데이터

- 노이즈 추가, 스피치 속도 조정 등의 오디오 처리
- 음성 변조, 환경 임의 추가 등의 오디오 변형

## 이미지 데이터 증강

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
	rotation_range=10,
	width_shift_range=0.1,
	height_shift_range=0.1,
	zoom_range=0.1,
	horizontal_flip=True,
)

train_generator = datagen.flow(x_train, y_train, batch_size=32)
```

1. `ImageDataGenerator` : 이미지 데이터를 증강하기 위한 Tensorflow의 유틸리티 클래스. 이 클래스를 사용하여 이미지 데이터를 동적으로 변환하고 증강
2. `rotation_range` : 이미지 회전 범위. (ex. `rotation_range=10`  ⇒ 이미지가 -10도 ~ 10도 랜덤)
3. `width_shift_range` : 이미지를 수평으로 이동시킬 범위 (ex. `width_shift_range=0.1`  ⇒ 이미지 넓이의 10% 범위 내 랜덤)
4. `height_shift_range` : 이미지를 수직으로 이동시킬 범위
5. `zoom_range` : 이미지를 확대/축소 할 범위 (ex. `zoom_range=0.1`  ⇒ 0.9배 ~ 1.1배 범위 내 랜덤)
6. `horizontal_flip` : 이미지를 수평으로 뒤집을지 여부 (ex. `horizontal_flip=True`  ⇒ 이미지가 50% 확률로 수평으로 뒤집힘)
7. `ImageDataGenerator.flow` : 데이터 증강을 적용하여 생성된 이미지 데이터의 배치를 반환
    - `flow()` 메소드를 사용하여 이미지 데이터와 레이블 데이터를 받아들이고, 지정된 데이터 증강 옵션에 따라 이미지를 동적으로 변환.
