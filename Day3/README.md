## 인공신경망

1. **개요**
    
    ![Untitled (2)](https://github.com/python-Moon/Python_project/assets/162897281/30199703-2def-48c8-b27c-0a9f4e3fc60d)
    
    - 생물학적 신경망에서 영감을 받아 만들어진 컴퓨터 모델
    - 복잡한 패턴인식, 분류, 예측 등 다양한 AI 모델 생성에 활용
2. **퍼셉트론 (Perceptron)**
    - 인공신경망의 기본 구성 요소중 하나로, 간단한 형태의 인공 뉴런
    - 구조
        - 입력 (Input): 퍼셉트론에 들어오는 신호나 정보를 받는다. 들어온 각 입력에 가중치가 곱해져서 전달
        - 가중치 (Weight): 입력에 곱해지는 계수, 각 입력에 대한 중요도. 이 값이 조절되면서 퍼셉트론이 학습됨.
        - 활성화 함수 (Activation Function): 가중치와 입력의 곱의 합이 어떤 임계값을 넘으면 출력을 내보내는 함수.
        - 출력 (Output): 활성화 함수의 결과로 출력. 다음 층으로 전달되나, 최종 출력
    - 동작 원리
        - 입력과 가중치의 곱을 계산
            
            $$
            z = x_1⋅w_1 + x_2⋅w_2 + ... + x_n⋅w_n
            $$
            
        - 활성화 함수를 통과하여 출력을 계산
            
            $$
            y = activation(z)
            $$
            
        - (이진분류 같은 경우) 결과가 임계값을 넘으면 1, 아니면 0 출력
3. **다층 퍼셉트론**
    
    
    
    - 개요
        - 여러개의 퍼셉트론을 조합하여 비선형 문제를 해결
        - 하나의 퍼셉트론으로 선형 분류문제를 해결 가능 ⇒ 가중치를 조절해서 입력 공간을 적절히 분리하는 선을 찾는것
        - 한계: ex) 타집합 문제
    - 구조
        - 입력층 (Input Layer): 신경망에 입력되는 데이터의 특성이 위치하는 층. 입력층의 노드 개수는 입력 데이터의 특성 개수
        - 은닉층 (Hidden Layer): 입력층과 출력층 사이에서 복잡한 문제를 해결하기 위해 학습을 하는 층
        - 출력층 (Output Layer): 최종 결과를 출력하는 층, Output 노드 갯수는 문제의 성격마다 다를 수 있다
4. **학습 알고리즘**
    - 순전파 (Feedforward): 입력층에서 출력층으로 신호가 전달되는 과정
    - 오차 계산: 실제 출력과 모델의 예측 출력 간의 차이를 계산
    - 역전파 (Backpropagation): 출력층 ⇒ 입력층 역방향으로 오차를 전파하면서 가중치를 업데이트
    - 가중치 업데이트: 가중치 업데이트를 위한 최적화 알고리즘 경사 하강법
    - 반복: 위 과정이 반복되면서 오차를 최소화하고 모델의 성능을 향상시킴
5. **활성화 함수**
    - 개요
        - 인경신경망에서 각 퍼셉트론의 출력을 결정하는 함수
        - 입력 신호의 가중치의 합을 받아서 적절한 값을 출력
        - 활성화 함수는 비선형성을 부여
        - 두 가지로 나뉜다: 선형 활성화 함수, 비선형 활성화 함수
    - 선형 활성화 함수 (Linear Activation Function)
        - 입력에 가중치를 곱한 값 그대로를 출력
            
            $$
            f(z) = z
            $$
            
        - 층을 추가해서 다층으로 쌓여도 효과 없음
    - 비선형 활성화 함수 (Nonlinear Activation Function)
        - 다층으로 쌓이면 인공신경망이 복잡한 관계를 학습
        - 시그모이드 함수: 0과 1사이의 값을 출력
        - 하이퍼볼릭 탄젠트 함수:  -1과 1사이의 값을 출력
        - 렐루 함수 (Rectified Linear Unit, ReLU): 음수를 0으로 처리하고, 양수는 그대로 출력
        - softmax : 클래스 분류시 모델이 각 클래스에 속할 확률을 구할때 사용, 합이 1이 되도
6. **경사 소실 문제 (Vanishing Gradient Problem)**
    - 원리
        - 시그모이드 함수 or 하이퍼볼릭 탄젠트 활성화 함수
            - 특정범위에서 미분 값이 매우 작거나, 매우 크거나
        - 역전파 알고리즘:
            - 출력 ⇒ 입력 오차를 전파하면서 가중치를 조정
            - Gradient(기울기)를 계산하는데 미분값이 매우 작으면 기울기가 사라짐
        - Gradient 소실
            - 신경망 상위층의 가중치가 거의 업데이트 되지 않고, 따라서 해당 층에서 학습이 거의 이루어지지 않음
    - 영향:
        - 깊은 네트워크에서 발생: 여러 은닉층을 가진 신경망에서 문제가 두드러짐
        - 학습 속도 저하: Gradient가 소실되면 업데이트가 거의 이루어지지 않음 ⇒ 학습이 거의 안됨
        - 문제 해결 불가: 적절한 가중치를 찾을 수 없음
    - 해결방안
        - ReLU 활성화 함수 사용
        - 가중치 초기화
        - 배치 정규화 (Batch Normalization): Gradient 분포를 안정화
7. **과적합 문제 (Overfitting Problem)**
    - 모델이 학습 과정에서 너무 훈련 데이터에 맞춰져서 학습이 되는 경우
    - 새로운 데이터에 대한 일반화 성능이 떨어지는 단점
        1. 일반화 성능 하락: 과적합된 모델은 훈련 데이터에 대해서는 높은 정확도를 보이지만, 새로운 데이터에 대한 성능이 떨어지므로, 실제 환경에서는 제대로 동작하지 않음
        2. 불필요한 특징 학습: 과적합된 모델은 훈련데이터에 노이즈와 특정 예제에 대한 지나친 학습으로 인해 불필요한 특징이나 예외적인 패턴까지 학습
        3. 일반화 불가능성
        4. 모델의 해석력 감소: 과적합된 모델은 훈련 데이터에 대한 예측을 맞추려고 노력하면서 모델 내부의 가중치가 복잡해지기 때문에 모델 해석이 어려워진다.
    - 해결방안
        1. 더 많은 데이터 수집: 더 많은 다양한 데이터로 일반적인 패턴을 학습하도록
        2. 데이터 증강(Data Augmentation): 이미 있는 데이터를 변형하거나 확장해서 데이터 수를 늘림
        3. 모델 복잡도 제어: 모델의 복잡도를 줄이고 단순한 모델로 만듬. 가중치 규제
        4. 드롭아웃(drop-out): 랜덤하게 일부 퍼셉트론을 비활성화
        5. 교차 검증 사용: 데이터를 훈련-테스트 세트로 나누어 사용
        6. 조기 종료(Eary Stopping): 훈련을 조기 종료하여 과적합을 방지
8. **응용 예시**
    - 이미지 분류, 자연어 처리
9. **손실 함수 (Loss Function)**
    - 개요
        - 머신러닝 모델이 예측한 값과 실제 값 간의 차이를 측정하는 함수
        - 모델이 얼마나 정확한지를 평가하고, 이 차이를 최소화하는데 사용
        - Loss function이 낮을수록 모델의 예측이 실제 값과 일치하며, 높을수록 부정확
    - 종류
        - **Mean Squared Error (MSE)**: 회귀 문제에 주로 사용되는 손실함수, 예측 값과 실제 값 사이의 평균 제곱 오차를 계산
            
            ```python
            model.compile(loss='mean_squared_error', optimizer='')
            ```
            
        - **Binary Crossentropy**: 이진 분류 문제에 사용되는 손실 함수로, 두 클래스에 대한 확률 분포의 차이를 측정
            
            ```python
            model.compile(loss='binary_crossentropy', optimizer='')
            ```
            
        - **Categorical Crossentropy**: 카테고리 분류
            
            ```python
            model.compile(loss='categorical_crossentropy', optimizer='')
            ```
            
        - **Sparse Categorical Crossentropy**: 카테고리 분류
            
            ```python
            model.compile(loss='sparse_categorical_crossentropy', optimizer='')
            ```
            
        - **Hinge Loss**: 서포트 벡터 머신
            
            ```python
            model.compile(loss='hinge', optimizer='')
            ```
            
10. **최적화 알고리즘 (Optimizer)**
    - 개요
        - 모델이 학습하는 동안 가중치를 업데이트하는 알고리즘 정의
        - 모델은 Loss 값을 최소화하기위해서  Optimizer를 사용하여 가중치 조정
    - 종류
        - **Stochastic Gradient Descent (SGD)**: 가장 기본적인 최적화 알고리즘 중 하나, 손실함수의 기울기(gradient) 계산
            
            ```python
            model.compile(loss='', optimizer='sgd')
            ```
            
        - **Adam**: Adaptive Moment Estimation, learning rate를 동적으로 조절하면서 최적화 수행
            
            ```python
            model.compile(loss='', optimizer='adam')
            ```
            
        - **RMSprop**: Root Mean Square Propagation, 과거의 기울기 정보를 고려하여, learning rate 조절
            
            ```python
            model.compile(loss='', optimizer='rmsprop')
            ```
            
        - **Adagrad**: Adaptive Gradient Algorithm, 각 파라미터에 대해 learning rate를 조절
            
            ```python
            model.compile(loss='', optimizer='adagrad')
            ```
            
        - **Adadelta**: Adagrad를 변형, 좀 더 안정적으로 가중치 수렴
            
            ```python
            model.compile(loss='', optimizer='adadelta')
            ```
