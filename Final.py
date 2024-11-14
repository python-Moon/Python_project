from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt

cancer = load_breast_cancer()

x_data = cancer.data
y_data = cancer.target

print(x_data)

print('<Wisconsin Breast Cancer Data Set>')
print('  Data Shape: ', x_data.shape)
print('Target Shape: ', y_data.shape)

for i, feature in enumerate(cancer.feature_names):
    print(f'Feature {(i + 1)}: {feature}')


class LogisticRegression:
    def __init__(self, learning_rate=0.01):
        self.w = None  # 가중치
        self.b = None  # bias
        self.lr = learning_rate  # 학습률
        self.losses = []  # epoch > loss 저장
        self.weight_history = []  # epoch > weight 저장
        self.bias_history = []  # epoch > bias 저장

    # Forward Pass
    def forward(self, x):
        z = np.sum(self.w * x) + self.b
        z = np.clip(z, -50, None)
        return z

    # 손실함수
    def loss(self, y, a):
        return -(y * np.log(a) + (1 - y) * np.log(1 - a))

    # 활성화함수
    def activation(self, z):
        a = 1 / (1 + np.exp(-z))
        a = np.clip(a, 1e-10, 1- (1e-10))
        return a

    # Gradient 계산
    def gradient(self, x, y):
        z = self.forward(x)
        a = self.activation(z)

        w_grad = x * (-y + a)
        b_grad = -y + a

        return w_grad, b_grad

    def fit(self, x_data, y_data, epochs=30):
        self.w = np.ones(x_data.shape[1])  # 가중치 초기화
        self.b = 0  # 바이어스 초기화

        for epoch in range(epochs):
            l = 0  # epoch > 손실 누적 변수
            w_grad = np.zeros(x_data.shape[1])  # 가중치의 기울기
            b_grad = 0  # 바이어스 기울기

            for x, y in zip(x_data, y_data):
                z = self.forward(x)
                a = self.activation(z)

                l += self.loss(y, a)

                w_i, b_i = self.gradient(x, y)

                w_grad += w_i
                b_grad += b_i

            self.w -= self.lr * (w_grad/len(y_data))
            self.b -= self.lr * (b_grad/len(y_data))

            print(f'Epoch {(epoch+1)} ===> Loss: {l/len(y_data):.4f} | Weight: {self.w[0]:.4f} | Bias: {self.b:.4f}')
            self.losses.append(l/len(y_data))
            self.weight_history.append(self.w)
            self.bias_history.append(self.b)

model = LogisticRegression(learning_rate=0.0005)
model.fit(x_data, y_data, epochs=60)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(model.losses)
plt.show()
