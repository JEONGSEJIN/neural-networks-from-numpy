# 0. 사용한 라이브러리 및 모듈
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. 계층
class Layer_Dense:
    # 레이어 초기화
    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        # 가중치 및 편향 초기화
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        # 정규화 상수 설정
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    # 순방향
    def forward(self, inputs, training):
        # 계층 입력값
        self.inputs = inputs
        # 입력값 및 가중치, 편향으로 해당 계층 연산
        self.output = np.dot(inputs, self.weights) + self.biases

    # 역방향
    def backward(self, dvalues):
        # 가중치와 편향을 미분한 것 (= 기울기)
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # 정규화 상수를 미분한 것 (= 기울기)
        # L1 정규화일 때의 가중치
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        # L2 정규화일 때의 가중치
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        # L1 정규화일 때의 편향
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        # L2 정규화일 때의 편향
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        # 위에서 미분한 것들을 사용해서 구한 입력값
        self.dinputs = np.dot(dvalues, self.weights.T)

# 2. 드롭아웃
class Layer_Dropout:
    # 드롭아웃 설정
    def __init__(self, rate):
        # = 전체(1)에서 드롭시킬비율(rate)을 뺀 것 
        # 예시) 1(전체) - 0.2(드롭시킬비율) = 0.8(남는비율)
        self.rate = 1 - rate

    # 순방향
    def forward(self, inputs, training):
        # 계층 입력값
        self.inputs = inputs

        # 드롭아웃은 학습할 때만 사용.
        # 아래는, 검증이나 테스트할 때 사용.
        if not training:
            self.output = inputs.copy()
            return

        # 드롭아웃 시킬 때 사용할 마스크
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        # 마스크를 이용하여 드롭아웃 적용
        self.output = inputs * self.binary_mask

    # 역방향
    def backward(self, dvalues):
        # 위에서 미분한 것들을 사용해서 구한 입력값
        self.dinputs = dvalues * self.binary_mask

# 3. 입력 계층
class Layer_Input:
    def forward(self, inputs, training):
        self.output = inputs

# 4. 활성화 함수
# 4.1. ReLU 함수
class Activation_ReLU:
    # 순방향
    def forward(self, inputs, training):
        # 계층 입력값
        self.inputs = inputs
        # ReLU 계층의 입력으로 들어온 것으로 ReLU 연산
        self.output = np.maximum(0, inputs)

    # 역방향
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs):
        return outputs

# 4.2. Softmax 함수
class Activation_Softmax:
    # 순방향
    def forward(self, inputs, training):
        # 계층 입력값
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)
    
# 5. 손실 함수
# 5.1. 손실 함수 연산
class Loss:
    def regularization_loss(self):
        regularization_loss = 0
        for layer in self.trainable_layers:
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)
        return regularization_loss

    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def calculate(self, output, y, *, include_regularization=False):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        if not include_regularization:
            return data_loss
        return data_loss, self.regularization_loss()
    
# 5.2. Categorical Cross-Entropy Loss (다중 클래스 분류 문제에서 사용하는 손실 함수)
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
            
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples

# 5.3. Categorical Cross-Entropy +  Softmax Activation (by Chain Rule) 
class Activation_Softmax_Loss_CategoricalCrossentropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):

        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

# 6. 최적화
# Adam(Adaptive Momentum) Optimizer (: RMSProp Optimizer + momentum from SGD Optimizer)
class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
            
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2
        
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))
        
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

# 7. 정확도 구하기
# 7.1 정확도 연산
class Accuracy:
    def calculate(self, predictions, y):
        comparisons = self.compare(predictions, y)
        accuracy = np.mean(comparisons)
        return accuracy
    
# 7.2. 분류 모델에서 정확도 연산
class Accuracy_Categorical(Accuracy):
    def __init__(self, *, binary=False):
        self.binary = binary

    def init(self, y):
        pass

    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y
 
# 8. 모델
class Model:
    def __init__(self):
        self.layers = []
        self.softmax_classifier_output = None

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def finalize(self):
        self.input_layer = Layer_Input()
        layer_count = len(self.layers)
        self.trainable_layers = []

        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]

            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]

            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        self.loss.remember_trainable_layers(self.trainable_layers)
        if isinstance(self.layers[-1], Activation_Softmax) and isinstance(self.loss, Loss_CategoricalCrossentropy):
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()

    def train(self, X, y, *, epochs=100, print_every=10, test_data=None):
        self.accuracy.init(y)
        data_losses = []
        regularization_losses = []
        losses = []
        accuracies = []
        learning_rates = []

        # 학습 루프
        for epoch in range(1, epochs+1):

            # 총 계층 순방향 실행
            output = self.forward(X, training=True)

            # 손실값 계산
            data_loss, regularization_loss = self.loss.calculate(output, y, include_regularization=True)
            data_losses.append(data_loss)
            regularization_losses.append(regularization_loss)   
            loss = data_loss + regularization_loss
            losses.append(loss)
            
            # 예측 및 정확도 계산
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)
            accuracies.append(accuracy)
            learning_rates.append(self.optimizer.current_learning_rate)

            # 총 계층 역방향 실행
            self.backward(output, y)

            # 총 계층 최적화 실행
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

            # 에폭 10마다 출력
            if not epoch % print_every:
                print(f'epoch: {epoch}, ' +
                      f'acc: {accuracy:.3f}, ' +
                      f'loss: {loss:.3f} (' +
                      f'data_loss: {data_loss:.3f}, ' +
                      f'reg_loss: {regularization_loss:.3f}), ' +
                      f'lr: {self.optimizer.current_learning_rate}')

        # 학습데이터 정확도 및 손실값
        print(f'training, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}')

        plt.plot(range(100), data_losses)
        plt.title('(training) Data Loss')
        plt.show()
        plt.plot(range(100), regularization_losses)
        plt.title('(training) Regularization Loss')
        plt.show()
        plt.plot(range(100), losses)
        plt.title('(training) Loss')
        plt.show()
        plt.plot(range(100), accuracies)
        plt.title('(training) Accuracy')
        plt.show()
        plt.plot(range(100), learning_rates)
        plt.title('(training) Learning Rate')
        plt.show()
        
        # 테스트 루프
        if test_data is not None:
            X_test, y_test = test_data
            output = self.forward(X_test, training=False)
            loss = self.loss.calculate(output, y_test)
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y_test)

            # 테스트데이터 정확도 및 손실값
            print(f'test, ' +
                  f'acc: {accuracy:.3f}, ' +
                  f'loss: {loss:.3f}')

    def forward(self, X, training):
        self.input_layer.forward(X, training)
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        return layer.output

    def backward(self, output, y):
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return

        self.loss.backward(output, y)

        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)


# 9. 데이터셋 설정
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train_reshape = []
for x_train_ in x_train:
    x_train_ = x_train_.reshape(-1)
    x_train_reshape.append(x_train_)
x_train_reshape = np.array(x_train_reshape)
#print(x_train_reshape.shape)
#print(x_train_reshape[0].shape)

x_test_reshape = []
for x_test_ in x_test:
    x_test_ = x_test_.reshape(-1)
    x_test_reshape.append(x_test_)
x_test_reshape = np.array(x_test_reshape)
#print(x_test_reshape.shape)
#print(x_test_reshape[0].shape)

# 10. 학습 및 테스트
# 모델 선언
model = Model()

# 계층 추가
model.add(Layer_Dense(28*28, 128, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(Activation_ReLU())
#model.add(Layer_Dense(64, 128))
#model.add(Activation_ReLU())
#model.add(Layer_Dense(128, 256))
#model.add(Activation_ReLU())
model.add(Layer_Dense(128, 10))
#model.add(Activation_ReLU())
#model.add(Layer_Dropout(0.2))
model.add(Activation_Softmax())

# 손실함수, 옵티마이저, 정확도 설정
model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(learning_rate=0.05, decay=5e-5),
    accuracy=Accuracy_Categorical()
)

model.finalize()

# 학습 및 테스트
model.train(x_train_reshape, y_train, test_data=(x_test_reshape, y_test), epochs=100, print_every=10)
