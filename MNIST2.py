#########MNIST 오차역전파로 구현 #############
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime

#MNIST dataset 불러오기
#raw는 60000,28,28 배열. 60000x784로 바꿔주기
(raw_x_train,t_train),(raw_x_test,t_test) = tf.keras.datasets.mnist.load_data()
x_train = raw_x_train.reshape(raw_x_train.shape[0],-1)
x_test = raw_x_test.reshape(raw_x_test.shape[0],-1)
#external funtion
def sigmoid(x):
    return 1/(1+np.exp(-x))

#class 정의
class NeuralNetwork:
    def __init__(self,input_nodes,hidden_nodes,output_nodes,learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        #은닉층 가중치 W2 = 784x100. Xavier/He 방법으로 self.W2 가중치 초기화
        self.W2 = np.random.randn(self.input_nodes,self.hidden_nodes)/np.sqrt(self.input_nodes/2)
        self.b2 = np.random.rand(self.hidden_nodes)

        #출력층 가중치 W3 = 100x10. Xavier/He 방법으로 self.W3 가중치 초기화
        self.W3 = np.random.randn(self.hidden_nodes,self.output_nodes)/np.sqrt(self.hidden_nodes/2)
        self.b3 = np.random.rand(self.output_nodes)

        #출력층 선형회귀값 Z3,출력값 A3 정의 (모두 행렬로 표시)
        self.Z3 = np.zeros([1,output_nodes])
        self.A3 = np.zeros([1,output_nodes])
        
        #은닉층 선형회귀 값 Z2, 출력값 A2 정의 (모두 행렬로 표시)
        self.Z2 = np.zeros([1,hidden_nodes])
        self.A2 = np.zeros([1,hidden_nodes])

        #입력층 선형회귀 값 Z1, 출력값 A1 정의 (모두 행렬로 표시)
        self.Z1 = np.zeros([1,input_nodes])
        self.A1 = np.zeros([1,input_nodes])

        #학습률 초기화
        self.learning_rate = learning_rate
    def feed_forward(self):
        delta = 1e-7        
        #입력층 선형회귀 값 Z1, 출력값 A1계산
        self.Z1 = self.input_data
        self.A1 = self.input_data
        #은닉층 선형회귀 값 Z2, 출력값 A2계산
        self.Z2 = np.dot(self.A1,self.W2)+self.b2
        self.A2 = sigmoid(self.Z2)
        #출력층 선형회귀 값 Z3, 출력값 A3계산
        self.Z3 = np.dot(self.A2,self.W3)+self.b3
        self.A3 = sigmoid(self.Z3)
        return -np.sum(self.target_data*np.log(self.A3+delta)+(1-self.target_data)*np.log((1-self.A3)+delta))
    def loss_val(self):
        delta = 1e-7        
        #입력층 선형회귀 값 Z1, 출력값 A1계산
        self.Z1 = self.input_data
        self.A1 = self.input_data
        #은닉층 선형회귀 값 Z2, 출력값 A2계산
        self.Z2 = np.dot(self.A1,self.W2)+self.b2
        self.A2 = sigmoid(self.Z2)
        #출력층 선형회귀 값 Z3, 출력값 A3계산
        self.Z3 = np.dot(self.A2,self.W3)+self.b3
        self.A3 = sigmoid(self.Z3)
        return -np.sum(self.target_data*np.log(self.A3+delta)+(1-self.target_data)*np.log((1-self.A3)+delta))        
    def train(self,input_data,target_data):
        self.target_data = target_data
        self.input_data = input_data
        #먼저 feed forward를 통해서 최종 출력값과 이를 바탕으로 현재의 에러 값 계산
        loss_val = self.feed_forward()
        #출력층 loss인 loss_3구함
        loss_3 = (self.A3-self.target_data)*self.A3*(1-self.A3)
        #출력층 가중치 W3, 출력층 바이어스 b3 업데이트
        self.W3 = self.W3 - self.learning_rate*np.dot(self.A2.T,loss_3)
        self.b3 = self.b3 - self.learning_rate*loss_3
        #은닉층 loss인 loss_2구함
        loss_2 = np.dot(loss_3,self.W3.T) * self.A2*(1-self.A2)
        #은닉층 가중치 W2, 은닉층 바이어스 b2 업데이트
        self.W2 = self.W2 - self.learning_rate*np.dot(self.A1.T,loss_2)
        self.b2 = self.b2 - self.learning_rate * loss_2
    def predict(self,input_data):
        Z2 = np.dot(input_data,self.W2)+self.b2
        A2 = sigmoid(Z2)
        Z3 = np.dot(A2,self.W3)+self.b3
        A3 = sigmoid(Z3)
        predicted_num = np.argmax(A3)
        return predicted_num
    def accuracy(self,test_data,ans_data):
        matched_list = []
        not_matched_list = []
        for index in range(len(ans_data)):
            label = int(ans_data[index])
            #one-hot encoding을 위한 데이터 정규화
            data = (test_data[index]/255.0*0.99)+0.01
            #predict를 위해서 vector를 matrix로 변환하여 인수로 넘겨줌
            predicted_num = self.predict(data)
            if label == predicted_num:
                matched_list.append(index)
            else:
                not_matched_list.append(index)
        print("Current Accuracy =",100*(len(matched_list)/(len(ans_data))),"%")
        return matched_list,not_matched_list

## NeuralNetwork 실행부분
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3
epochs = 2
nn = NeuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
start_time = datetime.now()
for i in range(epochs):
    for step in range(len(t_train)):
        #input_data, target_tat normalization
        target_data = np.zeros(output_nodes) + 0.01
        target_data[int(t_train[step])] = 0.99
        input_data = ((x_train[step]/255.0)*0.99)+0.01
        ## for문이 도는동안은 한 행씩 들어감. -> 입력은 2차원 배열로 다시 맞춰줘야 함.
        nn.train(np.array(input_data,ndmin=2),np.array(target_data,ndmin=2))

        if step % 400 ==0:
            print("step =",step,"loss_val",nn.loss_val())
end_time = datetime.now()
print("\nelapsed time =",end_time - start_time)

# 1차원 배열 -> 벡터, 
# 2차원 배열 -> 행렬, 
# 3차원 이상 -> 텐서
# (10,) 은 1차원 벡터
# (1,10)은 1행 10열짜리 행렬

###########MNIST 실습###########
from PIL import Image, ImageOps
# 손글씨 파일 전처리 함수
def preprocess(path):
    # 1 흑백 불러오기
    img = Image.open(path).convert('L')
    # 2 배경·글씨 반전 (필요 시)
    img = ImageOps.invert(img)
    # 3 28×28 리사이즈
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    # 4 numpy 배열로
    arr = np.asarray(img, dtype=np.float32)
    # 5 [0.01,1.0] 정규화
    norm = (arr / 255.0 * 0.99) + 0.01
    # 6 (1,784)로 펼치기
    return norm.reshape(1, -1)

# 각 파일을 따로 전처리
input1 = preprocess('my_number_3.png')   # shape (1,784)
input2 = preprocess('my_number_4.png')   # shape (1,784)
# 모델에 넣기
pred1 = nn.predict(input1)
pred2 = nn.predict(input2)

print("손으로 쓰신 글씨는 숫자",pred1,"으로 보입니다.")
print("손으로 쓰신 글씨는 숫자",pred2,"으로 보입니다.")






