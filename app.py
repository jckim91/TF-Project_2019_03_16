import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import io
from flask import Flask, render_template

from linear_regression import LinearRegression
from gradient_descent import GradientDescent
from k_mean import Kmean

app = Flask(__name__)

@app.route('/')
def index():
    t = LinearRegression()
    g = GradientDescent()
    k = Kmean()

    name = t.execute()
    name = g.execute()
    return render_template('index.html', name = name)

if __name__ == '__main__':
    app.run()

'''
def hello_world():
    rng = np.random
    # parameter
    learning_rate = 0.01  # 학습률
    training_epoch = 1000  # 학습횟수
    display_step = 50

    # Training Data
    train_X = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                          7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
    train_Y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                          2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
    n_samples = train_X.shape[0]
    # tf Graph input

    X = tf.placeholder("float")
    Y = tf.placeholder("float")

    # 모델에 가중치 주기
    W = tf.Variable(rng.randn(), name='weight')
    b = tf.Variable(rng.randn(), name='bias')

    # 모델 구축
    pred = tf.add(tf.multiply(X, W), b)

    # 비용함수
    cost = tf.reduce_sum(tf.pow(pred - Y, 2) / (2 * n_samples))  # pow 제곱

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    # 학습 시작
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(training_epoch):
            for (x, y) in zip(train_X, train_Y):  # zip 은 결합
                sess.run(optimizer, feed_dict={X: x, Y: y})  # feed_dict 은 주입되는 dictionary
        print('옵티마이저 최적화 종료')
        training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})

        plt.plot(train_X, train_Y, 'ro', label='오리지널 데이터')
        plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='최적화된 그래프')
        plt.legend()
        img = io.BytesIO()
        plt.savefig('static/img/result.svg')
    return render_template('index.html', name='LINEAR REGRESSION')


if __name__ == '__main__':
    app.run()
'''

'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import io

from flask import Flask,render_template

app = Flask(__name__)


@app.route('/')
def hello_world():
    rng = np.random
    # parameter
    learning_rate = 0.01
    training_epoch = 1000
    display_step = 50

    # training data
    train_X = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                          7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
    train_Y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                          2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
    n_samples = train_X.shape[0]

    # tf Graph input

    X = tf.placeholder("float")
    Y = tf.placeholder("float")

    # 모델에 가중치 주기
    W = tf.Variable(rng.randn(), name='Weight')
    b = tf.Variable(rng.randn(), name='bias')

    # 모델 구축
    pred = tf.add(tf.multiply(X, W), b)

    # 비용 함수
    cost = tf.reduce_sum(tf.pow(pred - Y, 2) / (2 * n_samples))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    # 학습 시작
    with tf.Session as sess:
        sess.run(init)
        for epoch in range(training_epoch):
            for (x, y) in zip(train_X, train_Y):  # zip은 결합
                sess.run(optimizer, feed_dict={X: x, Y: y})  # feed_dict는 주입되는 dictionary
        print('옵티마이저 최적화 종료')
        training_cost = sess.run(cost, feed_dict={X: x, Y: y})

        plt.plot(train_X, train_Y, 'ro', label='오리지널 데이터')
        plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='최적화된 그래프')
        plt.legend()
        # plt.show()
        img = io.BytesIO()
        plt.savefig('static/img/result.svg')
    return render_template('index.html', name='Linear Regression')


if __name__ == '__main__':
    app.run()
'''