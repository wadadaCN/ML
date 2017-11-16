import numpy as np

def fit(x, y, lr=1, epoch=1000):
    # 将输入的 x、y 转为 numpy 数组
    x, y = np.asarray(x, np.float32), np.asarray(y, np.float32)
    w = np.zeros(x.shape[1])
    b = 0
    for _ in range(epoch):
        y_pred = x.dot(w) + b  # 计算 w·x[i]+b，此处计算所有点的损失函数
        idx = np.argmax(np.maximum(0, -y_pred * y))  # maximum返回array各列最大值，argmax返回array最大值索引。凡是正数的点都将修改w，b
        if y[idx] * y_pred[idx] > 0:
            return w, b # 否则，让参数沿着负梯度方向走一步
        delta = lr * y[idx]
        w += delta * x[idx]
        b += delta

def fit2(x, y, lr=1, epoch=1000):
    # 将输入的 x、y 转为 numpy 数组
    x, y = np.asarray(x, np.float32), np.asarray(y, np.float32)
    w = np.zeros(x.shape[1])
    b = 0
    Gram = np.zeros((x.shape[0], x.shape[0]))
    for i in range(x.shape[0]): # Gram矩阵赋值
        for j in range(x.shape[0]):
            if i <= j:
                Gram[i][j] = x[i].dot(x[j])
            else:
                Gram[i][j] = Gram[j][i]
    alpha = np.zeros(x.shape[0])
    for _ in range(epoch):
        y_pred = []
        for i in range(x.shape[0]):
            alSum = 0
            for j in range(x.shape[0]):
                alSum += alpha[j] * y[j] * Gram[i][j]
            y_pred.append(alSum+b)
        y_pred = np.asarray(y_pred, np.float32)
        idx = np.argmax(np.maximum(0, -y_pred * y))
        if y[idx] * y_pred[idx] > 0:
            for i in range(x.shape[0]):
                w += alpha[i] * x[i] * y[i] # w等于x的alpha加权求和
            return w, b
        alpha[idx] += lr
        b += y[idx]

def test(x):
    xList = [[3, 3], [4, 3], [1, 1]]
    yList = [1, 1, -1]
    w, b = fit(x = xList,y = yList)
    # w, b = fit2(x = xList, y = yList)
    y = x.dot(w) + b
    return 1 if y > 0 else -1

x = [2,0]
print(test(np.asarray(x, np.float32)))
