import torch
import torch.nn as nn
loss = []
label_features = {}

tensor = torch.randn(30, 1024)
labeld = [19, 20, 1, 5]
labels = [0, 0, 19, 19, 19, 19, 0, 0, 0, 0, 0, 20, 20, 20, 20, 0, 1, 0, 20, 20, 20, 20, 0, 5, 5, 5, 5, 0, 0, 0]

indexed_labels = {}

# 遍历原始标签列表
for index, label in enumerate(labels):
    if label != 0:
        # 如果标签不等于0，则将其作为键，将索引添加到对应的列表中
        if label not in indexed_labels:
            indexed_labels[label] = [index]
        else:
            indexed_labels[label].append(index)
# 遍历字典中的键和值
for key in indexed_labels.keys():
    values = indexed_labels[key]
    min_index = min(values)  # 获取值列表中的最小索引
    max_index = max(values)  # 获取值列表中的最大索引
    if min_index - 1 >= 0:
        indexed_labels[key] = [min_index - 1] + values
        # 检查最大索引是否在列表范围内，如果在范围内则添加到值列表的后面
    if max_index + 1 < len(labels):
        indexed_labels[key] = indexed_labels[key] + [max_index + 1]
print(indexed_labels)
for i in labeld:
    indices = indexed_labels[i]
    seq_len = len(indices)
    if max(indices) < tensor.shape[0]:
        sliced_tensor = tensor[indices, :]
        target = torch.tensor([i])
        criterion = nn.CrossEntropyLoss()
        losse = criterion(sliced_tensor, target)
        loss.append(losse)
        print('sliced_tensor.shape',sliced_tensor.shape)

        if target in label_features:
            target_feature = label_features[target]

            label_features[target] = sliced_tensor.clone().detach().requires_grad_(True)  # 更新特征

        else:
            label_features[target] = sliced_tensor.clone().detach().requires_grad_(True)  # 将特征存储在CPU上，并允许梯度计算

    else:
        print("索引超出张量的范围")
stacked_loss = torch.stack(loss)
total_loss = torch.sum(stacked_loss)
print(total_loss)  # 输出总和



import numpy as np
from six.moves import xrange


import numpy as np
import editDistance as ed
import heapq as hq
from six.moves import xrange


    # 初始条件：T=0时，只能为 blank 或 seq[0]
    alphas[0, 0] = params[blank, 0]
    alphas[1, 0] = params[seq[0], 0]
    # T=0， alpha[:, 0] 其他的全部为 0


    c = np.sum(alphas[:, 0])
    alphas[:, 0] = alphas[:, 0] / c  # 这里 T=0 时刻所有可能节点的概率要归一化

    llForward = np.log(c)  # 转换为log域

    for t in xrange(1, T):
        # 第一个循环： 计算每个时刻所有可能节点的概率和
        start = max(0, L - 2 * (T - t))  # 对于时刻 t, 其可能的节点.与公式2一致。
        end = min(2 * t + 2, L)  # 对于时刻 t，最大节点范围不可能超过 2t+2
        for s in xrange(start, L):
            l = (s - 1) / 2
            # blank，节点s在偶数位置，意味着s为blank
            if s % 2 == 0:
                if s == 0:  # 初始位置，单独讨论
                    alphas[s, t] = alphas[s, t - 1] * params[blank, t]
                else:
                    alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1]) * params[blank, t]
            # s为奇数，非空
            # l = (s-1/2) 就是 s 所对应的 lable 中的字符。
            # ((s-2)-1)/2 = (s-1)/2-1 = l-1 就是 s-2 对应的lable中的字符
            elif s == 1 or seq[l] == seq[l - 1]:
                alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1]) * params[seq[l], t]
            else:
                alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1] + alphas[s - 2, t - 1]) \
                               * params[seq[l], t]

        # normalize at current time (prevent underflow)
        c = np.sum(alphas[start:end, t])
        alphas[start:end, t] = alphas[start:end, t] / c
        llForward += np.log(c)
    return llForward


def forward(y, labels):
    T, C = y.shape #T: timestep
    L = len(labels)
    alpha = np.zeros([T,L])

    alphas[0, 0] = params[labels[0], 0]
    alphas[1, 0] = params[labels[1], 0]
    for t in range(1,T):
        for i in range(L):
            s = labels[i]
            a = alpha[t-1,i]
            if i-1 >= 0:
                a += alpha[t-1,i-1]
            if i-2 >= 0 and s != 0 and s != labels[i-2]:
                a += alpha[t-1,i-2]
            alpha[t,i] = a*y[t,s]
    return alpha

def ctc_decodedd(params, labels):
    params = params - np.max(params, axis=0)
    params = np.exp(params)
    params = params / np.sum(params, axis=0)

    # Initialize alphas and forward pass

    # 初始条件：T=0时，只能为 blank 或 seq[0]
    alphas[0, 0] = params[blank, 0]
    alphas[1, 0] = params[seq[0], 0]
    print('labels',labels)
    new_length = len(labels) * 2 + 1  # 两个标签之间加一个0，所以是原始长度的两倍，再加1个开头的0
    # 创建一个全0的新张量
    extended = torch.zeros(new_length, dtype=torch.int32)
    # 计算需要填充labels的索引位置
    indices = torch.arange(1, new_length, 2)
    # 在指定位置填充原始标签
    extended[indices] = labels.to(torch.int32)
    extended_list = extended.tolist()
    print('extended_list',extended_list)
    alpha = forward(probabilities, extended_list)
    print('alpha.shape',alpha.shape)
    T,C = alpha.shape
    # 回溯找到最优路径
    path = []
    path1 = []
    max_prob_idx = np.argmax(alpha[-1][-2:])  # 在最后两个字符中找到概率最大的索引
    path1.append(max_prob_idx)
    path.append(labels[max_prob_idx])
    for t in range(T - 1, 0, -1):
        if path1[-1] % 2 == 0:  # 如果当前标签是非空白字符
            if path1[-1] - 2 >= 1:
                max_prob_idx = np.argmax([alpha[t - 1][max_prob_idx - 1], alpha[t - 1][max_prob_idx], alpha[t - 1][max_prob_idx - 2]])
            if path1[-1] - 1 == 0:
                max_prob_idx = np.argmax([alpha[t - 1][max_prob_idx - 1], alpha[t - 1][max_prob_idx]])

        else:  # 如果当前标签是空白字符
            if path1[-1] - 1 < 0:
                max_prob_idx = np.argmax([alpha[t - 1][max_prob_idx]])
            else:
                max_prob_idx = np.argmax([alpha[t - 1][max_prob_idx - 1], alpha[t - 1][max_prob_idx]])

        path.append(labels[max_prob_idx])
        path1.append(max_prob_idx)

    path.reverse()  # 将路径反转为正向顺序
    return path