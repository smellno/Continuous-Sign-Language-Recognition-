# import numpy as np
#
# # 给定概率数组
# probs = np.array([1.69029461e-73, 6.54274443e-72, 2.60074689e-68, 8.75338592e-62,
#  1.20432739e-62, 4.56828976e-57, 6.96412311e-59, 1.08744904e-51,
#  1.32723012e-53, 2.22886466e-53, 4.39138226e-55, 1.20230975e-54,
#  2.77282168e-52, 1.12435353e-47, 5.89442432e-49, 2.66035433e-45,
#  1.37329865e-44, 1.09815584e-34, 3.64276150e-34, 2.01581739e-33,
#  1.43843829e-33, 4.46957397e-33, 8.15819395e-34, 2.52556277e-30,
#  3.15989172e-31])
#
# # 找到最大概率的索引
# max_prob_idx = np.argmax(probs)
# target_value = 4.46957397e-33
# indices = np.where(probs == target_value)[0][0]
#
# print('特定值的索引:', indices)
#
# print("最大概率的索引:", max_prob_idx)
# import torch
#
# # 原始张量
# tensor1 = torch.tensor([28, 1086])
# tensor2 = torch.tensor([20, 1086])
#
# # 找到两个张量在第0维度上的差异
# max_size = max(tensor1[0].item(),tensor2[0].item())
#
#
# if tensor1[0].item() < max_size:
#     tensor1 = torch.unsqueeze(tensor1, dim=0).repeat(max_size - tensor1[0].item(), 1)
# if tensor2[0].item() < max_size:
#     tensor1_expanded = tensor1.unsqueeze(0).expand_as(tensor2)
# # 现在两个张量的形状相同
# print(tensor1)
# print(tensor2)
# import torch
# tensor1 = torch.randn(55, 1024)
#
# # 复制 tensor1 的部分数据以扩充到 (60, 1024)
# expanded_tensor1 = torch.cat((tensor1, tensor1[:5, :]), dim=0)
#
# print(expanded_tensor1.size())  # 输出应为 (60, 1024)
# # tensor1 = torch.randn(55, 1024)
# # tensor2 = torch.randn(60, 1024)
# # max_size = max(tensor1.shape[0], tensor2.shape[0])
# # # 使用unsqueeze在第0维度添加一个维度
# # if tensor2.shape[0] < max_size:
# #     target_feature = tensor2.unsqueeze(0).expand(max_size, -1)
# # if tensor1.shape[0] < max_size:
# #     tensor1.shape[0] = tensor1.shape[0].unsqueeze(0).expand(max_size, -1)
# # # 现在，两个张量的大小都为（60，1024）
# print(tensor1.size()[0])
# # print(tensor2.size())
# indexed_labels = {}
# path = [586, 586, 0, 0, 864, 864, 0, 545, 0, 0, 0, 0, 0, 0, 0, 0, 922, 922, 0, 389, 389, 389, 176, 176, 176, 0, 948, 181, 181, 181, 181, 0, 0, 0]
# for index, label in enumerate(path):
#     if label != 0:
#         # 如果标签不等于0，则将其作为键，将索引添加到对应的列表中
#         if label not in indexed_labels:
#             indexed_labels[label] = [index]
#         else:
#             indexed_labels[label].append(index)
# print(indexed_labels)
# import random
#
# path = [0,0,0,586, 586, 864, 864, 545,922, 922, 389, 389, 389, 176, 176, 176,  948,
#         181, 181, 181, 181]
# indexed_labels = {}  # 初始化字典
#
# # 定义两个变量来追踪零序列的开始和结束
# zero_start = None
#
# for index, label in enumerate(path):
#     if label != 0:
#         if label not in indexed_labels:
#             indexed_labels[label] = []
#         indexed_labels[label].append(index)
#
#         # 处理之前的零序列
#         if zero_start is not None:
#             zero_end = index - 1
#             zero_length = zero_end - zero_start + 1
#
#             if zero_length > 1:
#                 split_point = random.randint(1, zero_length - 1)
#                 left_zeros = list(range(zero_start, zero_start + split_point))
#                 right_zeros = list(range(zero_start + split_point, zero_end + 1))
#             else:
#                 # 零序列只有一个元素时随机分配
#                 left_zeros = []
#                 right_zeros = [zero_start] if random.choice([True, False]) else []
#
#             if zero_start == 0:
#                 indexed_labels[label].extend(right_zeros)
#             else:
#                 prev_label = path[zero_start - 1]
#                 print('prev_label',prev_label)
#
#                 indexed_labels[prev_label].extend(left_zeros)
#                 indexed_labels[label].extend(right_zeros)
#
#             zero_start = None  # 重置零序列开始点
#     else:
#         if zero_start is None:
#             zero_start = index
#
# # 处理最后一个零序列（如果存在）
# if zero_start is not None:
#     zero_end = len(path) - 1
#     left_zeros = list(range(zero_start, zero_end + 1))
#     prev_label = path[zero_start - 1]
#     indexed_labels[prev_label].extend(left_zeros)
# for key in indexed_labels:
#     indexed_labels[key].sort()
#
# print(indexed_labels)
# print(indexed_labels)
import numpy as np
import torch
import random

# # 假设 feature 是你的输入特征，形状为 (n, 1024)
# # feature = np.random.rand(35, 1024) # 仅作为示例
# feature = torch.randn(2,55,3,224,224)
# for i in range(feature.size()[0]):
#     feature1 = feature[i, :, :]
#     feature1 = feature1[1:10, :]
#
#     print(feature1.size())
#
# # 使用 array_split 分组，这里的参数 3 表示我们希望每组有3个（最后一组可能少于3个）
# grouped_feature = np.array_split(feature, range(3, feature.size()[0], 3), axis=0)
# for i in grouped_feature:
#     print(i.size())
# # grouped_feature 现在是一个列表，其中的每个元素都是一个子数组
import random
# for i in range(6):
#     # 生成随机数的个数，范围为 2 到 15
#     num_samples = random.randint(2, 15)
#
#     # 生成指定范围内的随机数列表
#     random_numbers = random.sample(range(1, 1087), num_samples)
#
#     print(random_numbers)
# def map_output_to_original(output_index, kernel_sizes= ['K5', "P2", 'K5', "P2"]):
#     min_original_index = output_index
#     max_original_index = output_index
#
#     for ks in reversed(kernel_sizes):
#         if ks[0] == 'K':  # 卷积层
#             min_original_index = min_original_index * 1  # 卷积层不改变时间步的起始位置，但增加了范围
#             max_original_index += (int(ks[1]) - 1)  # 每个输出时间步对应于原始序列中的k个连续时间步
#         elif ks[0] == 'P':  # 池化层
#             min_original_index = min_original_index * int(ks[1])  # 池化层通过其窗口大小扩展了每个时间步的影响范围
#             max_original_index = (max_original_index + 1) * int(ks[1]) - 1  # 考虑到池化操作的下采样，扩展最大索引
#
#     return min_original_index, max_original_index
#
# output_range_start = 9
# output_range_end = 9
#
# # 映射起始和结束索引到原始序列中的时间步范围
# original_range_start, _ = map_output_to_original(output_range_start)
# _, original_range_end = map_output_to_original(output_range_end)
#
# print(original_range_start, original_range_end)
# def map_output_to_original(output_index, kernel_sizes=['K5', "P2", 'K5', "P2"]):
#     min_original_index = output_index
#     max_original_index = output_index
#
#     for ks in reversed(kernel_sizes):
#         if ks[0] == 'K':  # 卷积层
#             min_original_index = min_original_index * 1  # 卷积层不改变时间步的起始位置，但增加了范围
#             max_original_index += (int(ks[1]) - 1)  # 每个输出时间步对应于原始序列中的k个连续时间步
#         elif ks[0] == 'P':  # 池化层
#             min_original_index = min_original_index * int(ks[1])  # 池化层通过其窗口大小扩展了每个时间步的影响范围
#             max_original_index = (max_original_index + 1) * int(ks[1]) - 1  # 考虑到池化操作的下采样，扩展最大索引
#
#     return min_original_index, max_original_index
#
# output_range_start = 10
# output_range_end = 11
#
# # 映射起始和结束索引到原始序列中的时间步范围
# original_range_start, original_range_end = map_output_to_original(output_range_start)
# _, original_range_end = map_output_to_original(output_range_end)
#
# print(original_range_start, original_range_end)
#
# labell = ['a','c']
# for batch_idx, i in enumerate(labell):
#     # 对每个i执行某些操作
#     # 例如，打印每个批次的索引和对应的标签
#     print(f"Batch {batch_idx}: Label {i}")
# indices = [10,11,34]
# def check_continuous(indices):
#     for i in range(1, len(indices)):
#         if indices[i] - indices[i-1] != 1:
#             return False
#     return True
#
# if not check_continuous(indices):
#     print(f"索引列表 {indices} 不连续")
import random

# 假设你有一个列表
my_list = [[1, 2],[3, 4],[5],[1, 2]]

# 使用 random.choice 函数来随机选择一个值
random_value = random.choice(my_list)

# 打印结果
print("随机选择的值：", random_value)




