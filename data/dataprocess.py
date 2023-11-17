import numpy as np
data = np.load('./data/PEMS03/PEMS03.npz')
data = data['data']
print(data.shape)
random_num = np.random.randint(0, data.shape[1])
new_data=data[:5760, random_num, 0]
np.savez(f'./data/PEMS03/new_pems03_num{random_num}.npz',data=new_data)
# # (1464, 45, 1) ——> 45个BRT站点连续61days共计1464个小时的客流量数据
# station_list = ['第一码头', '开禾路口', '思北', '斗西路',
#                 '二市', '文灶', '金榜公园', '火车站',
#                 '莲坂', '龙山桥', '卧龙晓城', '东芳山庄',
#                 '洪文', '前埔枢纽站', '蔡塘', '金山', '市政务服务中心',
#                 '双十中学', '县后', '高崎机场', 'T4候机楼', '嘉庚体育馆',
#                 '诚毅学院', '华侨大学', '大学城', '产业研究院', '中科院',
#                 '东宅', '田厝', '厦门北站', '凤林', '东安', '后田', '东亭',
#                 '美峰', '蔡店', '潘涂', '滨海新城西柯枢纽', '官浔', '轻工食品园',
#                 '四口圳', '工业集中区', '第三医院', '城南', '同安枢纽']
# # 随机选择一个站点(0-44之间的一个站点)
# # random_station = np.random.randint(0, data.shape[1])
# for i in range(0, 45):
#     station_data = data[:, i, 0]
#     # 将数据展平为一维数组
#     station_data = station_data.ravel()
#     station_name = station_list[i]
#     # 保存数据集为npz文件
#     np.savez(f'./data/BRT/{station_name}_brtdata.npz',data=station_data)