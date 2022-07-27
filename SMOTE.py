import numpy as np
import random
from sklearn.neighbors import NearestNeighbors

class Smote(object):
    def __init__(self, N=50, k=5, r=2):
        # 初始化self.N, self.k, self.r, self.newindex
        self.N = N
        self.k = k
        # self.r是距离决定因子
        self.r = r
        # self.newindex count creat cases
        self.newindex = 0
    # 构建训练函数
    def fit(self, dataset):
        # 初始化self.samples, self.T, self.numattrs
        self.data = dataset
        # self.T是少数类样本个数，self.numattrs是少数类样本的特征个数
        self.total_num, self.num_feat = self.data.shape

        # 查看N%是否小于100%
        if(self.N < 100):
            # 如果是，随机抽取N*T/100个样本，作为新的少数类样本
            np.random.shuffle(self.data)
            self.total_num = int(self.N * self.total_num/100)
            self.data = self.data[0:self.total_num,:]
            # N%变成100%
            self.N = 100

        # 查看从T是否不大于近邻数k
        if(self.total_num <= self.k):
            # 若是，k更新为T-1
            self.k = self.total_num - 1

        # 令N是100的倍数
        N = int(self.N/100)
        # 创建保存合成样本的数组
        self.synthetic = np.zeros((self.total_num * N, self.num_feat))

        # 调用并设置k近邻函数
        neighbors = NearestNeighbors(n_neighbors=self.k+1, 
                                     algorithm='ball_tree', 
                                     p=self.r).fit(self.data)

        # 对所有输入样本做循环
        for i in range(len(self.data)):
            # 调用kneighbors方法搜索k近邻
            nnarray = neighbors.kneighbors(self.data[i].reshape((1,-1)),
                                           return_distance=False)[0][1:]

            # 把N,i,nnarray输入样本合成函数self.__populate
            self.__populate(N, i, nnarray)

        # 最后返回合成样本self.synthetic
        return self.synthetic
    
    # 构建合成样本函数
    def __populate(self, N, i, nnarray):
        # 按照倍数N做循环
        for j in range(N):
            # attrs用于保存合成样本的特征
            attrs = []
            # 随机抽取1～k之间的一个整数，即选择k近邻中的一个样本用于合成数据
            nn = random.randint(0, self.k-1)
            
            # 计算差值
            diff = self.data[nnarray[nn]] - self.data[i]
            # 随机生成一个0～1之间的数
            gap = random.uniform(0,1)
            # 合成的新样本放入数组self.synthetic
            self.synthetic[self.newindex] = self.data[i] + gap*diff

            # self.newindex加1， 表示已合成的样本又多了1个
            self.newindex += 1

