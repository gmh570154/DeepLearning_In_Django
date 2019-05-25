class Vertex:
    #顶点类
    def __init__(self,vid,outList):
        self.vid = vid  # 出边
        self.outList = outList  # 出边指向的顶点id的列表，也可以理解为邻接表
        self.know = False  # 默认为假
        self.dist = float('inf')  # s到该点的距离,默认为无穷大
        self.prev = 0  # 上一个顶点的id，默认为0

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.vid == other.vid
        else:
            return False

    def __hash__(self):
        return hash(self.vid)

#创建顶点对象


v1 = Vertex(1, [2, 4])
v2 = Vertex(2, [4, 5])
v3 = Vertex(3, [1, 6])
v4 = Vertex(4, [3, 5, 6, 7])
v5 = Vertex(5, [7])
v6 = Vertex(6, [])
v7 = Vertex(7, [6])
#存储边的权值
edges = dict()


def add_edge(front, back, value):
    edges[(front, back)] = value


add_edge(1, 2, 2)
add_edge(1, 4, 1)
add_edge(3, 1, 4)
add_edge(4, 3, 2)
add_edge(2, 4, 3)
add_edge(2, 5, 10)
add_edge(4, 5, 2)
add_edge(3, 6, 5)
add_edge(4, 6, 8)
add_edge(4, 7, 4)
add_edge(7, 6, 1)
add_edge(5, 7, 6)
#创建一个长度为8的数组，来存储顶点，0索引元素不存
vlist = [False, v1, v2, v3, v4, v5, v6, v7]
#使用set代替优先队列，选择set主要是因为set有方便的remove方法
vset = set([v1, v2, v3, v4, v5, v6, v7])


def get_unknown_min():  # 此函数则代替优先队列的出队操作
    the_min = 0
    the_index = 0
    j = 0
    for i in range(1, len(vlist)):
        if vlist[i].know is True:  # 已经遍历的节点不需要出队
            continue
        else:
            if j == 0:
                the_min = vlist[i].dist
                the_index = i
            else:
                if vlist[i].dist < the_min:
                    the_min = vlist[i].dist
                    the_index = i
            j += 1
    #此时已经找到了未知的最小的元素是谁
    vset.remove(vlist[the_index]) # 相当于执行出队操作
    return vlist[the_index]  # 返回路径最小的节点


def main():
    #将v1设为顶点
    v1.dist = 0
    while len(vset) != 0:
        v = get_unknown_min()  # 获取最小路径长度的节点，该节点未遍历
        print(v.vid, v.dist, v.outList)  # 拓展最小路径节点的未初始化的节点的路径和父节点
        v.know = True  # 如果节点已经遍历，则不需要重新更新路径和父节点，标记节点已更新
        for w in v.outList:  # w为索引，遍历v节点的所有相邻节点，更新最短路径和父节点
            if vlist[w].know is True:  # 如果节点已经遍历，则不需要重新更新路径和父节点
                continue

            if vlist[w].dist == float('inf'):  #如果节点未初始化，则更新节点的路径和父节点
                vlist[w].dist = v.dist + edges[(v.vid, w)]
                vlist[w].prev = v.vid
            else:
                # v.dist  遍历的节点的路径   edges[(v.vid, w)] v节点到w节点的边长度
                if (v.dist + edges[(v.vid, w)]) < vlist[w].dist:  # 更新最短路径和父节点
                    vlist[w].dist = v.dist + edges[(v.vid, w)]  #更新最短路径
                    vlist[w].prev = v.vid  #更新父节点
                else:  # 原路径长更小，没有必要更新
                    pass


main()

print('v1.prev:', v1.prev, 'v1.dist', v1.dist)
print('v2.prev:', v2.prev, 'v2.dist', v2.dist)
print('v3.prev:', v3.prev, 'v3.dist', v3.dist)
print('v4.prev:', v4.prev, 'v4.dist', v4.dist)
print('v5.prev:', v5.prev, 'v5.dist', v5.dist)
print('v6.prev:', v6.prev, 'v6.dist', v6.dist)
print('v7.prev:', v7.prev, 'v7.dist', v7.dist)
