"""
A* grid planning has not been improved
author: Atsushi Sakai(@Atsushi_twi)
        Nikos Kanargias (nkana@tee.gr)
Modified by: DWDROME
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from math import factorial


show_animation = True # 是否显示动画

class AStarPlanner:

    def __init__(self, ox, oy, resolution, rr):
        self.resolution = resolution   # 地图分辨率 [m]
        self.rr = rr  # robot radius [m]
        self.calc_obstacle_map(ox,oy)
        self.motion = self.get_motion_model()  # 获取运动模型

    # 创建节点类，包含x、y，g(x)（cost代价），parent_index
    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x
            self.y = y
            self.cost = cost  # g(n)
            self.parent_index = parent_index  
        def __str__(self):    
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        """
        1. 定义nstart,ngoal为节点
        2. 创建open_set和closed_set
        3. 将nstart添加到open_set
        4. 遍历过程:
            a. 计算当前节点的代价f(x)，同时得到  [g(n)指当前节点到起点的代价（实际代价），h(n)指当前节点到目标点的代价（欧式距离）]
            b. 如果当前节点等于ngoal，返回路径
            c. 删除已经遍历过的open_set[c_id]，并将c_id对应的node（current）加入到closed_set
            d. 对于当前节点的八邻域，计算他们对应的g(n)
        5. 返回最终路径pathx, pathy

        g(n)为open_set[o]，其中open_set[nstart] = 0,之后八邻域open_set[n_id]=current.cost + move_cost的方式确定
        h(n)为当前节点到目标点的欧式距离
        按照原作者的阐述，大概是优化了启发式函数h(n)（原来基于曼哈顿距离），使得搜索更高效。
        """
        # 1. 定义nstart,ngoal为节点
        nstart = self.Node(self.calc_xyindex(sx,self.minx),
                            self.calc_xyindex(sy,self.miny),
                            0.0,-1)
        ngoal = self.Node(self.calc_xyindex(gx,self.minx),
                            self.calc_xyindex(gy,self.miny),
                            0.0,-1)
        # 2. 创建open_set和closed_set
        open_set_A,closed_set_A = dict(), dict()
        open_set_B,closed_set_B = dict(), dict()

        # 3. 将nstart添加到open_set
        open_set_A[self.calc_grid_index(nstart)] = nstart 
        open_set_B[self.calc_grid_index(ngoal)] = ngoal 

        current_A = nstart
        current_B = ngoal
        meet_point_A, meet_point_B = None, None

        # 4. 遍历过程:
        while 1:
            if len(open_set_A) == 0:
                print("Open_set_A is empty..")
                break
            if len(open_set_B) == 0:
                print("Open_set_B is empty..")
                break
            # a. 计算当前节点的代价f(x)，同时得到  [g(n)指当前节点到起点的代价（实际代价），h(n)指当前节点到目标点的代价（欧式距离）]
            # f(x) = g(n) + h(n)
            c_id_A = min(open_set_A, key=lambda o: open_set_A[o].cost + self.calc_heuristic(open_set_A[o],current_B))
            current_A = open_set_A[c_id_A]
            c_id_B = min(open_set_B, key=lambda o: open_set_B[o].cost + self.calc_heuristic(open_set_B[o],current_A))
            current_B = open_set_B[c_id_B]


            # 将当前节点显示出来
            if show_animation:
                plt.plot(self.calc_grid_position(current_A.x, self.minx),
                        self.calc_grid_position(current_A.y, self.miny),
                        "xc")   # 青色x 搜索点
                plt.plot(self.calc_grid_position(current_B.x, self.minx),
                        self.calc_grid_position(current_B.y, self.miny),
                        "xc")   # 青色x 搜索点
                # 按esc退出
                plt.gcf().canvas.mpl_connect('key_release_event',
                            lambda event: exit(0) if getattr(event, 'key', None) == 'escape' else None)
                if len(closed_set_A.keys()) % 10 == 0:
                    plt.pause(0.001)
                if len(closed_set_B.keys()) % 10 == 0:
                    plt.pause(0.001)
            # b. 如果当前节点等于ngoal，返回路径
            # if current.x == ngoal.x and current.y == ngoal.y:
            #     ngoal.parent_index = current.parent_index
            #     ngoal.cost = current.cost
            #     print("Find goal!!")
            #     print("ngoal_parent_index:", ngoal.parent_index)
            #     print("ngoal_cost:", ngoal.cost)
            #     break

            if current_A.x == current_B.x and current_A.y == current_B.y:
                meet_point_A = current_A
                meet_point_B = current_B
                break

            # c. 删除已经遍历过的open_set[c_id]，并将c_id对应的node（current）加入到closed_set
            del open_set_A[c_id_A]
            closed_set_A[c_id_A] = current_A
            del open_set_B[c_id_B]
            closed_set_B[c_id_B] = current_B

            # d. 对于当前节点的八邻域，计算他们对应的g(n)
            # 基于movetion model 做栅格扩展，也就是搜索方式，可进行改进，如使用双向广搜、JPS等   
            for move_x, move_y, move_cost in self.motion:
                node = [self.Node(current_A.x + move_x,
                                    current_A.y + move_y,
                                    current_A.cost + move_cost,
                                    c_id_A),
                        self.Node(current_B.x + move_x,
                                    current_B.y + move_y,
                                    current_B.cost + move_cost,
                                    c_id_B)
                        ]
                n_id = [self.calc_grid_index(node[0]),
                        self.calc_grid_index(node[1])
                        ]

                # 如果节点不可通过
                if not self.verify_node(node[0]):
                    continue
                # 如果节点已经在closed_set中
                if n_id[0] in closed_set_A:
                    continue
                # 如果节点不在open_set中
                if n_id[0] not in open_set_A:
                    open_set_A[n_id[0]] = node[0]   
                else:
                    if open_set_A[n_id[0]].cost > node[0].cost:
                        open_set_A[n_id[0]] = node[0]    # 更新最短路线     

                # 如果节点不可通过
                if not self.verify_node(node[1]):
                    continue
                # 如果节点已经在closed_set中
                if n_id[1] in closed_set_B:
                    continue
                # 如果节点不在open_set中
                if n_id[1] not in open_set_B:
                    open_set_B[n_id[1]] = node[1]   
                else:
                    if open_set_B[n_id[1]].cost > node[1].cost:
                        open_set_B[n_id[1]] = node[1]    # 更新最短路线   
        # 5. 返回最终路径pathx, pathy
        pathx , pathy = self.calc_final_AB_way(meet_point_A, meet_point_B, closed_set_A, closed_set_B)
        return pathx, pathy
    
    def calc_final_AB_way(self, ment_point_A, ment_point_B,  closed_set_A, closed_set_B):
        pathx_A , pathy_A = self.calc_final_path(ment_point_A,closed_set_A)
        pathx_B , pathy_B = self.calc_final_path(ment_point_B,closed_set_B)

        pathx_A.reverse()
        pathy_A.reverse()

        pathx = pathx_A + pathx_B
        pathy = pathy_A + pathy_B

        return pathx, pathy

    def calc_final_path(self, ngoal, closed_set):
        pathx, pathy = [self.calc_grid_position(ngoal.x, self.minx)], [
                        self.calc_grid_position(ngoal.y, self.miny)]
        parent_index = ngoal.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            pathx.append(self.calc_grid_position(n.x, self.minx))
            pathy.append(self.calc_grid_position(n.y, self.miny))
            parent_index = n.parent_index
        return pathx, pathy

    
    def calc_obstacle_map(self,ox,oy):
        self.minx = round(min(ox))    # 地图中的临界值 -10
        self.miny = round(min(oy))    # -10
        self.maxx = round(max(ox))    # 60 
        self.maxy = round(max(oy))    # 60
        print("minx:", self.minx)
        print("miny:", self.miny)
        print("maxx:", self.maxx)
        print("maxy:", self.maxy)
    
        self.xwidth = round((self.maxx - self.minx) / self.resolution)   # 35
        self.ywidth = round((self.maxy - self.miny) / self.resolution)   # 35
        print("xwidth:", self.xwidth)
        print("ywidth:", self.ywidth)

        self.obmap = [[False for _ in range(self.ywidth)] for _ in range(self.xwidth)]

        for idx_x in range(int(self.xwidth)):
            x = self.calc_grid_position(idx_x, self.minx)
            for idx_y in range(int(self.ywidth)):
                y = self.calc_grid_position(idx_y, self.miny)
                for iox, ioy in zip(ox, oy): # 将ox，oy打包成元组，返回列表，并遍历
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:  # 代价小于车辆半径，可征程通过，不会穿过障碍物
                        self.obmap[idx_x][idx_y] = True
                        break

    # 得到全局地图中的具体坐标
    def calc_grid_position(self, index, min_value):
        return  round(index * self.resolution + min_value)
    
    # 计算栅格地图节点的index: 传入某节点
    def calc_grid_index(self, node):
        return (node.y - self.miny)* self.xwidth + (node.x - self.minx)
    
    # 启发函数
    @staticmethod
    def calc_heuristic(node1,node2):
        # # 曼哈顿距离
        # h = np.abs(node1.x - node2.x) + np.abs(node1.y - node2.y)

        # # 欧几里得距离
        # h = math.hypot(node1.x - node2.x, node1.y - node2.y )

        # #切比雪夫距离
        # dx = np.abs(node1.x - node2.x)
        # dy = np.abs(node1.y - node2.y)
        # min_xy = min(dx, dy)
        # h = dx + dy + (math.sqrt(2) - 2)* min_xy
        # return h

        # 对欧氏距离的改进
        dx = np.abs(node1.x - node2.x)  # Diagnol distance
        dy = np.abs(node1.y - node2.y)
        min_xy = min(dx, dy)
        d = dx + dy + (math.sqrt(2) - 2) * min_xy
        if d > 12:
            w = 2.8
        else:
            w = 0.8
        h = w * d

        return h
        return h
    
    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [
                    [1, 0, 1],
                    [0, 1, 1],
                    [-1, 0, 1],
                    [0, -1, 1],
                    [1, 1, math.sqrt(2)],
                    [1, -1, math.sqrt(2)],
                    [-1, 1, math.sqrt(2)],
                    [-1, -1, math.sqrt(2)]    
                 ]
        
        return motion
    
    def verify_node(self, node):
        posx = self.calc_grid_position(node.x, self.minx)
        posy = self.calc_grid_position(node.y, self.miny)

        if posx < self.minx:
            return False
        elif posy < self.miny:
            return False
        elif posx >= self.maxx:
            return False
        elif posy >= self.maxy:
            return False

        if self.obmap[int(node.x)][int(node.y)]:
            return False

        return True
    
    def calc_xyindex(self, position, min_pos):
        return round((position - min_pos) / self.resolution)


def comb(n, k):
    return factorial(n) // (factorial(k) * factorial(n-k))
def get_bezier_curve(points):
    n = len(points) - 1
    return lambda t: sum(comb(n, i)*t**i * (1-t)**(n-i)*points[i] for i in range(n+1))
def evaluate_bezier(points, total):
    bezier = get_bezier_curve(points)
    new_points = np.array([bezier(t) for t in np.linspace(0, 1, total)]) # 0~1之间生成total个linear sequence
    return new_points[:,0], new_points[:,1] 

def main():
    print( __file__ + ' Start!!')

    # start and goal position  [m]
    sx = -5.0
    sy = -5.0
    gx = 50 
    gy = 50
    grid_size = 2.0
    robot_radius = 1.0

        # obstacle positions
    ox, oy = [],[]
    for i in range(-10, 60): 
        ox.append(i)
        oy.append(-10)      # y坐标-10的一行-10~60的坐标添加到列表并显示为黑色障碍物
    for i in range(-10, 60):
        ox.append(i)
        oy.append(60)       # y坐标60的一行-10~60的坐标添加到列表并显示为黑色障碍物
    for i in range(-10, 61):
        ox.append(-10)
        oy.append(i)        # x坐标-10的一列-10~61的坐标添加到列表并显示为黑色障碍物
    for i in range(-10, 61):
        ox.append(60)
        oy.append(i)        # x坐标60的一列-10~61的坐标添加到列表并显示为黑色障碍物
    for i in range(-10, 40):
        ox.append(20)
        oy.append(i)        # x坐标20的一列-10~40的坐标添加到列表并显示为黑色障碍物
    for i in range(0, 40):
        ox.append(40)
        oy.append(60 - i)   # x坐标40的一列20~60的坐标添加到列表并显示为黑色障碍物

    if show_animation:
        plt.plot(ox,oy,".k") # 黑色.       障碍物
        plt.plot(sx,sy,"og") # 绿色圆圈    开始坐标
        plt.plot(gx,gy,"xb") # 蓝色x       目标点
        plt.grid(True)
        plt.axis('equal')  # 保持栅格的横纵坐标刻度一致

    a_star = AStarPlanner(ox,oy,grid_size,robot_radius)
    pathx,pathy = a_star.planning(sx, sy, gx ,gy)

    if show_animation:
        plt.plot(pathx, pathy, "-r")  # 红色直线 最终路径
        plt.show()
        plt.pause(0.001)   # 动态显示

    # 贝塞尔曲线的控制点，为了方便更改，可根据出图效果调整
    # points = np.array([[pathx[0], pathy[0]], [pathx[1], pathy[1]], [pathx[2], pathy[2]], 
    #                     [pathx[3], pathy[3]],[pathx[4], pathy[4]],[pathx[5], pathy[5]],
    #                     [pathx[6], pathy[6]],[pathx[7], pathy[7]],[pathx[8], pathy[8]],
    #                     [pathx[9], pathy[9]],[pathx[10], pathy[10]],[pathx[11], pathy[11]],
    #                     [pathx[12], pathy[12]],[pathx[13], pathy[13]],[pathx[14], pathy[14]],
    #                     [pathx[15], pathy[15]]])
    # points1 = np.array([[pathx[15], pathy[15]], [pathx[16], pathy[16]],[pathx[17], pathy[17]]])                    
    # points2 = np.array([[pathx[17], pathy[17]],[pathx[18], pathy[18]], [pathx[19], pathy[19]], [pathx[20], pathy[20]], 
    #                     [pathx[21], pathy[21]],[pathx[22], pathy[22]],[pathx[23], pathy[23]],
    #                     [pathx[24], pathy[24]],[pathx[25], pathy[25]],[pathx[26], pathy[26]],
    #                     [pathx[27], pathy[27]]])
    # points3 = np.array([[pathx[27], pathy[27]],[pathx[28], pathy[28]],[pathx[29], pathy[29]]])  
    # points4 = np.array([[pathx[29], pathy[29]],[pathx[30], pathy[30]], [pathx[31], pathy[31]], [pathx[32], pathy[32]], 
    #                     [pathx[33], pathy[33]],[pathx[34], pathy[34]],[pathx[35], pathy[35]],
    #                     [pathx[36], pathy[36]],[pathx[37], pathy[37]],[pathx[38], pathy[38]],
    #                     [pathx[39], pathy[39]],[pathx[40], pathy[40]],[pathx[41], pathy[41]],
    #                     [pathx[42], pathy[42]],[pathx[43], pathy[43]],[pathx[44], pathy[44]],
    #                     [pathx[45], pathy[45]],[pathx[46], pathy[46]],[pathx[47], pathy[47]],
    #                     [pathx[48], pathy[48]],[pathx[49], pathy[49]],[pathx[50], pathy[50]],
    #                     [pathx[51], pathy[51]],[pathx[52], pathy[52]]])
    
    # bx, by = evaluate_bezier(points, 50)
    # bx1, by1 = evaluate_bezier(points1, 50)
    # bx2, by2 = evaluate_bezier(points2, 50)
    # bx3, by3 = evaluate_bezier(points3, 50)
    # bx4, by4 = evaluate_bezier(points4, 50)

    # if show_animation:
    #     plt.plot(pathx, pathy, "-r")  # 红色线 最终路径
    #     plt.plot(bx, by, 'b-')        # 蓝色线 贝塞尔曲线
    #     plt.plot(bx1, by1, 'b-')
    #     plt.plot(bx2, by2, 'b-')
    #     plt.plot(bx3, by3, 'b-')
    #     plt.plot(bx4, by4, 'b-')

    #     plt.show()
    #     plt.pause(0.001)   # 动态显示

if __name__ == '__main__':
    main()
