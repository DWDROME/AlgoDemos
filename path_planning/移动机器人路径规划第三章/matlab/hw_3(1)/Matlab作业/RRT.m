%***************************************
%Author: Chaoqun Wang
%Date: 2019-10-15
%***************************************
%% 流程初始化
clc
clear all;
close all;
x_I=1; y_I=1;           % 设置初始点
x_G=700; y_G=700;       % 设置目标点（可尝试修改终点）
Thr=50;                 % 设置目标点阈值
Delta= 30;              % 设置扩展步长
%% 建树初始化
T.v(1).x = x_I;         % T是我们要做的树，v是节点，这里先把起始点加入到T里面来
T.v(1).y = y_I; 
T.v(1).xPrev = x_I;     % 起始节点的父节点仍然是其本身
T.v(1).yPrev = y_I;
T.v(1).dist=0;          % 从父节点到该节点的距离，这里可取欧氏距离
T.v(1).indPrev = 0;     %
nodeCoords = [T.v(1).x, T.v(1).y];  % 存储所有节点的(x,y)，方便KNN查找
%% 开始构建树，作业部分
figure(1);
ImpRgb=imread('newmap.png');
Imp=rgb2gray(ImpRgb);
imshow(Imp)
xL=size(Imp,2);%地图x轴长度
yL=size(Imp,1);%地图y轴长度
hold on
plot(x_I, y_I, 'ro', 'MarkerSize',10, 'MarkerFaceColor','r');
plot(x_G, y_G, 'go', 'MarkerSize',10, 'MarkerFaceColor','g');% 绘制起点和目标点
count=1;
bFind = false;

for iter = 1:3000
    n=1;
    x = randi([1, xL], 1, n);  % 生成1到100之间的整数x坐标
    y = randi([1, yL], 1, n);  % 生成1到100之间的整数y坐标
    x_rand=[x(1),y(1)];
    %Step 1: 在地图中随机采样一个点x_rand
    %提示：用（x_rand(1),x_rand(2)）表示环境中采样点的坐标

    % x_near=[];

    [x_near, dist, ind_near] = find_nearest_node(nodeCoords, x_rand);  % 传入nodeCoords和x_rand
    %Step 2: 遍历树，从树中找到最近邻近点x_near 
    %提示：x_near已经在树T里
    
    dir=atan2(x_rand(2)-x_near(2),x_rand(1)-x_near(1));
    x_new = x_near + Delta * [cos(dir), sin(dir)];  % 注意：原代码缺少x_near的偏移，这里已修正
    %Step 3: 扩展得到x_new节点
    %提示：注意使用扩展步长Delta
    
    %检查节点是否是collision-free
    if ~collisionChecking(x_near,x_new,Imp) 
        continue;
    end
    
    %Step 4: 将x_new插入树T 
    %提示：新节点x_new的父节点是x_near

    new_node.x = x_new(1);
    new_node.y = x_new(2);
    new_node.xPrev = x_near(1);
    new_node.yPrev = x_near(2);
    new_node.dist = Delta;
    new_node.indPrev = ind_near;
    count=count+1;
    T.v(count) = new_node;
    nodeCoords = [nodeCoords; new_node.x, new_node.y];
    %Step 5:检查是否到达目标点附近 
    %提示：注意使用目标点阈值Thr，若当前节点和终点的欧式距离小于Thr，则跳出当前for循环
    if Thr > sqrt((x_G - new_node.x)^2 + (y_G - new_node.y)^2)
        bFind = true;
        % 补充绘制最后一段路径（x_near到x_new）
        plot([x_near(1), x_new(1)], [x_near(2), x_new(2)], 'b-', 'LineWidth', 1.5);
        plot(x_new(1), x_new(2), 'ro', 'MarkerSize', 5);
        hold on;
        break;
    end

    
    %Step 6:将x_near和x_new之间的路径画出来
    %提示 1：使用plot绘制，因为要多次在同一张图上绘制线段，所以每次使用plot后需要接上hold on命令
    %提示 2：在判断终点条件弹出for循环前，记得把x_near和x_new之间的路径画出来

    plot([x_near(1), x_new(1)], [x_near(2), x_new(2)], 'b-', 'LineWidth', 1.5);
    plot(x_new(1), x_new(2), 'ro', 'MarkerSize', 5);
    pause(0.001);
end
% %% 路径已经找到，反向查询
% if bFind
%     path.pos(1).x = x_G; path.pos(1).y = y_G;
%     path.pos(2).x = T.v(end).x; path.pos(2).y = T.v(end).y;
%     pathIndex = T.v(end).indPrev; % 终点加入路径
%     j=0;
%     while 1
%         path.pos(j+3).x = T.v(pathIndex).x;
%         path.pos(j+3).y = T.v(pathIndex).y;
%         pathIndex = T.v(pathIndex).indPrev;
%         if pathIndex == 1
%             break
%         end
%         j=j+1;
%     end  % 沿终点回溯到起点
%     path.pos(end+1).x = x_I; path.pos(end).y = y_I; % 起点加入路径
%     for j = 2:length(path.pos)
%         plot([path.pos(j).x; path.pos(j-1).x;], [path.pos(j).y; path.pos(j-1).y], 'b', 'Linewidth', 3);
%     end
% else
%     disp('Error, no path found!');
% end
%% 路径已经找到，反向查询（修正后）
if bFind
    path = struct('x', [], 'y', []);  % 初始化路径结构体
    % 从最后一个节点（靠近目标的节点）开始回溯
    current_idx = length(T.v);  % 最后一个节点的索引（T.v(end)的索引）
    while current_idx ~= 0  % 当索引为0时停止（根节点的indPrev=0）
        path(end+1).x = T.v(current_idx).x;
        path(end).y = T.v(current_idx).y;
        current_idx = T.v(current_idx).indPrev;  % 回溯到父节点
    end
    % 反转路径，从起点到目标点（原路径是从目标到起点）
    path = flip(path);
    
    % 绘制最终路径（从起点到目标）
    for j = 2:length(path)
        plot([path(j-1).x, path(j).x], [path(j-1).y, path(j).y], 'r-', 'LineWidth', 3);
    end
    % 补充绘制目标点到最后一个节点的路径
    plot([path(end).x, x_G], [path(end).y, y_G], 'r-', 'LineWidth', 3);
else
    disp('Error, no path found!');
end