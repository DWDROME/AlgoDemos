% % 主函数：查找最近邻点，返回坐标、最小距离、索引
% function [x_near, min_dist, ind_near] = find_nearest_node(T, target_x, target_y)
%     min_dist = Inf;  % 初始化最小距离为无穷大
%     % 初始化最近点为根节点（取根节点的x和y坐标）
%     x_near = [T.v(1).x, T.v(1).y];  
%     ind_near = 1;  % 根节点索引为1
% 
%     % 调用DFS函数，需传入ind_near作为参数
%     [x_near, min_dist, ind_near] = dfs_find_nearest(T, 1, target_x, target_y, min_dist, x_near, ind_near);
% end

% 替换原find_nearest_node函数，改用knnsearch
% 放在代码末尾（作为子函数）
function [x_near, dist, ind_near] = find_nearest_node(nodeCoords, x_rand)
    % 功能：基于nodeCoords（所有节点坐标）查找x_rand的最近邻
    % 输入：
    %   nodeCoords：n×2矩阵，存储树中所有节点的(x,y)
    %   x_rand：1×2向量，随机采样点坐标
    % 输出：
    %   x_near：最近邻节点坐标（1×2）
    %   dist：最近距离
    %   ind_near：最近邻节点在树中的索引（对应T.v的索引）
    
    % 使用knnsearch查找最近邻
    [ind_near, dist] = knnsearch(nodeCoords, x_rand, 'K', 1);  % K=1表示仅查找最近的1个
    x_near = nodeCoords(ind_near, :);  % 提取最近邻坐标
end


% 递归函数：DFS遍历查找最近邻点
% 补充ind_near作为输入参数，确保递归中能传递索引
function [x_near, min_dist, ind_near] = dfs_find_nearest(T, current_index, target_x, target_y, min_dist, x_near, ind_near)
    % 打印当前访问节点（可选，用于调试）
    fprintf('访问节点 %d: (%f, %f)\n', current_index, T.v(current_index).x, T.v(current_index).y);
    
    % 获取当前节点坐标
    current_x = T.v(current_index).x;
    current_y = T.v(current_index).y;
    
    % 计算当前节点到目标点的距离
    dist = sqrt((current_x - target_x)^2 + (current_y - target_y)^2);
    
    % 若当前节点更近，更新最近点信息（包括索引）
    if dist < min_dist
        min_dist = dist;          % 更新最小距离
        x_near = [current_x, current_y];  % 更新最近点坐标
        ind_near = current_index; % 关键：更新最近点的索引为当前节点索引
    end
    
    % 递归遍历所有子节点
    num_nodes = length(T.v);
    for i = 1:num_nodes
        % 找到当前节点的子节点（父索引为当前节点索引）
        if T.v(i).indPrev == current_index
            % 递归调用，传递所有参数（包括ind_near）
            [x_near, min_dist, ind_near] = dfs_find_nearest(T, i, target_x, target_y, min_dist, x_near, ind_near);
        end
    end
end