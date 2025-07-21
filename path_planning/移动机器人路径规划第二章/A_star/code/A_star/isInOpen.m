function flag = isInOpen(node_x,node_y,OPEN,OPEN_COUNT)

    %EXPANDED ARRAY FORMAT
    %--------------------------------
    %|X val |Y val ||h(n) |g(n)|f(n)|
    %------------------
    flag = -1;
    for i = 1:OPEN_COUNT
        if node_x == OPEN(i,2) && node_y == OPEN(i,3) && OPEN(i,1)
            flag = i;
        end
    end