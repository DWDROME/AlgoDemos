function dist = distance(x1,y1,x2,y2)
%This function calculates the distance between any two cartesian 
%coordinates.
%   Copyright 2009-2010 The MathWorks, Inc.
%dist=sqrt((x1-x2)^2 + (y1-y2)^2);
dx = abs(x1 - x2);
dy = abs(y1 - y2);
dist = (dx + dy) + (sqrt(2) - 2) * min(dx, dy);  % 用sqrt()替代math.sqrt()