function [y,type]=qujian2jida(x,type,a,b)
%实现区间指标转为正向指标，返回正向指标矩阵
%x为原始数据矩阵, 一行代表一个样本, 每列对应一个指标
%type设定正向指标1,负向指标2
%a,b为区间端点
[n,m]=size(x);
y=zeros(n,m);
xmin=min(x);
xmax=max(x);
M=max(a-xmin,xmax-b);
for i=1:n
    if x(i,1)<a
        y(i,1)=1-(a-x(i,1))/M;
    elseif x(i,1)>b
        y(i,1)=1-(x(i,1)-b)/M;
    else
        y(i,1)=1;
    end
end
type=type+1;
