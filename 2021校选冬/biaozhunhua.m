function X=biaozhunhua(x,ind)
%实现用熵值法求各指标(列）的权重及各数据行的得分
%x为原始数据矩阵, 一行代表一个样本, 每列对应一个指标
%ind指示向量，指示各列正向指标还是负向指标，1表示正向指标，2表示负向指标
%s返回各行（样本）得分，w返回各列权重
[n,m]=size(x); % n个样本, m个指标
%%数据的归一化处理
X=zeros(n,m);%存储归一化矩阵
x=x./repmat(sum(x.*x).^0.5,n,1);%标准化
for i=1:m
    if ind(i)==1 %正向指标归一化
        X(:,i)=guiyi(x(:,i),1,0.002,0.996);    %若归一化到[0,1], 0会出问题
    elseif ind(i)==0 
        x(:,i)=qujian2jida(x(:,i));
        X(:,i)=guiyi(x(:,i),1,0,1);
    else %负向指标归一化
        X(:,i)=guiyi(x(:,i),2,0.002,0.996);
    end
end