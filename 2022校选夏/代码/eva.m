%%
ave_xgb=[0.8134851063829808;1.4374723404255318;0.4676957446808508;4.278489361702127];
ave_RF=[0.38378304702127586;0.787007691702125;0.22579476659574313;2.568723737234042];
ave_DT=[0.6711680851063809;1.469976595744681;0.4363978723404252;3.2782297872340425];
save ave_xgb ave_xgb
save ave_RF ave_RF
save ave_DT ave_DT
%%
clc; clear
close all
%% 导入数据
% 分别导入不同方法的均值数据
% 每一行为一个参数
% 每一列为一个参变量，误差值
load ave_BPGA
load ave_reg
load ave_xgb
load ave_RF
load ave_DT

%% 数据处理
par1=[ave_BPGA(1,:);ave_reg(1,:)];
par2=[ave_BPGA(2,:);ave_reg(2,:)];
par3=[ave_BPGA(3,:);ave_reg(3,:)];
par4=[ave_BPGA(4,:);ave_reg(4,:)];
%% 因子分析法
% 计算相关系数矩阵
% par应为行表示观测值，列表示随机变量
par1=zscore(par1);
r_par1=corrcoef(par1);
%下面利用相关系数矩阵求主成分解，val的列为r的特征向量，即主成分的系数
[vec_par1,val_par1,con1_par1]=pcacov(r_par1); %val为r的特征值，con为各个主成分的贡献率
num_par1=input('请选择参数的公共因子的个数：');  %交互式选取主因子的个数
f1_par1=repmat(sign(sum(vec_par1)),size(vec_par1,1),1);
vec_par1=vec_par1.*f1_par1;     %特征向量正负号转换
f2_par1=repmat(sqrt(val_par1)',size(vec_par1,1),1);
a_par1=vec_par1.*f2_par1;   %计算因子载荷矩阵
am_par1=a_par1(:,1:num_par1);   %提出两个主因子的载荷矩阵
[bm_par1,t_par1]=rotatefactors(am_par1,'method', 'varimax'); %am旋转变换,bm为旋转后的载荷阵
bt_par1=[bm_par1,a_par1(:,num_par1+1:end)];  %旋转后全部因子的载荷矩阵,前两个旋转，后面不旋转
con2_par1=sum(bt_par1.^2);       %计算因子贡献
check_par1=[con1_par1,con2_par1'/sum(con2_par1)*100];%该语句是领会旋转意义,con1是未旋转前的贡献率
rate_par1=con2_par1(1:num_par1)/sum(con2_par1); %计算因子贡献率
coef_par1=r_par1\bm_par1;          %计算得分函数的系数
score_par1=par1*coef_par1  ;         %计算各个因子的得分
weight_par1=rate_par1/sum(rate_par1);  %计算得分的权重
Tscore_par1=score_par1*weight_par1' ;  %对各因子的得分进行加权求和，即求各部分综合得分

%% 熵权法
%行表示方法
%列表示指标
% A=[ave_BPGA';ave_reg';ave_xgb';ave_RF';ave_DT'];
A=A';
[n,m]=size(A);
% B=1./A;
A=1./A;     %正向化

%标准化
if isempty(find(A < 0))
    %disp('不存在负数，标准化后的矩阵为:')
    % A_stand = A./ repmat(sum(A.*A).^(1/2),n,1) % 按列求和
    A_stand = (A - repmat(min(A),n,1))./(repmat(max(A)-min(A),n,1)); 
else
    disp('存在负数')
    A_stand = (A - repmat(min(A),n,1))./(repmat(max(A)-min(A),n,1)); 
end

% 计算概率矩阵，标准化的指标归一化
P = A_stand./repmat(sum(A_stand),n,1);

% 计算每个指标的信息熵
E = -sum(P.*My_log(P))/log(n);

% 计算权重
W = (1-E)./sum(E);
% 归一化
W = W./sum(W);

% 计算最终得分
score = sum(W.*A_stand,2);
score_stand = score ./ sum(score);
[score_stand_sort, index] = sort(score_stand, 'descend');
disp('最终名次为:')
disp(index)

