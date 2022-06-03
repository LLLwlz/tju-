clc; clear
close all
%% 加载神经网络的训练样本，测试样本每列一个样本，输入P，输出T
% 样本数据就是前面问题描述中列出的数据
load data
% out0=[in(:,1:2)]';
% in0=[in(:,3:6),out(:,:)]';
% load data1
% in(:,8)=[];
out0=out';
in0=in';
[~,N]=size(out0);
N_test=floor(N*0.1);
[in0,Pin]=mapminmax(in0,0,1);
[out0,Pout]=mapminmax(out0,0,1);
P=in0(:,1:N-N_test);
T=out0(:,1:N-N_test);
P_test=in0(:,N-N_test+1:N);
T_test=out0(:,N-N_test+1:N);

% 初始隐含层神经元个数
hiddennum = 13;                 % 输入层个数*2 + 1
% 输入向量的最大值和最小值
% threshold = [0 1;0 1;0 1;0 1;0 1;0 1;0 1;0 1];
% threshold = [0 1;0 1;0 1;0 1;0 1;0 1;];
threshold = [0 1;0 1;0 1;0 1;0 1;0 1;0 1;0 1;0 1;0 1;];
inputnum = size(P, 1);                      % 输入层神经元个数
outputnum = size(T, 1);                     % 输出层神经元个数
w1num = inputnum * hiddennum;               % 输入层到隐含层的权值个数
w2num = outputnum * hiddennum;              % 隐含层到输出层的权值个数
N = w1num + hiddennum + w2num + outputnum;  % 待优化的变量个数
 
%% 定义遗传算法参数
NIND = 40;                      % 种群大小
MAXGEN = 100;                    % 最大遗传代数
PRECI = 10;                     % 个体长度
GGAP = 0.95;                    % 代沟
px = 0.7;                       % 交叉概率
pm = 0.01;                      % 变异概率
trace = zeros(N + 1, MAXGEN);   % 寻优结果的初始值
FieldD = [repmat(PRECI, 1, N); repmat([-0.5; 0.5], 1, N); repmat([1;0;1;1], 1, N)]; % 区域描述器
Chrom = crtbp(NIND, PRECI * N); % 创建任意离散随机种群
%% 优化
gen = 0;                                                % 代计数器
X = bs2rv(Chrom, FieldD);                               % 计算初始种群的十进制转换
ObjV = Objfun(X, P, T, hiddennum, P_test, T_test);      % 计算目标函数值
while gen < MAXGEN
    fprintf('%d\n', gen)
    FitnV = ranking(ObjV);                              % 分配适应度值
    SelCh = select('sus', Chrom, FitnV, GGAP);          % 选择
    SelCh = recombin('xovsp', SelCh, px);               % 重组
    SelCh = mut(SelCh, pm);                             % 变异
    X = bs2rv(SelCh, FieldD);                           % 子代个体的二进制到十进制转换
    ObjVSel = Objfun(X, P, T, hiddennum, P_test, T_test);       % 计算子代的目标函数值
    [Chrom, ObjV] = reins(Chrom, SelCh, 1, 1, ObjV, ObjVSel);   % 将子代重插入到父代，得到新种群
    X = bs2rv(Chrom, FieldD);
    gen = gen + 1;                                      % 代计数器增加
    % 获取每代的最优解及其序号，Y为最优解，I为个体的序号
    [Y, I] = min(ObjV);
    trace(1: N, gen) = X(I, :);                         % 记下每代的最优值
    trace(end, gen) = Y;                                % 记下每代的最优值
end
%% 画进化图
figure(1);
plot(1: MAXGEN, trace(end, :));
grid on
xlabel('遗传代数')
ylabel('误差的变化')
title('进化过程')
bestX = trace(1: end - 1, end);
bestErr = trace(end, end);
fprintf(['最优初始权值和阈值：\nX=', num2str(bestX'), '\n最小误差 err = ', num2str(bestErr), '\n'])


%% 误差评价法
[M, ~] = size(T_test);
Obj = Bptar(bestX, P, T, hiddennum, P_test);
Obj = mapminmax('reverse',Obj,Pout);
T_test = mapminmax('reverse',T_test,Pout);

% P_per=[57.5,52.5;   108.62,96.87;   44.5,46.61; 20.09,22.91;    79.17,80.10;    22.72,23.34;    10.51,11.03;    17.05,13.29;];
% P_per = mapminmax('apply',P_per,Pin);
% T_per = Bptar(bestX, P, T, hiddennum, P_per);
% T_per = mapminmax('reverse',T_per,Pout);
% T_per

Bias=sum((Obj-T_test)./M,2);    %误差均值
MSE=sum((Obj-T_test).^2./M,2);  %均方误差
RMSE=sqrt(MSE);                     %均方根误差
MAE=sum(abs(Obj-T_test)./M,2);  %绝对评价误差
MAPE=sum(abs((Obj-T_test)./T_test)./M,2);   %绝对百分比误差
SMAPE=sum(abs(Obj-T_test)./(abs(Obj)+abs(T_test)).*2./M,2); %对称平均绝对百分比误差 

%% 拟合曲线
subplot(2,2,1)
plot(1: N_test, Obj(1,:));
hold on
grid on
plot(1: N_test, T_test(1,:));
xlabel('数据组')
ylabel('指标A')
legend('预测值','真实值')

subplot(2,2,2)
plot(1: N_test, Obj(2,:));
hold on
grid on
plot(1: N_test, T_test(2,:));
ylabel('指标B')
legend('预测值','真实值')

subplot(2,2,3)
plot(1: N_test, Obj(3,:));
hold on
grid on
plot(1: N_test, T_test(3,:));
ylabel('指标C')
legend('预测值','真实值')

subplot(2,2,4)
plot(1: N_test, Obj(4,:));
hold on
grid on
plot(1: N_test, T_test(4,:));
ylabel('指标D')
legend('预测值','真实值')

%% 判定系数R2
TSS=sum((mean(T_test)-T_test).^2,2);
RSS=sum((Obj-T_test).^2,2);
R2=1-RSS./TSS;

%% Kappa 统计
% 一致性建议与衡量分类精度
% 正确率时使用

%% 数据导出
%ave_BPGA=[Bias,MSE,RMSE,MAE,SMAPE,R2]
ave_BPGA=[MSE]
save ave_BPGA ave_BPGA