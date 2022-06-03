clc; clear
close all
%% 加载数据
load data1
% y10=out(:,1);
% y20=out(:,2);
% y30=out(:,3);
% y40=out(:,4);
% X0=[ones(235,1),in(:,1),in(:,2),in(:,3),in(:,4),in(:,5),in(:,6),in(:,1).*in(:,2) ,in(:,1).*in(:,3), in(:,1).*in(:,4) ,in(:,1).*in(:,5), in(:,1).*in(:,6),in(:,2).*in(:,3) ,in(:,2).*in(:,4) ,in(:,2).*in(:,5) ,in(:,2).*in(:,6), in(:,3).*in(:,4) ,in(:,3).*in(:,5) ,in(:,3).*in(:,6),in(:,4).*in(:,5) ,in(:,4).*in(:,6) ,in(:,5).*in(:,6)];

% y10=in(:,1);
% y20=in(:,2);
% X0=[ones(235,1),out(:,1),out(:,2),out(:,3),out(:,4),in(:,5),in(:,6),in(:,3),in(:,4)];

y10=out(:,1);
y20=out(:,2);
y30=out(:,3);
y40=out(:,4);
X0=[ones(1638,1),in(:,1),in(:,2),in(:,3),in(:,4),in(:,5),in(:,6),in(:,7),in(:,8),in(:,9),in(:,10)];

[N,~]=size(out);
N_test=floor(N*0.1);
X=X0(1:N-N_test,:);
X_test=X0(N-N_test+1:N,:);

y1=y10(1:N-N_test,:);
y1_test=y10(N-N_test+1:N,:);
y2=y20(1:N-N_test,:);
y2_test=y20(N-N_test+1:N,:);
y3=y30(1:N-N_test,:);
y3_test=y30(N-N_test+1:N,:);
y4=y40(1:N-N_test,:);
y4_test=y40(N-N_test+1:N,:);
%% 拟合
b1 = regress(y1,X);
b2 = regress(y2,X);
b3 = regress(y3,X);
b4 = regress(y4,X);

[~,M]=size(X_test);

y1_pre=X_test*b1;
y2_pre=X_test*b2;
y3_pre=X_test*b3;
y4_pre=X_test*b4;

% Bias1=sum((y1_pre-y1_test)./M);    %误差均值
% Bias2=sum((y2_pre-y2_test)./M);
% Bias3=sum((y3_pre-y3_test)./M);
% Bias4=sum((y4_pre-y4_test)./M);
% Bias=[Bias1;Bias2;Bias3;Bias4]; 

MSE1=sum((y1_pre-y1_test).^2./M);  %均方误差
MSE2=sum((y2_pre-y2_test).^2./M);
MSE3=sum((y3_pre-y3_test).^2./M);
MSE4=sum((y4_pre-y4_test).^2./M);
MSE=[MSE1;MSE2;MSE3;MSE4];
% MSE=[MSE1;MSE2;];

% RMSE=sqrt(MSE);                     %均方根误差
% 
% MAE1=sum(abs(y1_pre-y1_test)./M);  %绝对评价误差
% MAE2=sum(abs(y2_pre-y2_test)./M);
% MAE3=sum(abs(y3_pre-y3_test)./M);
% MAE4=sum(abs(y4_pre-y4_test)./M);
% MAE=[MAE1;MAE2;MAE3;MAE4];
% 
% SMAPE1=sum(abs(y1_pre-y1_test)./(abs(y1_pre)+abs(y1_test)).*2./M); %对称平均绝对百分比误差 
% SMAPE2=sum(abs(y2_pre-y2_test)./(abs(y2_pre)+abs(y2_test)).*2./M);
% SMAPE3=sum(abs(y3_pre-y3_test)./(abs(y3_pre)+abs(y3_test)).*2./M);
% SMAPE4=sum(abs(y4_pre-y4_test)./(abs(y4_pre)+abs(y4_test)).*2./M);
% SMAPE=[SMAPE1;SMAPE2;SMAPE3;SMAPE4];
% 
% TSS1=sum((mean(y1_test)-y1_test).^2);
% RSS1=sum((y1_pre-y1_test).^2);
% R21=1-RSS1./TSS1;
% 
% TSS2=sum((mean(y2_test)-y2_test).^2);
% RSS2=sum((y2_pre-y2_test).^2);
% R22=1-RSS2./TSS2;
% 
% TSS3=sum((mean(y3_test)-y3_test).^2);
% RSS3=sum((y3_pre-y3_test).^2);
% R23=1-RSS3./TSS3;
% 
% TSS4=sum((mean(y4_test)-y4_test).^2);
% RSS4=sum((y4_pre-y4_test).^2);
% R24=1-RSS4./TSS4;
% 
% R2=[R21;R22;R23;R24];

%% 数据导出
%ave_reg=[Bias,MSE,RMSE,MAE,SMAPE,R2]
ave_reg=[MSE]
save ave_reg ave_reg
%%
% %使用残差诊断离群值
% [~,~,r,rint] = regress(y1,X,0.01);
% %通过计算不包含 0 的残差区间 rint 来诊断离群值。
% contain0 = (rint(:,1)<0 & rint(:,2)>0);
% idx = find(contain0==false)
% %创建残差的散点图。填充与离群值对应的点。
% hold on
% scatter(y1,r)
% scatter(y1(idx),r(idx),'b','filled')
% xlabel("Last Exam Grades")
% ylabel("Residuals")
% hold off
%%
%logistics非线性拟合
% beta0=[50,10,1]';
% [beta,r,j] = nlinfit(X,y1,'logistic',beta0);
% [yy,delta] = nlprodei('logistic',X,beta,r,j);

%%
%逐步回归
% stepwise(X,y1)
