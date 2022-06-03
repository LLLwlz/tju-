%%
clear
clc
N=14229;
time=xlsread('C:\Users\wlz\Desktop\校选\校选\附件1.xlsx',1,'A2:A14230');
tempuare=xlsread('C:\Users\wlz\Desktop\校选\校选\附件1.xlsx',1,'B2:C14230');
plot(1:N,tempuare(:,1));
hold on
plot(1:N,tempuare(:,2));
%%
clear
clc
N=240;
time=xlsread('C:\Users\wlz\Desktop\校选\校选\附件1.xlsx',2,'A2:A241');
=xlsread('C:\Users\wlz\Desktop\校选\校选\附件1.xlsx',2,'B2:E241');
