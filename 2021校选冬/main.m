clc
clear
num = xlsread("C:\Users\wlz\Desktop\testa.xlsx",1, 'D25:I28');
ind=[1,2,2,2,2,2];
[n,m]=size(num);
num0=num;
num0(:,1)=num0(:,1)-7.5;
[s,w]=shang(num,ind);
y=[0,1.5,2.5,4.5,5.5;
    2,4,6,10,15;
    15,15,20,30,40;
    %3,3,4,6,10;
    0.15,0.5,1.0,1.5,2.0;
    0.02,0.1,0.2,0.3,0.4
    0.2,0.5,1.0,1.5,2.0;];

Q=zeros(n,5);
for i=1:n
    S=lishu(num0(i,:),y);
    a=w*S
    S=S.*w';
    for j=1:5
        Q(i,j)=sum(S(:,j));
    end
end