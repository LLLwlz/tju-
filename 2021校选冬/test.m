clear;
clc;
x=[0.8,3,9.2,0.9,0.096];
y=[0,1.5,2.5,4.5,5.5;
    2,4,6,10,15;
    15,15,20,30,40;
    3,3,4,6,10;
    0.02,0.1,0.2,0.3,0.4];
S=lishu(x,y);
a=[0.2025,0.2023,0.2498,0.2033,0.1421];
S1=[0.47,0.53,0,0,0;0.500000000000000,0.500000000000000,0,0,0;0.500000000000000,0.500000000000000,0,0,0;0.500000000000000,0.500000000000000,0,0,0;0.0500000000000000,0.950000000000000,0,0,0];
Q=S1.*a;
y=sum(Q(:,1));
