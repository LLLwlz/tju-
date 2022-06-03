function S=lishu(x,y)
    [~,n]=size(x);
    S=zeros(n,5);
    for i=1:n
        S(i,:)=F(x(i),y(i,:));
    end
    


