clear
clc

a=rand+0.2;
b=rand+0.2;
m=10;
n1=rand*3;
n2=rand*3;
n3=rand*3;

phi=linspace(0,2*pi,10^3);
r=1./((1/a*abs(cos(phi*m/4))).^n2+(1/b*abs(sin(phi*m/4))).^n3).^(1/n1);
plot(r.*cos(phi),r.*sin(phi),'linewidth',1.5);
% axis off;
axis equal;