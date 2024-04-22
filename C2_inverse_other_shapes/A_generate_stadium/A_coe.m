clear
clc

%% global parameter

valley_index=1;
k=9;
V=20;

a=0.4;

x_center=1.5;
y_center=0;

s1=sign(k);
s2=sign(k-V);

k1=abs(k);
k2=abs(k-V);

% inside
Nm = 4*40;
Rm = 0.7; 

thetm1 = linspace(-pi/2, pi/2, Nm/4);
thetm2 = pi/2*ones(1, Nm/4);
thetm3 = linspace(pi/2, 3*pi/2, Nm/4);
thetm4 = -pi/2*ones(1, Nm/4);

zRm1 = a + Rm*exp(sqrt(-1)*thetm1);
zRm2 = linspace(a, -a, length(thetm2)) + Rm*exp(sqrt(-1)*thetm2);
zRm3 = -a + Rm*exp(sqrt(-1)*thetm3);
zRm4 = linspace(-a, a, length(thetm4)) + Rm*exp(sqrt(-1)*thetm4);
Zm = [zRm1 zRm2(2:end) zRm3(2:end) zRm4(2:end-1)];
Zm=Zm+x_center+sqrt(-1)*y_center;
Nm = length(Zm);

% outside

Nl = 4*44;
Rl = 0.9; 

thetl1 = linspace(-pi/2, pi/2, Nl/4);
thetl2 = pi/2*ones(1, Nl/4);
thetl3 = linspace(pi/2, 3*pi/2, Nl/4);
thetl4 = -pi/2*ones(1, Nl/4);

zRl1 = a + Rl*exp(sqrt(-1)*thetl1);
zRl2 = linspace(a, -a, length(thetl2)) + Rl*exp(sqrt(-1)*thetl2);
zRl3 = -a + Rl*exp(sqrt(-1)*thetl3);
zRl4 = linspace(-a, a, length(thetl4)) + Rl*exp(sqrt(-1)*thetl4);
Zl = [zRl1 zRl2(2:end) zRl3(2:end) zRl4(2:end-1)];
Zl=Zl+x_center+sqrt(-1)*y_center;
Nl = length(Zl);

% boundary

Nj = (Nm+Nl)*4;
Rj = 0.8; 

thetj1 = linspace(-pi/2, pi/2, Nj/4);
thetj2 = pi/2*ones(1, Nj/4);
thetj3 = linspace(pi/2, 3*pi/2, Nj/4);
thetj4 = -pi/2*ones(1, Nj/4);

zRj1 = a + Rj*exp(sqrt(-1)*thetj1);
zRj2 = linspace(a, -a, length(thetj2)) + Rj*exp(sqrt(-1)*thetj2);
zRj3 = -a + Rj*exp(sqrt(-1)*thetj3);
zRj4 = linspace(-a, a, length(thetj4)) + Rj*exp(sqrt(-1)*thetj4);
Zj = [zRj1 zRj2(2:end) zRj3(2:end) zRj4(2:end-1)];
Zj=Zj+x_center+sqrt(-1)*y_center;
Nj = length(Zj);

% Construct matrix

[zl, zjl] = meshgrid(Zm, Zj);
Djl = zjl - zl; 
Phi_A = angle(Djl);
R_A= abs(Djl);

[zl, zjl] = meshgrid(Zl, Zj);
Djl = zjl - zl; 
Phi_B = angle(Djl);
R_B = abs(Djl);

L=1;
nL=-L:L;

plnA=besselh(0,1,abs(Zj)*k1)/sqrt(2);
plnB=sqrt(-1)*besselh(1,1,abs(Zj)*k1).*exp(sqrt(-1)*angle(Zj))/sqrt(2);

pln=conj([plnA plnB]');

A1=[];A2=[];
C1=[];C2=[];

for l=1:length(nL)    
    
        HjmL1 = besselh(nL(l), 1, k1*R_A).*exp(sqrt(-1)*nL(l)*Phi_A);
        HjmL2 = sqrt(-1)^valley_index*s1*besselh(nL(l)+valley_index, 1, k1*R_A).*exp(sqrt(-1)*(nL(l)+valley_index)*Phi_A);
        
        A1 = [A1 HjmL1];
        A2 = [A2 HjmL2];
        
        HjlR1 = besselh(nL(l), 1, k2*R_B).*exp(sqrt(-1)*nL(l)*Phi_B);
        HjlR2 = sqrt(-1)^valley_index*s2*besselh(nL(l)+valley_index, 1, k2*R_B).*exp(sqrt(-1)*(nL(l)+valley_index)*Phi_B);
        
        C1 = [C1  HjlR1];
        C2 = [C2  HjlR2];
        
end

MA=[A1;A2];
MB=[C1;C2];

M=[MA -MB];
C = -pinv(M)*pln;
    
ee = norm(M*C + pln)/norm(pln);

plot(real(Zm),imag(Zm),'r*');hold on

plot(real(Zl),imag(Zl),'bo');hold on

plot(real(Zj),imag(Zj),'k--');hold on
axis equal

save data_MMP