clear
clc

%% global parameter

valley_index=1;
k=6;
V=25;

x_center=1.5;
y_center=0;

s1=sign(k);
s2=sign(k-V);

k1=abs(k);
k2=abs(k-V);

% inside
Nm = 300;
Rm = 0.9; 

thetm = linspace(0,2*pi,Nm+1);
thetm = thetm(1:Nm);
Zm=Rm.*exp(sqrt(-1)*thetm);
Zm=Zm+x_center+sqrt(-1)*y_center;

% outside

Nl = 300;
Rl = 1.1; 

thetl = linspace(0,2*pi,Nl+1);
thetl = thetl(1:Nl);
Zl=Rl.*exp(sqrt(-1)*thetl);
Zl=Zl+x_center+sqrt(-1)*y_center;

% boundary

Nj = (Nm+Nl)*4;
Rj=1; 

thetj= linspace(0,2*pi,Nj+1);
thetj = thetj(1:Nj);
Zj=Rj*exp(sqrt(-1)*thetj);
Zj=Zj+x_center+sqrt(-1)*y_center;

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