clear
clc

load('data_MMP.mat')

N=64;
x_choose=linspace(3,5,N);
y_choose=linspace(-1,1,N);

[xx,yy]=meshgrid(x_choose,y_choose);
zz=xx+sqrt(-1)*yy;

psiA=zeros(N,N);
psiB=zeros(N,N);

X=real(Zj);
Y=imag(Zj);

In=inpolygon(xx,yy,X,Y);
Out=zeros(N,N);

for i=1:N
    for j=1:N
        if In(i,j)==0
            Out(i,j)=1;
        end
    end
end

t=1;
for i=1:length(nL)
    
    for j=1:Nm
        
        Z_p=(zz-Zm(j));
        r_p=abs(Z_p);
        theta_p=angle(Z_p);
        
        psiA=psiA+C(t)*besselh(nL(i), 1, k1*r_p).*exp(sqrt(-1)*nL(i)*theta_p).*Out;
        psiB=psiB+s1*sqrt(-1)^valley_index*C(t)*besselh(nL(i)+valley_index, 1, k1*r_p).*exp(sqrt(-1)*(nL(i)+valley_index)*theta_p).*Out;
        
        t=t+1;
    end
    
    disp(t)
end

for i=1:length(nL)
    
    for j=1:Nl
        
        Z_p=(zz-Zl(j));
        r_p=abs(Z_p);
        theta_p=angle(Z_p);
        
        psiA=psiA+C(t)*besselh(nL(i), 1, k2*r_p).*exp(sqrt(-1)*nL(i)*theta_p).*In;
        psiB=psiB+s2*sqrt(-1)^valley_index*C(t)*besselh(nL(i)+valley_index, 1, k2*r_p).*exp(sqrt(-1)*(nL(i)+valley_index)*theta_p).*In;
        
        t=t+1;
    end
    
	disp(t)
end

psiA=psiA+besselh(0,1,abs(zz)*k1)/sqrt(2).*Out;
psiB=psiB+sqrt(-1)*besselh(1,1,abs(zz)*k1).*exp(sqrt(-1)*angle(zz))/sqrt(2).*Out;

norm_sca=sum(sum(abs(psiA).^2+abs(psiB).^2))/N^2;
psiA=psiA./sqrt(norm_sca);
psiB=psiB./sqrt(norm_sca);

phi=zeros(4,64,64);
phi(1,:,:)=real(psiA);
phi(2,:,:)=imag(psiA);
phi(3,:,:)=real(psiB);
phi(4,:,:)=imag(psiB);

figure()
subplot(2,2,1)
pcolor(xx, yy,  real(psiA)); hold on;
plot(X, Y, 'k--');
shading flat;
colorbar

subplot(2,2,2)
pcolor(xx, yy,  real(psiB)); hold on;
plot(X, Y, 'k--');
shading flat; 
colorbar

subplot(2,2,3)
pcolor(xx, yy,  imag(psiA)); hold on;
plot(X, Y, 'k--');
shading flat;
colorbar

subplot(2,2,4)
pcolor(xx, yy,  imag(psiB)); hold on;
plot(X, Y, 'k--');
shading flat; 
colorbar

save phi_circle_k_60_V_250.mat phi