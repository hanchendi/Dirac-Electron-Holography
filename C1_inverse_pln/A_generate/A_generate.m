clear
clc

N=64;
x_choose=linspace(3,5,N);
y_choose=linspace(-1,1,N);

[xx,yy]=meshgrid(x_choose,y_choose);
zz=xx+sqrt(-1)*yy;

%% Generate a plane wave with energy k0
k0=10;
s0=1;

psiA_target=exp(sqrt(-1)*k0*xx)/sqrt(2);
psiB_target=s0*exp(sqrt(-1)*k0*xx)/sqrt(2);
norm_target=sum(sum(abs(psiA_target).^2+abs(psiB_target).^2))/N^2;
psiA_target=psiA_target./sqrt(norm_target);
psiB_target=psiB_target./sqrt(norm_target);

phi=zeros(4,64,64);
phi(1,:,:)=real(psiA_target);
phi(2,:,:)=imag(psiA_target);
phi(3,:,:)=real(psiB_target);
phi(4,:,:)=imag(psiB_target);

figure()
subplot(2,2,1)
pcolor(xx, yy,  real(psiA_target)); hold on;
shading flat;
colorbar

subplot(2,2,2)
pcolor(xx, yy,  real(psiB_target)); hold on;
shading flat; 
colorbar

subplot(2,2,3)
pcolor(xx, yy,  imag(psiA_target)); hold on;
shading flat;
colorbar

subplot(2,2,4)
pcolor(xx, yy,  imag(psiB_target)); hold on;
shading flat; 
colorbar

save data_10.mat phi
