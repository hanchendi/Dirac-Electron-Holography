clear
clc

load('data_5.mat')
for i=1:4
    subplot(2,4,i)
    temp=phi(i,:,:);
    temp=reshape(temp,[64,64]);
    pcolor(temp)
    shading flat;colorbar
    caxis([-1/sqrt(2) 1/sqrt(2)])
end
phi_real_A=reshape(phi(1,:,:),[64,64])+sqrt(-1)*reshape(phi(2,:,:),[64,64]);
phi_real_B=reshape(phi(3,:,:),[64,64])+sqrt(-1)*reshape(phi(4,:,:),[64,64]);

load('data_5_design.mat')
for i=1:4
    subplot(2,4,i+4)
    temp=phi(i,:,:);
    temp=reshape(temp,[64,64]);
    pcolor(temp)
    shading flat;colorbar
    caxis([-1/sqrt(2) 1/sqrt(2)])
end
phi_inverse_A=reshape(phi(1,:,:),[64,64])+sqrt(-1)*reshape(phi(2,:,:),[64,64]);
phi_inverse_B=reshape(phi(3,:,:),[64,64])+sqrt(-1)*reshape(phi(4,:,:),[64,64]);

f=sum(sum(phi_real_A.*conj(phi_inverse_A)+phi_real_B.*conj(phi_inverse_B)))/(64^2);
disp(abs(f))
