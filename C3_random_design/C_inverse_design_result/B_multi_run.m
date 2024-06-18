clear
clc

%% Energy range from 5 to 10
E_choose=[5 5.5 6 6.5 7 7.5 8 8.5 9 9.5 10];
E_idx=11;
load(['data_E_',num2str(E_idx),'.mat'])
load(['R_error_',num2str(E_idx),'.mat'])
load(['R_V_',num2str(E_idx),'.mat'])

F_final=zeros(1,200);
E=E_choose(E_idx);
for idx=1:200
    
    V=R_V(idx,:);
    [psiA_inverse,psiB_inverse,score,ee] = MMP_single(E,V);
    psiA_target=phi_gather(idx,1,:,:)+sqrt(-1)*phi_gather(idx,2,:,:);
    psiA_target=reshape(psiA_target,[64 64]);
    psiB_target=phi_gather(idx,3,:,:)+sqrt(-1)*phi_gather(idx,4,:,:);
    psiB_target=reshape(psiB_target,[64 64]);
    
    F_final(idx)=abs(sum(sum(psiA_target.*conj(psiA_inverse)+psiB_target.*conj(psiB_inverse)))/(64^2));
    disp(F_final(idx))
end
   
save(['data_final_',num2str(E_idx),'.mat'],'F_final')
