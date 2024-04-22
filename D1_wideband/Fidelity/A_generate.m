clear
clc

E_choose=linspace(5,10,500);
V=[0.6307    2.0731   -3.3894   15.2217    3.0394    0.6676];
F_circle=zeros(1,500);

for E_idx=1:500
    
    E=E_choose(E_idx);
    [phi_inverse_A,phi_inverse_B,score,ee6] = MMP_6_single(E,V);
    [phi_circle_A,phi_circle_B,ee_circle] = MMP_circle(E);
    
    F_circle(E_idx)=abs(sum(sum(phi_circle_A.*conj(phi_inverse_A)+phi_circle_B.*conj(phi_inverse_B)))/(64^2));
    disp([E F_circle(E_idx)])
    
end

save data_fidelity