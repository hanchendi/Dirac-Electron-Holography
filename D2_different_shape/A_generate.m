clear
clc

%% Fidelity is computed using energy range from 5 to 10, the potential is not normalized when input to the MMP
E_choose=linspace(5,10,500);
V=[ 0.5632   11.7291    8.7194   10.9642    4.1055   10.3942];
F_circle=zeros(1,500);
F_star=zeros(1,500);

for E_idx=1:500
    
    E=E_choose(E_idx);
    [phi_inverse_A,phi_inverse_B,score,ee6] = MMP_6_single(E,V);
    [phi_circle_A,phi_circle_B,ee_circle] = MMP_circle(E);
    [phi_star_A,phi_star_B,ee_star] = MMP_star(E);
    
    F_circle(E_idx)=abs(sum(sum(phi_circle_A.*conj(phi_inverse_A)+phi_circle_B.*conj(phi_inverse_B)))/(64^2));
    F_star(E_idx)=abs(sum(sum(phi_star_A.*conj(phi_inverse_A)+phi_star_B.*conj(phi_inverse_B)))/(64^2));
    disp([E F_circle(E_idx) F_star(E_idx)])
    
end

save data_fidelity
