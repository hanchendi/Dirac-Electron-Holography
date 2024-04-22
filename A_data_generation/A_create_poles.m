clear
clc

%% inside Ni, outside No
Ni=40;
No=44;
Nj=(Ni+No)*3;

delta_in=0.1;
delta_out=0.1;

Zi=[];
Zo=[];
Zj=[];

theta_i=linspace(0,2*pi,Ni+1);
theta_o=linspace(0,2*pi,No+1);
theta_j=linspace(0,2*pi,Nj+1);

theta_i=theta_i(1:Ni);
theta_o=theta_o(1:No);
theta_j=theta_j(1:Nj);
rj=0.35;
ri=(1-delta_in)*rj;
ro=(1+delta_out)*rj;

for j=1:3
    for i=1:2
        
        xc=i;
        yc=j-2;
        
        x=xc+ri*cos(theta_i);
        y=yc+ri*sin(theta_i);
        Zi=[Zi x+sqrt(-1)*y];
        
        x=xc+ro*cos(theta_o);
        y=yc+ro*sin(theta_o);
        Zo=[Zo x+sqrt(-1)*y];
        
        x=xc+rj*cos(theta_j);
        y=yc+rj*sin(theta_j);
        Zj=[Zj x+sqrt(-1)*y];
        
    end
end

% Construct matrix -- inside

for i=1:6
    
    [zl, zjl] = meshgrid(Zo((i-1)*No+1:i*No), Zj((i-1)*Nj+1:i*Nj));
    Djl = zjl - zl; 
    eval(['Phi_',num2str(i),' = angle(Djl);']);
    eval(['R_',num2str(i),' = abs(Djl);']);
    
end

for i=1:6
    for j=1:6
        
        [zl, zjl] = meshgrid(Zi((i-1)*Ni+1:i*Ni), Zj((j-1)*Nj+1:j*Nj));
        Djl = zjl - zl; 
        eval(['Phi_',num2str(i),'_',num2str(j),' = angle(Djl);']);
        eval(['R_',num2str(i),'_',num2str(j),' = abs(Djl);']);
        
    end
end

save data_poles