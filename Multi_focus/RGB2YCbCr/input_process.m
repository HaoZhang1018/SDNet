clc;
clear all;
for num=1:300
  I_under=(imread(strcat('MFF\Far\RGB\',num2str(num),'.png')));  
  [Y_under,Cb_under,Cr_under]=RGB2YCbCr(I_under);
  imwrite(Y_under, strcat('MFF\Near\Y\',num2str(num),'.png'));
    
  I_over=(imread(strcat('MFF\Far\RGB\',num2str(num),'.png')));  
  [Y_over,Cb_over,Cr_over]=RGB2YCbCr(I_over);
  imwrite(Y_over, strcat('MFF\Near\Y\',num2str(num),'.png'));

end