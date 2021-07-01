clc;
clear all;
for num=1:300
  I_ir=(imread(strcat('Medical\Function\RGB\',num2str(num),'.png')));  
  [Y,Cb,Cr]=RGB2YCbCr(I_ir); 
  imwrite(Y, strcat('Medical\Function\Y\',num2str(num),'.png'));
end