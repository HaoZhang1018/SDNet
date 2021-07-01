clc;
clear all;
for num=1:300
  I_ir=(imread(strcat('IR_VIS\VISIBLE\RGB\',num2str(num),'.png')));  
  [Y,Cb,Cr]=RGB2YCbCr(I_ir); 
  imwrite(Y, strcat('IR_VIS\VISIBLE\Y\',num2str(num),'.png'));
end