1. Remove color components. If the source data is color image, please run "input_process.m" to obtain the Y component as the training data. Otherwise, proceed directly to the second step.

2. Training. Put the training data in the "Train_far" and "Train_near" folders according to their modalities. Then run "CUDA_VISIBLE_DEVICES=0 python train.py" to train the network. 

3. Test. Put the test data in the "Test_far" and "Test_near" folders according to their modalities. Then run "CUDA_VISIBLE_DEVICES=0 python test.py" to test the trained model. You can also directly use the trained model we provide.

4. Restore color components. If the source data is color image, please run "output_process.m" to obtain the final fused images. Otherwise, do not perform this step.