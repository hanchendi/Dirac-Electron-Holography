Simulation code for Dirac electron holography. Uploaded 04/22/2024.

-----A_data_generation----- Matlab code to generate training data. 
A_create_poles.m: This will generate the matrix that is needed for the MMP method.
B_multi_run.m: This will generate random numbers for energy and potential that are used to compute MMP.
C_test.m: Used to plot the sample data.
MMP_single.m: A Matlab function. The input will be the potential and energy, and the output will be the wave function and the error.

-----B1_neural_network-----Python code to train the neural network. The code follows a standard generator structure but with a residual connection. After the neural network is trained, it will save the first one thousand data to compute the train, validate, and test errors.
Generator_Res.py: Standard MSE loss used in the paper.
Generator_Res_custom.py: Physical loss follows [Chen, Mingkun, et al. "High speed simulation and freeform optimization of nanophotonic devices with physics-augmented deep learning." ACS Photonics 9.9 (2022): 3110-3123.], but the physical loss does not give any advantages for this problem.

------B2_train_result-----Matlab code to plot the training result. It is also useful for checking the prediction and nearest training data around it.

------C1_inverse_pln-----
A_generate: Matlab code to generate the target plane wave. It will save the data based on its energy.
B_inverse: Python code to do the inverse design, it will load the neural network saved in the training part. The optimization will start from some random number and gradient descent. Manually changing some optimization algorithms will make the results better, but it is not needed.
C_result: Matlab code that compares the design result, the computing uses the ground truth (MMP method), not the neural network prediction, and the real plane wave.
D_cylinder_fidelity: Matlab code to compute the fidelity in free space (without any scattering).

------C2_inverse_other_shapes-----
A_generate_circle: A_coe.m used to create the matrix using for MMP computation. B_wave.m computing the scattering wave for the whole region. C_far_filed.m computing the scattering wave in the desired region only, then normalized and saved. 
A_generate_stadium and A_generate_star follows the same structure. The star shape needs some parameters from Gielis formula. Then you can run the same inverse design function as a plane wave to get the best parameters.

------C3_random_design-----
The target is randomly generated gate potential. The data generation part is the same as generating training data. The inverse function is also the same.

------D1_wideband-----
For two targets from the same shape, the loss is added directly. Finally, using the MMP method to compute the wave from the circle and 6 dots scatters to get the fidelity.

------D2_different_shape-----
For two targets of different shapes, the loss is added directly. Finally, the MMP method is used to compute the wave from the circle, star, and six dots scatters to get the fidelity.

------Final_figures-----
Matlab code and data to generate figure in the article.
