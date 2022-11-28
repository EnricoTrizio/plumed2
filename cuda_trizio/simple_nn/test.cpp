  #include <torch/torch.h>
  #include <torch/script.h>
  #include <torch/cuda.h>
  //#include <cuda_runtime.h>
  //#include <cudnn.h>
//   #include <ATen/ATen.h>

// ########################### START FUNCTIONS ###########################

std::vector<float> tensor_to_vector(const torch::Tensor& x) {
  return std::vector<float>(x.data_ptr<float>(), x.data_ptr<float>() + x.numel());
}

// ############################ END FUNCTIONS ############################


// #######################################################################
// ############################ MAIN CODE ################################
// #######################################################################
int main() {

// init net extremes
unsigned _n_in = 4;
unsigned _n_out = 2;
  
std::string fname="cuda_cv_4_2.pt";
torch::jit::script::Module _model = torch::jit::load(fname);
  
torch::Device device = torch::kCPU; // otherwise the if scope is wrong
// initialize device type
if (! torch::cuda::is_available()) {
  std::cout << "CUDA not available." << std::endl;
  device = torch::kCPU;
} else{
  std::cout << "CUDA is available." << std::endl;
  device = torch::kCUDA;
}
// //device = torch::kCPU;
std::cout << "Device is: " <<  device << std::endl;
// // move model to device
_model.to(device);

// create input vector and move it to tensor
// we have to place it already on the right device
// the torch::jit::IValue doesn't have the .to() method 
std::vector<float> input_test (_n_in); // create standard vector
for(unsigned j=0; j<_n_in; j++) {
  input_test[j] = j+1;
} 
torch::Tensor single_input = torch::tensor(input_test).view({1,_n_in}).to(device);
// set requires grad!!
single_input.set_requires_grad(true);
std::vector<torch::jit::IValue> inputs;
inputs.push_back(single_input);

// apply basic model that computes the square of the input 
torch::Tensor output = _model.forward( inputs ).toTensor();
std::vector<std::vector<float>> final_der (_n_out, std::vector<float>(_n_in));

// compute some derivatives
for(unsigned j=0; j<_n_out; j++) {
    auto grad_output = torch::ones({1}).expand({1, 1}).to(device);
    auto gradient = torch::autograd::grad({output.slice(/*dim=*/1, /*start=*/j, /*end=*/j+1)},
                        {single_input},
        /*grad_outputs=*/ {grad_output},
        /*retain_graph=*/true,
        /*create_graph=*/false)[0]; // the [0] is to get a tensor and not a vector<at::tensor>

    // get outputs and derivatives as vectors --> use tensor_to_vector
    // everything needs to go back to the cpu now
    gradient = gradient.to(torch::kCPU);
    std::vector<float> der = tensor_to_vector ( gradient );
    std::cout << "- Gradients " << j << ": \t" << der << std::endl;
    copy(der.begin(), der.end(), final_der[j].begin());
}

output = output.to(torch::kCPU);
std::vector<float> out_vec = tensor_to_vector (output);

std::cout << "" << std::endl;
std::cout << "We can get our results!" << std::endl;
std::cout << "- Input: \n\t" << input_test << std::endl;
std::cout << "- Output: \n\t" << out_vec << std::endl;
std::cout << "- Gradients: \n\t" << final_der << std::endl;
std::cout << "- Gradients: \n";
for (int i = 0; i < final_der.size(); i++) 
    {
      std::cout << "\t"<< final_der[i] << " \n";
        }    
std::cout << std::endl;
return 0;
}
