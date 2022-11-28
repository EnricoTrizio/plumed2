#ifdef __PLUMED_HAS_LIBTORCH
// convert LibTorch version to string
//#define STRINGIFY(x) #x
//#define TOSTR(x) STRINGIFY(x)
//#define LIBTORCH_VERSION TO_STR(TORCH_VERSION_MAJOR) "." TO_STR(TORCH_VERSION_MINOR) "." TO_STR(TORCH_VERSION_PATCH)

#include "core/PlumedMain.h"
#include "function/Function.h"
#include "function/ActionRegister.h"

#include <torch/torch.h>
#include <torch/script.h>

#include <fstream>
#include <cmath>


#include <torch/torch.h>
#include <torch/script.h>
#include <torch/cuda.h>
  //#include <cuda_runtime.h>
  //#include <cudnn.h>
//   #include <ATen/ATen.h>

using namespace std;

namespace PLMD {
namespace function {
namespace pytorch {

class PytorchModelCuda :
  public Function
{
  unsigned _n_in;
  unsigned _n_out;
  torch::jit::script::Module _model;
  torch::Device device = torch::kCPU;
public:
  explicit PytorchModelCuda(const ActionOptions&);
  void calculate();
  static void registerKeywords(Keywords& keys);

  std::vector<float> tensor_to_vector(const torch::Tensor& x);
};

PLUMED_REGISTER_ACTION(PytorchModelCuda,"PYTORCH_MODEL_CUDA")

void PytorchModelCuda::registerKeywords(Keywords& keys) {
  Function::registerKeywords(keys);
  keys.use("ARG");
  keys.add("optional","FILE","Filename of the PyTorch compiled model");
  keys.addFlag("CUDA",false,"enable computation on the GPU");
  keys.addOutputComponent("node", "default", "Model outputs");
}


std::vector<float> PytorchModelCuda::tensor_to_vector(const torch::Tensor& x) {
  return std::vector<float>(x.data_ptr<float>(), x.data_ptr<float>() + x.numel());
}

PytorchModelCuda::PytorchModelCuda(const ActionOptions&ao):
  Action(ao),
  Function(ao)
{ //print pytorch version

  //number of inputs of the model
  _n_in=getNumberOfArguments();

  //parse model name
  std::string fname="model.ptc";
  parse("FILE",fname);
  
  //parse GPU flag
  bool _cuda;
  parseFlag("CUDA",_cuda);
  if(_cuda){
    log.printf("CUDA is requested.\n");
    log.printf("Number of cuda devices: %d \n", torch::cuda::device_count() ); 
    if (torch::cuda::is_available()) {
      log.printf("CUDA enabled\n");
      device = torch::kCUDA;
    } else {
      log.printf("WARNING: unable to activate CUDA support.\n");
    }
  }

  //deserialize the model from file
  try {
    _model = torch::jit::load(fname);
  }
  //if an error is thrown check if the file exists or not
  catch (const c10::Error& e) {
    std::ifstream infile(fname);
    bool exist = infile.good();
    infile.close();
    if (exist) {
      // print libtorch version
      std::stringstream ss;
      ss << TORCH_VERSION_MAJOR << "." << TORCH_VERSION_MINOR << "." << TORCH_VERSION_PATCH;
      std::string version;
      ss >> version; // extract into the string.
      plumed_merror("Cannot load FILE: '"+fname+"'. Please check that it is a Pytorch compiled model (exported with 'torch.jit.trace' or 'torch.jit.script') and that the Pytorch version matches the LibTorch one ("+version+").");
    }
    else {
      plumed_merror("The FILE: '"+fname+"' does not exist.");
    }
  }

  checkRead();

//move model to device
  _model.to(device); 

 //check the dimension of the output
  log.printf("Checking output dimension:\n");
  std::vector<float> input_test (_n_in);
  torch::Tensor single_input = torch::tensor(input_test).view({1,_n_in});
  single_input = single_input.to(device);
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back( single_input );
  //std::cout << 'Input: ' << inputs << std::endl;
  //std::cout << 'Input: '<< inputs[0] << std::endl;
  torch::Tensor output = _model.forward( inputs ).toTensor();
  //std::cout << 'Output: '<< output << std::endl;
  output = output.to(torch::kCPU);
  vector<float> cvs = this->tensor_to_vector (output);
  //std::cout << "CV: " << cvs << std::endl;
  _n_out=cvs.size();

//create components
  for(unsigned j=0; j<_n_out; j++) {
    string name_comp = "node-"+std::to_string(j);
    addComponentWithDerivatives( name_comp );
    componentIsNotPeriodic( name_comp );
  }

  //print log
  //log.printf("Pytorch Model Loaded: %s \n",fname);
  log.printf("Number of input: %d \n",_n_in);
  log.printf("Number of outputs: %d \n",_n_out);
  log.printf("  Bibliography: ");
  log<<plumed.cite("Bonati, Rizzi and Parrinello, J. Phys. Chem. Lett. 11, 2998-3004 (2020)");
  log<<plumed.cite("Trizio and Parrinello, J. Phys. Chem. Lett. 12, 8621-8626 (2021)");
  log.printf("\n");

}




void PytorchModelCuda::calculate() {

// retrieve arguments
vector<float> current_S(_n_in);
for(unsigned i=0; i<_n_in; i++)
  current_S[i]=getArgument(i);
//convert to tensor
torch::Tensor input_S = torch::tensor(current_S).view({1,_n_in}).to(device);
input_S.set_requires_grad(true);
//convert to Ivalue
std::vector<torch::jit::IValue> inputs;
inputs.push_back( input_S );
//calculate output
torch::Tensor output = _model.forward( inputs ).toTensor();


for(unsigned j=0; j<_n_out; j++) {  
auto grad_output = torch::ones({1}).expand({1, 1}).to(device);
auto gradient = torch::autograd::grad({output.slice(/*dim=*/1, /*start=*/j, /*end=*/j+1)},
                       {input_S},
    /*grad_outputs=*/ {grad_output},
    /*retain_graph=*/true,
    /*create_graph=*/false)[0]; // the [0] is to get a tensor and not a vector<at::tensor>

gradient = gradient.to(torch::kCPU);
vector<float> der = this->tensor_to_vector ( gradient ); //TODO check this 
string name_comp = "node-"+std::to_string(j);
    //set derivatives of component j
    for(unsigned i=0; i<_n_in; i++)
      setDerivative( getPntrToComponent(name_comp),i, der[i] ); //TODO maybe have to move to CPU
}




//set CV values
output = output.to(torch::kCPU);
vector<float> cvs = this->tensor_to_vector (output);
for(unsigned j=0; j<_n_out; j++) {
    string name_comp = "node-"+std::to_string(j);
    getPntrToComponent(name_comp)->set(cvs[j]);
  }




}






// get outputs and derivatives as vectors --> use tensor_to_vector
// everything needs to go back to the cpu now
// output = output.to(torch::kCPU);
// std::vector<float> out_vec = tensor_to_vector (output);
// gradient = gradient.to(torch::kCPU);
// std::vector<float> der = tensor_to_vector ( gradient );
// std::cout << "- Gradients " << j << ": \t" << der << std::endl;
// copy(der.begin(), der.end(), final_der[j].begin());
// }

// output = output.to(torch::kCPU);
// std::vector<float> out_vec = tensor_to_vector (output);

// std::cout << "" << std::endl;
// std::cout << "We can get our results!" << std::endl;
// std::cout << "- Input: \n\t" << input_test << std::endl;
// std::cout << "- Output: \n\t" << out_vec << std::endl;
// std::cout << "- Gradients: \n\t" << final_der << std::endl;
// std::cout << "- Gradients: \n";
// for (int i = 0; i < final_der.size(); i++) 
//     {
//       std::cout << "\t"<< final_der[i] << " \n";
//         }    
// std::cout << std::endl;
// return 0;
// }


} //PLMD
} //function
} //pytorch

#endif
