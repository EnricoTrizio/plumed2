  #include <torch/torch.h>
  #include <torch/script.h>
  #include <torch/cuda.h>
  //#include <cuda_runtime.h>
  //#include <cudnn.h>
//   #include <ATen/ATen.h>

    int main() {

      // int numGPUs; 
      // cudaGetDeviceCount(&numGPUs);
      // cudaSetDevice(0); // use GPU0
      // std::cout <<"Found " << numGPUs <<" GPUs." << std::endl;

      torch::Device device(torch::kCPU);
      std::cout << "Device now is: " <<  device << std::endl;

      torch::Tensor tensor = torch::rand({2, 3});
      //torch::Tensor tensor = torch::rand({2, 3}, device); // one can also directly send to device
      std::cout << "Initialize CPU tensor." << std::endl;
      std::cout << tensor << std::endl;

      std::cout << " " << std::endl;
      std::cerr << "CUDA: " << torch::cuda::is_available() << std::endl;
      if (! torch::cuda::is_available()) {
        std::cout << "CUDA not available." << std::endl;
        return -1;
      } else{
        std::cout << "CUDA is available." << std::endl;
        std::cout << "Changing device to CUDA." << std::endl;
        torch::Device device(torch::kCUDA);
        std::cout << "Device now is: " << device << std::endl;
        tensor = tensor.to(device); // the tensor = .. is necessary, .to() is not enough
	std::cout << "Moving tensor from CPU to CUDA." << std::endl;
	std::cout << tensor << std::endl;
      }     
      return 0;
    }
