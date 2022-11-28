  #include <torch/torch.h>
  #include <torch/script.h>
  #include <torch/cuda.h>

    int main() {
      torch::Tensor tensor = torch::rand({2, 3});
      torch::Device device = torch::kCUDA;
//      tensor.to(device);

      std::cerr << "CUDA: " << torch::cuda::is_available() << std::endl;
      if (! torch::cuda::is_available()) {
        std::cout << "CUDA not available." << std::endl;
        return -1;
      } else{
        std::cout << "CUDA is available." << std::endl;
        device = torch::kCUDA;
        tensor.to(device);
      }     
      return 0;
    }
