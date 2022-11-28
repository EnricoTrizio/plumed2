import torch
import torch.nn as nn


print(torch.__version__)

class test_NN(nn.Module):      # call as NN_DeeperLDA(nnLayersVector)
    def __init__(self): 
        super(test_NN, self).__init__() 
        modules = []
        modules.append(nn.Linear(8, 1))
        #modules.append(nn.ReLU(True))
        #modules.append(nn.Linear(4, 1))
        self.nn = nn.Sequential(*modules) 

    # perform a complete forward application of the NN
    def forward(self, x):
        # apply the hidden network on the input
        z = self.nn(x)
        return z

# -- DEFINE INPUT -- 
#random 
#x = torch.rand(input_size, dtype=torch.float32, requires_grad=True).unsqueeze(0)
#or by choosing the value(s) of the array
my_torch_cv = test_NN()
my_torch_cv.to(torch.device('cuda'))

x = torch.tensor([1.,2.,3.,4.,5.,6.,7.,8.], dtype=torch.float32, requires_grad=True)#, device='cuda')
x = x.to ( torch.device('cuda') )

# -- CALCULATE CV -- 
y = my_torch_cv(x)
y2 = my_torch_cv(x)

print(x,y,y2)

# -- CALCULATE DERIVATIVES -- 
for yy in y:
    dy = torch.autograd.grad(yy, x, create_graph=True)
    # -- PRINT -- 
    print('CV TEST')
#    print('n_input\t: {}'.format(input_size))
    print('x\t: {}'.format(x))
    print('cv\t: {}'.format(yy))
    print('der\t: {}'.format(dy))

# Compile via tracing
traced_cv   = torch.jit.trace ( my_torch_cv, example_inputs=x )
# Compile via scripting
scripted_cv = torch.jit.script( my_torch_cv )

filename='cuda_cv.pt'
traced_cv.save(filename)

# -- SAVE SERIALIZED FUNCTION -- 
filename='cuda_cv_scripted.pt'
scripted_cv.save(filename)
 
