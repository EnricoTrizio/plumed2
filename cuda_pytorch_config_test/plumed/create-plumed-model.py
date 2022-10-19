import torch
print(torch.__version__)

def my_torch_cv(x):
    '''
    Here goes the definition of the CV.

    Inputs:
        x (torch.tensor): input, either scalar or 1-D array
    Return:
        y (torch.tensor): collective variable (scalar)
    '''
    # CV definition
    y = x ** 2 #equivalent to torch.pow(x,2)

    return y

input_size = 2

# -- DEFINE INPUT -- 
#random 
#x = torch.rand(input_size, dtype=torch.float32, requires_grad=True).unsqueeze(0)
#or by choosing the value(s) of the array
x = torch.tensor([1.,2.], dtype=torch.float32, requires_grad=True, device='cuda')

# -- CALCULATE CV -- 
y = my_torch_cv(x)

# -- CALCULATE DERIVATIVES -- 
for yy in y:
    dy = torch.autograd.grad(yy, x, create_graph=True)
    # -- PRINT -- 
    print('CV TEST')
    print('n_input\t: {}'.format(input_size))
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
 
