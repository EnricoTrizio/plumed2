import mlcvs 
from mlcvs.models.nn import NeuralNetworkCV as NN
import torch
import numpy as np

torch.manual_seed(42)

model_name = '_4_2' 
model = NN([4,2])
log_file = open(f'derivatives{model_name}.txt', 'w')

def double_print(text="", log_file= log_file):
    print(text)
    print(text, file=log_file)

double_print('Test with simple square model..')

def func(x):
    return torch.pow(x,2)#.sum(dim=1)

x = torch.Tensor([1,2,3,4,5])
double_print(f'Input: {x} {x.shape}')
x.requires_grad = True
y = func(x)
double_print(f'Output: {y} {y.shape}')

double_print('Derivatives. ')
grad_output = torch.ones_like(y)
for yy in y:
    go = torch.ones_like(yy)
    g = torch.autograd.grad(yy,x,grad_outputs=go, retain_graph=True)
    double_print(f'{g}')


double_print()
double_print('Test with simple NN..')

for s in ['cpu', 'cuda']:
    double_print(f'Test with {s}..')
    device = torch.device(s)

    model.to(device)

    x = np.asarray([1.,2.,3.,4.])
    x = torch.tensor(x,dtype=torch.float32,device=device)
    x.requires_grad = True
    #x.to(device)
    y = model(x)
    double_print(y)
    # -- CALCULATE DERIVATIVES -- 
    for yy in y:
        go = torch.ones_like(yy)
        g = torch.autograd.grad(yy,x,grad_outputs=go, retain_graph=True)
        double_print(f"{g}")
        
    #     dy = torch.autograd.grad(yy, x,retain_graph=True )
    #     # -- PRINT -- 
    #     print('CV TEST')
    # #    print('n_input\t: {}'.format(input_size))
    #     print(f'x\t: {x}')
    #     print(f'cv\t: {yy}')
    #     print(f'der\t: {dy}')
    double_print()


print('Export model..')
traced_cv   = torch.jit.trace ( model, example_inputs=x )
filename=f'cuda_cv{model_name}.pt'
traced_cv.save(filename)
double_print(f'Model exported as cuda_cv{model_name}.pt')
