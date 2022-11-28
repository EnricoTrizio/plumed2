import mlcvs 
from mlcvs.models.nn import NeuralNetworkCV as NN
import torch
import numpy as np

device = torch.device('cuda')


model = NN([4,1])
model.to(device)

x = np.asarray([1.,2.,3.,4.])
x = torch.tensor(x,dtype=torch.float32,device=device)
q = np.asarray([1.,2.,3.,4.])
q = torch.tensor(q,dtype=torch.float32,device=device)
x.requires_grad = True
q.requires_grad = True
#x.to(device)
y = model(x)
z = model(x)
print(x,q,y,z)

# -- CALCULATE DERIVATIVES -- 
for yy in y:
    dy = torch.autograd.grad(yy, x,retain_graph=True )
    # -- PRINT -- 
    print('CV TEST')
#    print('n_input\t: {}'.format(input_size))
    print(f'x\t: {x}')
    print(f'cv\t: {yy}')
    print(f'der\t: {dy}')


model.export('./')
