'''
Tensors
    - Inputs, Outputs and Learning weights are all in the form of Tensors
        - A multidimensional array with a lot of extra bells and whistles.
'''
import torch

z = torch.zeros(5,3)#This creates a 5 X 3 Matrix filled with Zeros.
#print(z)
#print(z.dtype)#Querying the datatype to find that zeros are 32-bit floating point numbers, the default with PyTorch
'''
One can ask for 16-bit integers instead
'''
i = torch.ones((5,3), dtype=torch.int16)
#print(i)



'''
It's common to initialize learning weights randomly, often with a specific seed for the PRNG for reproducibility of results
'''
torch.manual_seed(1729)
r1 = torch.rand(2,2)
#print('A random tensor:')
#print(r1)

r2 = torch.rand(2,2)
#print('\nA new different random tensor')
#print(r2)

torch.manual_seed(1729)
r3 = torch.rand(2,2)
#print('\nShould match r1:')
#print(r3)

'''
PyTorch tensors perform arithmetic operations intuitively. Tensors of similar shapes may be added, multiplied, etc.
Operations between scalars and a tensor are distributed over all the cells of the tensor
'''
ones = torch.ones(2,3)
print(ones)
twos = torch.ones(2,3) * 2 #every element is multiplied by two
print(twos)

threes = ones + twos 
print(threes)
print(threes.shape)

#below will give an error because there is no clean way to do element wise arithmetic operations between two tensors of different shapes.
r1 = torch.rand(2,3)
r2 = torch.rand(3,2)
r3 = r1 + r2 




