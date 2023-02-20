'''
Tensors
    - Inputs, Outputs and Learning weights are all in the form of Tensors
        - A multidimensional array with a lot of extra bells and whistles.
'''
import torch

def matrixes():
    z = torch.zeros(5,3)#This creates a 5 X 3 Matrix filled with Zeros.
    print(z)
    print(z.dtype)#Querying the datatype to find that zeros are 32-bit floating point numbers, the default with PyTorch
    '''
    One can ask for 16-bit integers instead
    '''
    i = torch.ones((5,3), dtype=torch.int16)
    print(i)

def randomWeightInitialization():
    '''
    It's common to initialize learning weights randomly, often with a specific seed for the PRNG for reproducibility of results
    '''
    torch.manual_seed(1729)
    r1 = torch.rand(2,2)
    print('A random tensor:')
    print(r1)

    r2 = torch.rand(2,2)
    print('\nA new different random tensor')
    print(r2)

    torch.manual_seed(1729)
    r3 = torch.rand(2,2)
    print('\nShould match r1:')
    print(r3)

def arithmeticWithPytorchTensors():
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

def MathematicalOperationsAvailableOnPyTorch():
    '''
    Mathematical operations available on PyTorch tensors
    '''
    # creating a random tensor and adjusting it's values between -1 and 1.
    r = torch.rand(2,2) - 0.5 * 2
    print('A random matrix, r:')
    print(r)

    #getting the absolute value of it to get all of them to turn positive.
    print("Get the absolute value.")
    print(torch.abs(r))

    #getting the inverse sine of it to get an angle back. Will compute and inverse sine of each element in r and return a new tensor of the same shape with the result
    print("Get the inverse sine.")
    print(torch.asin(r))

    # can do linear algebra operations like determinant and singular value decomposition 

    #determinant of r
    print("Get the determinant.")
    print(torch.det(r))

    #singular value decomposition of r
    print("Get the singular value decomposition.")
    print(torch.svd(r))

    # can do statistical and aggregate operations

    #will print out the average and standard deviation of r
    print("Get the average and standard deviation.")
    print(torch.std_mean(r))

    # will print out the maximum value of r
    print("Maximum value of the torch")
    print(torch.max(r))


'''
Introduction to AUTOGRAD
The automatic differentiation engine
'''
def aSimpleRecurrentNeuralNetwork():
    #A simple recurrent neural network(RNN). Starts with 4 tensors.
    x = torch.randn(1, 10)#The input
    prev_h = torch.randn(1, 20)#Hidden state of the RNN that gives it it's memory
    #Two sets of learning weights one for the input and the hidden state.
    W_h = torch.randn(20, 20)
    W_x = torch.randn(20, 10)
    #multipy the weights by their respective tensors.
    i2h = torch.mm(W_x, x.t())
    h2h = torch.mm(W_h, prev_h.t())

    #add the outputs of the two Matrix multiplications.
    next_h = i2h + h2h
    #pass the result through an activation function.
    next_h = next_h.tanh()

    #compute the loss for the output
    #the loss is the difference between the correct output and the actual output of the model
    loss = next_h.sum()
    #here is where you have to compute the derivatives of the loss.
    #with respect to every parameter of the model.
    #and use the gradients over the learning weights to decide how to adjust those weights in a way that reduces the loss.
    loss.backward()
    '''
    Gradient w.r.t the input Tensors is computed step-by-step from loss to top in reverse.
    '''


'''
BUILDING MODELS IN PYTORCH
'''
def initializeModel():
    import torch.nn as nn#Contains the neural network layers that we'll componse into the model
    import torch.nn.functional as F#Activation functions
    '''
    Lenet5 - earliest convulutional neural networks.
        Built to read small images of hand-written numbers and correctly classify which digit
        was represented in the image.
    Below we'll express Lenet5 in code.
    '''
    class LeNet(nn.Module):
        def __init__(self):
            super(LeNet, self).__init__()
            # 1 input image channel (black & white), 6 output channels, 3x3 square convolution
            # kernel
            self.conv1 = nn.Conv2d(1, 6, 3)
            self.conv2 = nn.Conv2d(6, 16, 3)
            # an affine operation: y = Wx + b
            self.fc1 = nn.Linear(16 * 6 * 6, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            # Max pooling over a (2, 2) window
            x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
            # If the size is a square you can only specify a single number 
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = x.view(-1, self.num_flat_features(x))
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

        def num_flat_features(self, x):
            size = x.size()[1:]#all dimensions except the batch dimension
            num_features = 1
            for s in size:
                num_features *= s
            return num_features

    net = LeNet()
    print(net)# what the object tells us about itself
    
    input = torch.rand(1, 1, 32, 32) # stand-in for a 32x32 black & white image
    print('\nImage batch shape')
    print(input.shape)

    output = net(input) #We don't call forward() directly
    print('\nRaw output:')
    print(output)
    print(output.shape)


#initializeModel()
