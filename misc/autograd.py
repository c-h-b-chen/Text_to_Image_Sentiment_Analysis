# Practice using auotgradient for PyTorch

import torch

def main():
    # Create a tensor and set requires_grad=True to trach computation with it.
    x = torch.ones(2, 2, requries_grad=True)

    # Perform an operation on the tensor
    y = x + 2

    # Because y was created as a result of an operation, it has a grad_fn.
    print('y.grad_fn', y.grad_fn)

    z = y * y * 3
    out = z.mean()

    # Again, z, and out will have a grad_fn bc they were created with
    # operations.
    print(z, out)

    # Practice performing operations in place.
    a = torch.rand(3, 3)
    print('a.requires_grad', a.requires_grad)
    a.requires_grad_(True)
    print('a.requires_grad', a.requires_grad)

    # Now start calculating gradients.
    # This performs back propagation on the tensor.
    out.backward(torch.tensor(1))
    # equivalent to 
    # out.backward()

    # Print the gradient d(out)/dx
    print(x.grad)



if __name__ == '__main__':
    main()
