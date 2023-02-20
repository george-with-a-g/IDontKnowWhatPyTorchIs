# Introduction to PyTorch
## Relevant knowledge

**_Matrix_**
- an array of numbers, symbols or expressions arranged in rows and columns.
- can be used to represent linear transformations, such as rotations or scaling, and to solve systems of linear equations.
- they can be added and subtracted, multiplied by scalars and multiplied by other matrices.
- the product of two matrices is defined only if the number of columns in the first matrix is equal to the number of rows in the second matrix.

**_Scalar_**
- In linear algebra a scalar is a single number as opposed to a matrix or a vector which are arrays of numbers.
- It can be used to scale a matrix or a vector.
- Here's an example of a 2x2 matrix multiplied by the scalar 2.
- The 2x2 matrix
| 1 2 |
| 3 4 |
- The 2x2 matrix multiplied by the scalar
| 2 4 |
| 6 8 |

**_Standard deviation_**
- The standard deviation of a matrix is a measure of the amount of variation or dispersion in the values of the matrix elements.
- To calculate the standard deviation of a matrix:
    - You first calculate the mean of the matrix elements,
    - Calculate the sum of the squared differences between each element and the mean.
    - This sum is then divided by the number of elements in the matrix minus one,
    - The square root of the result is taken to obtain the standard deviation.
- To calculate in PyTorch.
    ```
    print(torch.std_mean(r))#Will print out the standard deviation and the mean.
    ```

**_Determinant of a Matrix_**

**_Singular Value Decomposition of a Matrix_**

