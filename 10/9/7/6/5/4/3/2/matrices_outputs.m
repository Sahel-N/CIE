
function [sigmoid, sigmmoid, tanh] = matrices_outputs(input_matrix, m, n)
    % Calculates the outputs using bipolar sigmoid, its derivative, and tanh for each element in the input matrix.
    %
    % Inputs:
    %   input_matrix: Input matrix.
    %   m: Number of rows in the output matrix.
    %   n: Number of columns in the output matrix.
    %
    % Outputs:
    %   sigmoid: Output matrix using bipolar sigmoid.
    %   sigmmoid: Output matrix using the derivative of bipolar sigmoid.
    %   tanh: Output matrix using tanh.
    
    input_matrix = [0, 0; 0, 0]

    % [m, n] = size(input_matrix); another way to get the dimensions 

    % Initialize output matrices
    sigmoid = zeros(m, n);
    sigmmoid = zeros(m, n);
    tanh = zeros(m, n);

    % Iterate through each element in the input matrix
    for i = 1:m
        for j = 1:n
            sigmoid(i, j) = bipolar_sigmoid(input_matrix(i, j));
            sigmmoid(i, j) = bipolar_sigmoid_derivative(input_matrix(i, j));
            tanh(i, j) = tan_h(input_matrix(i, j));
        end
    end
end

function y = bipolar_sigmoid(x)
    y = -1 + 2/ (1 + exp(-x));
end

function dy = bipolar_sigmoid_derivative(x)
    y = bipolar_sigmoid(x);
    dy = 1/2 *(1 - y.^2);
end

function y = tan_h(x)
    y = (exp(x) - exp(-x)) / (exp(x) + exp(-x));
end
