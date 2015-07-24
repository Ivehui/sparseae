function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1));
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
%

m=size(data,2);
%feedforward
net.a{1}=data;%64*10000
net.a{2}=sigmoid(bsxfun(@plus,W1*net.a{1},b1));%25*10000
net.a{3}=sigmoid(bsxfun(@plus,W2*net.a{2},b2));%64*10000

%
[SUM_KL, de_KL]=KL_diver(net.a{2},sparsityParam);%sparsity penalty

deW_KL=bsxfun(@times,   ...%25*64
              (net.a{2}.*(1-net.a{2}))*net.a{1}'/m, ...
              de_KL);
deb_KL=de_KL.*sum(net.a{2}.*(1-net.a{2}),2) /m;%25*1
%cost function Jsparse
cost=1/m/2*sum(sum((net.a{3}-net.a{1}).^2))  ...     %squared error cost
    +lambda/2*(sum(sum(W1.^2))+sum(sum(W2.^2)))  ... %weight decay term
    +beta*SUM_KL;          %sparsity penalty

%backpropagation
net.delta{3}= -(net.a{1}-net.a{3})  .*  net.a{3}  .*  (1-net.a{3});%64*10000
net.delta{2}= W2' * net.delta{3}    .*  net.a{2}  .*  (1-net.a{2});%25*10000  %squared error delta
% net.delta{2}= bsxfun(@plus, ...   %25*10000
%              W2' * net.delta{3}     .*  net.a{2}  .*  (1-net.a{2}) ... %squared error delta
%              ,beta*de_KL);%sparsity penalty
%gradient descent
W2grad= 1/m * net.delta{3} * net.a{2}' ...%64*25 %squared error grad
      + lambda * W2;%weight decay term
b2grad= sum(net.delta{3},2) /m;%64*1
W1grad= 1/m * net.delta{2} * net.a{1}' ...%25*64 %squared error grad
      + beta*deW_KL   ...%sparsity penalty
      + lambda * W1;%weight decay term
% W1grad= 1/m * net.delta{2} * net.a{1}' ...%25*64 %squared error grad
%       + lambda * W1;%weight decay term
b1grad= sum(net.delta{2},2) /m + beta*deb_KL;%25*1

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

%sparsity penalty
%SUM_KL:sum of KL
%de_KL :derivative of KL
function [SUM_KL, de_KL]= KL_diver(a2,rhoParam)
    rho=sum(a2,2)/size(a2,2);
    SUM_KL=sum(rhoParam*log(rhoParam./rho)+(1-rhoParam)*log((1-rhoParam)./(1-rho)));
    de_KL =(-rhoParam./rho+(1-rhoParam)./(1-rho));
end
% SUM_KL=sum(rho.*log(rho/rhoParam)+(1-rho).*log((1-rho)/(1-rhoParam)));
%     de_KL =-rho/rhoParam + (1-rho)/(1-rhoParam);