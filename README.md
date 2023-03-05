# ml-rs

# Feed Forward

This project is me playing around with neural network ideas from [3Blue1Brown's YouTube series](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi). It really doesn't do much yet.

```
cargo run --bin output-mnist-images
```

# Text Embedding

Explorations in text embedding.

## Notes

Activation - The number that each node in the neural network holds.
Weight - The value connecting each node to the next node in the next layer.
Hidden layer - The layers between the input and the output.

Activations should be ranged 0-1. A common function to do this is called the sigmoid function.

Logistics curve:
σ(x) = 1 / ( 1 + e^-x )

This is applied by:

Activation = σ(w1a1 + w2a2 + w3a3 + ... + wnan)

Then there is a bias added in. The weight tells you the general weight, and the bias applies how much this value should activate the neuron.

Activation = σ(w1a1 + w2a2 + w3a3 + ... + wnan - bias)
Feed forward:
  a¹ = σ(Wa⁰ + b)
  a² = σ(Wa¹ + b)

Learning: Finding the right weights and biases to solve your problem.

In order to train the neural network you need a cost function.

## Other helpful articles

* https://towardsdatascience.com/step-by-step-tutorial-on-linear-regression-with-stochastic-gradient-descent-1d35b088a843
* http://neuralnetworksanddeeplearning.com/

## Formulas:

## Back-propagating the weight

$$\Large
{\partial C     \over \partial w^L_{jk}}
=
{\partial C     \over \partial a^L_j}     \cdot
{\partial a^L_j \over \partial z^L_{jk}}  \cdot
{\partial z^L_j \over \partial w^L_{jk}}
$$

$$\large
\\=
2(a_j^L - y)                              \cdot
\sigma'(z^L_{jk})                         \cdot
a^L_j
$$

## Back-propagating the bias

$$\Large
{\partial C     \over \partial b^L_{jk}}
=
{\partial C     \over \partial a^L_j}     \cdot
{\partial a^L_j \over \partial z^L_{jk}}
$$

$$\large
\\=
2(a_j^L - y)                              \cdot
\sigma'(z^L_{jk})                         \cdot
$$

## Backpropagation


The cost function is defined as:

| Formula | Description |
| ------- | ----------- |
| $w$ | The weights vector for a layer.
| $y$ | The desired output vector for a layer.
| $b$ | The bias vector for a layer.
| $L$ | The current layer
| $L-1$ | The previous layer
| $\eta$ | The learning rate, eta, applied to $\Large \partial C_k \over \partial w^{L}$
| $ \sigma(x) = {\Large 1 \over {1 + e^{-x}}}$ | The [logistic](https://en.wikipedia.org/wiki/Logistic_function) [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) keeps the activations between -1 and 1.
| $z^{(L)} = w^{(L)}a^{(L-1)}+b^{(L)}$ | The interior of the activation function, without the sigmoid applied.
| $a^{(L)}=σ(z^{(L)})$ | The activation function requires a sigmoid, which keeps the values between -1 and 1.
| $a^{(L)}=σ(w^{(L)}a^{(L-1)}+b^{(L)})$ | The full activation function
| $C_0(...) = (a^L-y)^2$ | Cost
| $\Large {\partial C_0 \over \partial w^{(L)}} = {\partial z^{(L)} \over \partial w^{(L)}} {\partial a^{(L)} \over \partial z^{(L)}} {\partial C_0 \over \partial a^{(L)}}$ | The partial derivative of the cost with respect to the weight, determined using the chain rule.
| ${\Large \partial z^{(L)} \over \partial w^{(L)}} = a^{(L-1)}$ | Derivative of  $z^{(L)} = w^{(L)}a^{(L-1)}+b^{(L)}$
| ${\Large \partial a^{(L)} \over \partial z^{(L)}} = \sigma'(z^{(L)}) $ | Derivative of $\sigma(z^{(L)})$
| ${\Large \partial C_0 \over \partial a^{(L)}} = 2(a^{(L)} - y)$ | Derivative of  $(a^{(L)} - y)^2$
| ${\Large \partial C_0 \over \partial w^{(L)}} = a^{(L-1)} \sigma'(z^{(L)}) 2(a^{(L)} - y)$ | The formula for the derivative
| ${\Large \partial C_0 \over \partial w^{(L)}} = {\Large 1 \over n} \ \ {\Large \sum_{k=0}^{n-1}} \ \ {\Large \partial C_k \over \partial w^{(L)}}$ | Derivative of the full cost function is the average of all training examples.


## Applying the weights

The weight of the next training round $t + 1$:

$w^{(L)(t+1)} = w^{(L)(t)} - \eta {\Large \partial C \over \partial w^{L}}$

The weight of the next training round $t + 1$:

$b^{(L)}(t+1) = w^{(L)(t)} - \eta {\Large \partial C \over \partial b^{L}}$
