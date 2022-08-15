# Toy Rust Neural Network

This repo is me playing around with neural network ideas from [3Blue1Brown's YouTube series](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi). It really doesn't do much yet.

```
cargo run --example basic
```

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
