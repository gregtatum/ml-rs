extern crate rand;
use self::rand::distributions::{IndependentSample, Range};
use image_data::Images;
use std::{cell::RefCell, f64::consts::E, iter::zip};

#[derive(Debug)]
pub struct Node {
    // The weights are edges in a graph that point back to the nodes in the previous
    // layer.
    weights: Vec<f64>,
    cost: f64,
    bias: f64,
    activation: RefCell<f64>,
}

/// A network layer. This can be the hidden layers and the output layer. The input
/// layer is static, and is loaded in from data.
/// The function for the activation is given as: a¹ = σ(Wa⁰ + b)
#[derive(Debug)]
pub struct Layer(Vec<Node>);

impl Layer {
    /// Creating a new node layer needs to reference the previous layer, as it will
    /// be used to compute the activations of each node in the layer.
    pub fn new(node_count: usize, previous_node_count: usize) -> Layer {
        let between = Range::new(-1f64, 1.0);
        let mut random = rand::thread_rng();
        let mut nodes = Vec::with_capacity(node_count);
        // Initialize all weights and nodes to values between -1 and 1.
        for _ in 0..node_count {
            nodes.push(Node {
                // The weights go from this layer to the previous, so have
                // one edge between this layer's nodes and the previous.
                weights: (0..previous_node_count)
                    // Initialize with a random -1 to 1 value.
                    .map(|_| between.ind_sample(&mut random))
                    .collect(),
                bias: between.ind_sample(&mut random),
                cost: 0.0,
                activation: RefCell::new(0.0),
            });
        }
        Layer(nodes)
    }

    pub fn len(&self) -> usize {
        return self.0.len();
    }

    pub fn iter(&self) -> impl Iterator<Item = &Node> + '_ {
        self.0.iter()
    }
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Node> + '_ {
        self.0.iter_mut()
    }
}

/// All of the data needed for a neural network implementation.
/// This is an implementation of:
/// https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
#[derive(Debug)]
pub struct Network {
    /// This network could be made more generic, but for now only operate on the images
    /// provided by mnist.
    pub images: Images,
    /// The layers include the hidden, and output layers. The input layer is provided
    /// by the images.
    pub layers: Vec<Layer>,
    // The pixel_count for the images.
    pub input_node_count: usize,
    /// The hidden layers are between the input and the output. They do the computation
    /// and their node counts are independent from the input and output node count.
    pub hidden_layer_count: usize,
    /// Assume all the hidden layers all of the same node count.
    pub hidden_node_count: usize,
    /// How many answers do we want? This is the output node count.
    pub output_node_count: usize,
}

impl Network {
    /// Creates a new network with the properly sized layers.
    pub fn new(
        images: Images,
        hidden_layer_count: usize,
        hidden_node_count: usize,
        output_node_count: usize,
    ) -> Network {
        let input_node_count = images.pixel_count;
        let mut layers = Vec::with_capacity(hidden_node_count + 2);
        let mut prev_node_count = 0;

        // Add the input layer.
        layers.push(Layer::new(input_node_count, prev_node_count));
        prev_node_count = input_node_count;

        // Add the hidden layers.
        for _ in 0..hidden_layer_count {
            layers.push(Layer::new(hidden_node_count, prev_node_count));
            prev_node_count = hidden_node_count;
        }

        // Add the output layer
        layers.push(Layer::new(output_node_count, prev_node_count));

        Network {
            images,
            layers,
            input_node_count,
            hidden_layer_count,
            hidden_node_count,
            output_node_count,
        }
    }

    /// Run the neural network using feed forward. For the implementation of the math,
    /// it would be better to use a linear algebra library, but for this didactic
    /// implementation, I'm doing the linear algebra myself.
    pub fn run(&self, image_index: usize) -> Vec<f64> {
        {
            let image_data = self.images.list.get(image_index).unwrap();
            let input_layer = self.layers.first().expect("Failed to get first layer.");

            for (node, pixel) in zip(input_layer.iter(), image_data) {
                // Images come in as u8 ranged 0-255, map them to f64 ranged 0-1.
                *node.activation.borrow_mut() = (*pixel as f64) / 255f64
            }
        }

        for window in self.layers.windows(2) {
            let input_layer = &window[0];
            let output_layer = &window[1];

            for node in output_layer.iter() {
                let mut multiplication_result = 0f64;

                for (input_node, weight) in zip(input_layer.iter(), &node.weights) {
                    multiplication_result += weight * *input_node.activation.borrow();
                }

                *node.activation.borrow_mut() = sigmoid(multiplication_result + node.bias);
            }
        }

        self.layers
            .last()
            .expect("Failed to get last layer.")
            .iter()
            .map(|node| *node.activation.borrow())
            .collect()
    }

    fn cost(&self, answer_index: usize) -> Vec<f64> {
        let mut answer_vec = vec![0.0; self.output_node_count];
        let answer_node = answer_vec
            .get_mut(answer_index)
            .expect("Network does not have enough output nodes for that answer");
        *answer_node = 1.0;

        let layer = self.layers.last().expect("Failed to get last layer.");
        let cost_vec = zip(layer.iter(), &answer_vec)
            .map(|(node, answer)| *answer - *node.activation.borrow())
            .collect::<Vec<f64>>();

        cost_vec
    }
}

/// Sigmoid functions keep the values ranged from -1 to 1. This particular function
/// is the logistic function given by: σ(x) = 1 / ( 1 + e^-x )
///
/// https://en.wikipedia.org/wiki/Sigmoid_function
fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + E.powf(value))
}

fn average_cost() {}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn network_layer() {
        let layer = Layer::new(3, 2);
        assert_eq!(layer.len(), 3, "There were three node weights created");
        for nodes in layer.0 {
            assert_eq!(
                nodes.weights.len(),
                2,
                "There were two weight edges created per node."
            );
        }
    }

    #[test]
    fn network() {
        let pixel_count = 4;
        let hidden_layer_count = 2;
        let hidden_node_count = 3;
        let output_node_count = 5;
        let network = Network::new(
            Images {
                dimensions: (2, 2),
                pixel_count: pixel_count,
                list: vec![
                    vec![0, 1, 2, 3],
                    vec![4, 5, 6, 7],
                    vec![8, 9, 10, 11],
                    vec![12, 13, 14, 15],
                ],
                labels: vec![0, 1, 2, 3],
            },
            hidden_layer_count,
            hidden_node_count,
            output_node_count,
        );

        assert_eq!(network.layers.len(), hidden_layer_count + 2);

        let input_layer = network.layers.get(0).unwrap();
        let hidden_layer_1 = network.layers.get(1).unwrap();
        let hidden_layer_2 = network.layers.get(2).unwrap();
        let output_layer = network.layers.get(3).unwrap();

        assert_eq!(input_layer.len(), pixel_count);
        assert_eq!(input_layer.0.get(0).unwrap().weights.len(), 0);

        assert_eq!(hidden_layer_1.len(), hidden_node_count);
        assert_eq!(hidden_layer_1.0.get(0).unwrap().weights.len(), pixel_count);

        assert_eq!(hidden_layer_2.len(), hidden_node_count);
        assert_eq!(
            hidden_layer_2.0.get(0).unwrap().weights.len(),
            hidden_node_count
        );

        assert_eq!(output_layer.len(), output_node_count);
        assert_eq!(
            output_layer.0.get(0).unwrap().weights.len(),
            hidden_node_count
        );

        assert_eq!(network.layers.get(4).is_none(), true);
    }

    #[test]
    fn feed_forward_test() {
        let pixel_count = 4;
        let network = Network::new(
            Images {
                dimensions: (2, 2),
                pixel_count,
                list: vec![
                    vec![0, 1, 2, 3],
                    vec![4, 5, 6, 7],
                    vec![8, 9, 10, 11],
                    vec![12, 13, 14, 15],
                ],
                labels: vec![0, 1, 2, 3],
            },
            2, // hidden layer count
            3, // hidden node count
            5, // output node count
        );
        let results = network.run(0);
        println!("results: {:?}", results);
    }
}
