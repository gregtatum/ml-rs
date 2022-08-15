extern crate rand;
use self::rand::distributions::{IndependentSample, Range};
use image_data::Images;
use std::f64::consts::E;

#[derive(Debug)]
pub struct Node {
    // The weights are edges in a graph that point back to the nodes in the previous
    // layer.
    weights: Vec<f64>,
    bias: f64,
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
            });
        }
        Layer(nodes)
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
        // Initialize the hidden network layers plus 1 output layer.
        let layers = (0..(hidden_layer_count + 1))
            .map(|layer_index| {
                // The hidden layers all have the same node count in this design,
                // but the first hidden layer needs to compute the weights for
                // the inpute image.
                let previous_node_count = if layer_index == 0 {
                    images.pixel_count
                } else {
                    hidden_node_count
                };

                // The node count only changes for the last layer, which is the output
                // layer.
                let node_count = if layer_index == hidden_layer_count {
                    output_node_count
                } else {
                    hidden_node_count
                };

                Layer::new(node_count, previous_node_count)
            })
            .collect();

        Network {
            images,
            layers,
            hidden_layer_count,
            hidden_node_count,
            output_node_count,
        }
    }

    /// Run the neural network using feed forward. For the implementation of the math,
    /// it would be better to use a linear algebra library, but for this didactic
    /// implementation, I'm doing the linear algebra myself.
    pub fn run(&self, image_index: usize) -> Vec<f64> {
        let image_data = self.images.list.get(image_index).unwrap();

        let image_layer = image_data
            .iter()
            // Images come in as u8 ranged 0-255, map them to f64 ranged 0-1.
            .map(|i| (*i as f64) / 255f64)
            .collect();

        let mut hidden_layer_activations1 = Vec::with_capacity(self.hidden_node_count);
        let mut hidden_layer_activations2 = Vec::with_capacity(self.hidden_node_count);
        let mut output_activations = Vec::with_capacity(self.output_node_count);
        let mut ping_pong = 0;

        for (layer_index, layer) in self.layers.iter().enumerate() {
            // Select the activations buffer in and out by ping ponging between them.
            ping_pong = (ping_pong + 1) % 2;
            let (mut activations_in, mut activations_out) = if ping_pong == 0 {
                (&hidden_layer_activations1, &mut hidden_layer_activations2)
            } else {
                (&hidden_layer_activations2, &mut hidden_layer_activations1)
            };

            if layer_index == 0 {
                // This is the first hidden layer, use the initial image layer as input.
                activations_in = &image_layer;
            }

            if layer_index == self.layers.len() - 1 {
                // This is the last layer, use the output layer.
                activations_out = &mut output_activations;
            }

            // Clear any previous intermediate activations.
            activations_out.clear();

            for Node { weights, bias } in &layer.0 {
                let mut multiplication_result = 0f64;

                for (i, weight) in weights.iter().enumerate() {
                    let input_node = activations_in
                        .get(i)
                        .expect("Failed to get an input node value.");
                    multiplication_result += weight * input_node;
                }

                activations_out.push(sigmoid(multiplication_result + bias));
            }
        }
        output_activations
    }
}

/// Sigmoid functions keep the values ranged from -1 to 1. This particular function
/// is the logistic function given by: σ(x) = 1 / ( 1 + e^-x )
///
/// https://en.wikipedia.org/wiki/Sigmoid_function
fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + E.powf(value))
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn network_layer() {
        let layer = Layer::new(3, 2);
        assert_eq!(layer.0.len(), 3, "There were three node weights created");
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

        assert_eq!(network.layers.len(), hidden_layer_count + 1);

        let layer0 = network.layers.get(0).unwrap();
        let layer1 = network.layers.get(1).unwrap();
        let layer2 = network.layers.get(2).unwrap();

        assert_eq!(layer0.0.len(), hidden_node_count);
        assert_eq!(layer0.0.get(0).unwrap().weights.len(), pixel_count);

        assert_eq!(layer1.0.len(), hidden_node_count);
        assert_eq!(layer1.0.get(0).unwrap().weights.len(), hidden_node_count);

        assert_eq!(layer2.0.len(), output_node_count);
        assert_eq!(layer2.0.get(0).unwrap().weights.len(), hidden_node_count);

        assert_eq!(network.layers.get(3).is_none(), true);
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
