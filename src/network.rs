extern crate rand;
use image_data::{Images};
use std::f64::consts::E;
use self::rand::distributions::{IndependentSample, Range};

/**
 * This is a bit of a custom solution, and could be made much shorter with
 * a proper linear algebra library.
 */
pub fn feed_forward (network: &Network, image_index: usize) -> Vec<f64> {
    let image_data = network.images.list.get(image_index).unwrap();

    let image_buffer = image_data.iter()
        // Images come in as u8 ranged 0-255, map them to f64 ranged 0-1.
        .map(|i| (*i as f64) / 255f64).collect();

    let mut hidden_layers_buffer1 = Vec::with_capacity(network.hidden_node_count);
    let mut hidden_layers_buffer2 = Vec::with_capacity(network.hidden_node_count);
    let mut final_results_buffer = Vec::with_capacity(network.output_node_count);
    let mut buffer_ping_pong = 0;

    for (layer_index, layer) in network.layers.iter().enumerate() {
        // Select the buffer in and out by ping ponging between them.
        buffer_ping_pong = (buffer_ping_pong + 1) % 2;
        let (buffer_in, mut buffer_out) = if buffer_ping_pong == 0 {
            (&hidden_layers_buffer1, &mut hidden_layers_buffer2)
        } else {
            (&hidden_layers_buffer2, &mut hidden_layers_buffer1)
        };

        // Figure out the target buffers for the results.
        let previous_result = if layer_index == 0 {
            &image_buffer
        } else {
            buffer_in
        };
        let mut next_result = if layer_index == network.layers.len() - 1 {
            &mut final_results_buffer
        } else {
            buffer_out
        };

        // Reset the buffer to be empty.
        next_result.clear();

        for next_node_index in 0..layer.node_count {
            let weights = layer.node_weights.get(next_node_index).unwrap();
            let bias = layer.node_biases.get(next_node_index).unwrap();
            let mut result = 0f64;

            for previous_node_index in 0..weights.len() {
                let weight = weights.get(previous_node_index).unwrap();
                let node_value = previous_result.get(previous_node_index).unwrap();
                result += weight * node_value;
            }

            next_result.push(result + bias);
        }
    }
    final_results_buffer
}

#[derive(Debug)]
pub struct NetworkLayer {
    pub node_weights: Vec<Vec<f64>>,
    pub node_biases: Vec<f64>,
    pub node_count: usize
}

#[derive(Debug)]
pub struct Network {
    pub images: Images,
    pub layers: Vec<NetworkLayer>,
    pub hidden_layer_count: usize,
    pub hidden_node_count: usize,
    pub output_node_count: usize
}

impl NetworkLayer {
    pub fn new(
        node_count: usize,
        previous_node_count: usize
    ) -> NetworkLayer {
        let between = Range::new(-1f64, 1.0);
        let mut random = rand::thread_rng();

        NetworkLayer {
            // Start building out the node weights, one set per node
            // in this layer.
            node_weights: (0..node_count).map(|_| {
                // The weights go from this layer to the previous, so have
                // one edge between this layer's nodes and the previous.
                (0..previous_node_count)
                    // Initialize with a random -1 to 1 value.
                    .map(|_| between.ind_sample(&mut random))
                    .collect()
            }).collect(),

            // There is one bias per node in a layer.
            node_biases: (0..node_count)
                // Initialize with a random -1 to 1 value.
                .map(|_| between.ind_sample(&mut random))
                .collect(),

            node_count: node_count
        }
    }
}

impl Network {
    pub fn new(
        images: Images,
        hidden_layer_count: usize,
        hidden_node_count: usize,
        output_node_count: usize
    ) -> Network {
        // Initialize the NetworkLayers
        let layers = (0..(hidden_layer_count + 1)).map(|layer_index| {
            // Adjust the previous nodes length.
            let previous_node_count = if layer_index == 0 {
                images.pixel_count
            } else {
                hidden_node_count
            };

            // Adjust the node count to handle both hidden layers, and
            // the output layer.
            let node_count = if layer_index == hidden_layer_count {
                output_node_count
            } else {
                hidden_node_count
            };

            NetworkLayer::new(
                node_count,
                previous_node_count
            )
        }).collect();

        Network {
            images: images,
            layers: layers,
            hidden_layer_count: hidden_layer_count,
            hidden_node_count: hidden_node_count,
            output_node_count: output_node_count
        }
    }

}

fn sigmoid (value: f64) -> f64 {
    1.0 / (1.0 + E.powf(value))
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn network_layer () {
        let layer = NetworkLayer::new(3, 2);
        assert_eq!(layer.node_weights.len(), 3, "There were three node weights created");
        for weights in layer.node_weights {
            assert_eq!(weights.len(), 2, "There were two weight edges created per node.");
        }
        assert_eq!(layer.node_biases.len(), 3, "There were three node biases created");
    }

    #[test]
    fn network () {
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
                labels: vec![0, 1, 2, 3]
            },
            hidden_layer_count,
            hidden_node_count,
            output_node_count
        );

        assert_eq!(network.layers.len(), hidden_layer_count + 1);

        let layer0 = network.layers.get(0).unwrap();
        let layer1 = network.layers.get(1).unwrap();
        let layer2 = network.layers.get(2).unwrap();

        assert_eq!(layer0.node_weights.len(), hidden_node_count);
        assert_eq!(layer0.node_biases.len(), hidden_node_count);
        assert_eq!(layer0.node_weights.get(0).unwrap().len(), pixel_count);

        assert_eq!(layer1.node_weights.len(), hidden_node_count);
        assert_eq!(layer1.node_biases.len(), hidden_node_count);
        assert_eq!(layer1.node_weights.get(0).unwrap().len(), hidden_node_count);

        assert_eq!(layer2.node_weights.len(), output_node_count);
        assert_eq!(layer2.node_biases.len(), output_node_count);
        assert_eq!(layer2.node_weights.get(0).unwrap().len(), hidden_node_count);


        assert_eq!(network.layers.get(3).is_none(), true);
    }

    #[test]
    fn feed_forward_test () {
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
                labels: vec![0, 1, 2, 3]
            },
            hidden_layer_count,
            hidden_node_count,
            output_node_count
        );
        let results = feed_forward(&network, 0);
        println!("results: {:?}", results);
    }
}
