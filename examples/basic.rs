extern crate ml;
use ml::image_data::*;

fn main() {
    let images = load_in_training_images().unwrap();
    for i in 0..100 {
        output_image(&images, i);
    }
}
