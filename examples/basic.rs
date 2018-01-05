extern crate term_painter;
extern crate byteorder;

use std::io;
use std::io::prelude::*;
use std::fs::File;
use std::convert::From;
use byteorder::{BigEndian, ReadBytesExt};
use term_painter::Color::*;

// Collect all potential error messages here:
#[derive(Debug)]
enum Error {
    Message(&'static str), // No reason to over-complicate with specific enums.
    IO(io::Error)
}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Error {
        Error::IO(err)
    }
}

type ImageData = Vec<u8>;

#[derive(Debug)]
struct Images {
    dimensions: (usize, usize),
    list: Vec<ImageData>,
    labels: Vec<u8>
}

fn read_in_images(path: &str) -> Result<Images, Error> {
    /*
     * According to: http://yann.lecun.com/exdb/mnist/
     *
     * [offset] [type]          [value]          [description]
     * 0000     32 bit integer  0x00000803(2051) magic number
     * 0004     32 bit integer  60000            number of images
     * 0008     32 bit integer  28               number of rows
     * 0012     32 bit integer  28               number of columns
     * 0016     unsigned byte   ??               pixel
     * 0017     unsigned byte   ??               pixel
     * ........
     * xxxx     unsigned byte   ??               pixel
     */
    match File::open(path) {
        Ok(ref mut file) => {
            let magic_number = file.read_i32::<BigEndian>()?;
            let number_of_images = file.read_i32::<BigEndian>()? as usize;
            let number_of_rows = file.read_i32::<BigEndian>()? as usize;
            let number_of_cols = file.read_i32::<BigEndian>()? as usize;
            let bytes_per_image = number_of_rows * number_of_cols;

            if magic_number != 2051 {
                return Err(Error::Message("The image data's magic number is not correct."));
            }

            let mut images = Vec::with_capacity(number_of_images as usize);
            for _ in 0..number_of_images {
                // Create the new vector.
                let mut image: Vec<u8> = Vec::with_capacity(bytes_per_image as usize);

                // Read the data in from the file.
                file
                    .take(bytes_per_image as u64)
                    .read_to_end(&mut image)?;

                // Double check that what we read in agrees with the header.
                if image.len() != bytes_per_image {
                    return Err(Error::Message("An image being read in was truncated and not the expected length"));
                }

                images.push(image);
            }

            Ok(Images {
                dimensions: (number_of_rows, number_of_cols),
                list: images,
                labels: Vec::new(),
            })
        },
        Err(err) => Err(Error::IO(err))
    }
}

fn read_in_labels(path: &str) -> Result<Vec<u8>, Error> {
    /*
     * According to: http://yann.lecun.com/exdb/mnist/
     *
     * [offset] [type]          [value]          [description]
     * 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
     * 0004     32 bit integer  60000            number of items
     * 0008     unsigned byte   ??               label
     * 0009     unsigned byte   ??               label
     * ........
     * xxxx     unsigned byte   ??               label
     */
    let mut file = File::open(path)?;

    // Get the header data.
    let magic_number = file.read_i32::<BigEndian>()?;
    let number_of_items = file.read_i32::<BigEndian>()? as usize;

    // Assert that the header makes sense.
    if magic_number != 2049 {
        return Err(Error::Message("The image data's magic number is not correct."));
    }

    // Slurp out the labels.
    let mut labels = Vec::with_capacity(number_of_items);
    file.read_to_end(&mut labels)?;

    // Verify the length
    if labels.len() != number_of_items {
        return Err(Error::Message("The data read in for labels is not the correct length as stated by the header."));
    }

    Ok(labels)
}

fn output_image(images: &Images, index: usize) -> String {
    let &Images {
        dimensions: (width, height),
        ref list,
        ref labels,
    } = images;

    let image = list.get(index).unwrap();
    let label = labels.get(index).unwrap();

    let string = String::new();

    for i in 0..height {
        for j in 0..width {
            let index = i * height + j;
            if *image.get(index).unwrap() > 50 {
                print!("X");
            } else {
                print!(".");
            }
        }
        println!();
    }

    println!("\n^ This image is labeled \"{:?}\"\n", label);

    string
}

fn load_in_test_images () -> Result<Images, Error> {
    let labels = read_in_labels("./data/t10k-labels-idx1-ubyte")?;
    let mut images = read_in_images("./data/t10k-images-idx3-ubyte")?;
    images.labels = labels;
    Ok(images)
}

fn load_in_training_images () -> Result<Images, Error> {
    let labels = read_in_labels("./data/train-labels-idx1-ubyte")?;
    let mut images = read_in_images("./data/train-images-idx3-ubyte")?;
    images.labels = labels;
    Ok(images)
}

fn main() {
    let images = load_in_training_images().unwrap();
    for i in 0..100 {
        output_image(&images, i);
    }
}
