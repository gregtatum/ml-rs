use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn process_text(file: &str) {
    let file = match File::open(file) {
        Ok(file) => file,
        Err(e) => {
            println!("Error opening file: {}", e);
            return;
        }
    };

    let reader = BufReader::new(file);
    let mut lines = reader.lines().peekable();

    while let Some(Ok(line)) = lines.peek() {
        if line.starts_with("@@") {
            break;
        } else {
            lines.next();
        }
    }

    let mut char_counts: HashMap<char, usize> = HashMap::new();
    let mut id_string = String::new();
    for raw_line in lines {
        let raw_line = raw_line.expect("Unable to get next line from the text.");

        // Start processing the string line
        // "@@1514 Albert of Prussia..."
        //  ^^
        if !raw_line.starts_with("@@") {
            println!("{}", raw_line);
            panic!("Line does not start with @@");
        }

        // "1514 Albert of Prussia..."
        let raw_line = &raw_line[2..];

        // Extract the number part of the string.
        // "@@1514 Albert of Prussia..."
        //    ^^^^
        id_string.clear();
        for ch in raw_line.chars() {
            if ch.is_ascii_digit() {
                id_string.push(ch);
            } else {
                break;
            }
        }
        let id = id_string.parse::<u32>().expect("Could not parse a string");

        // Skip the space.
        // "@@1514 Albert of Prussia..."
        //        ^
        if raw_line.as_bytes()[id_string.len()] != ' ' as u8 {
            panic!("Expected a space after the id");
        }
        // "Albert of Prussia..."
        let text = &raw_line[(id_string.len() + 1)..];
        for ch in text.chars() {
            *char_counts.entry(ch).or_insert(0) += 1;
        }
    }

    let mut char_vec: Vec<(char, usize)> = char_counts.into_iter().collect();
    char_vec.sort_by(|(_, count1), (_, count2)| count2.cmp(count1));

    for (ch, frequency) in char_vec {
        println!("{}: {}", ch, frequency);
    }
}

/// Ensure that the paths work when executing from the root directory or inside of
/// the text-embedding directory.
fn set_cwd() {
    let cwd = env::current_dir().unwrap();
    if cwd.file_name().unwrap() == "text-embedding" {
        // Go up a directory.
        env::set_current_dir(cwd.parent().unwrap()).unwrap();
    }
}

fn main() {
    set_cwd();
    process_text("./data/text/wiki-en.txt");
    println!("Done processing text.");
}
