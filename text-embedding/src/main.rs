#![allow(dead_code)]
use crate::string_table::StringTable;
use rand::Rng;
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};

mod string_table;

type StringIndex = usize;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
struct Symbol(StringIndex);

impl Symbol {
    fn string<'a>(&self, string_table: &'a StringTable) -> &'a str {
        string_table.string(self.0)
    }

    fn index(&self) -> usize {
        self.0
    }
}

fn string_slice(string_table: &StringTable, symbols: &[Symbol]) -> String {
    let mut string = String::new();
    for symbol in symbols {
        string.push_str(symbol.string(&string_table));
    }
    string
}

fn process_text(file: &str, mut callback: impl FnMut(u32, &str)) {
    let file = match File::open(file) {
        Ok(file) => file,
        Err(e) => {
            println!("Error opening file: {}", e);
            return;
        }
    };

    let reader = BufReader::new(file);

    let mut id_string = String::new();
    for raw_line in line_iter(reader) {
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
        callback(id, text);
    }
}

/// Skips the preamble, and returns an iterator over the lines.
fn line_iter(reader: BufReader<File>) -> impl Iterator<Item = std::io::Result<String>> {
    let mut lines = reader.lines().peekable();

    while let Some(Ok(line)) = lines.peek() {
        if line.starts_with("@@") {
            break;
        } else {
            lines.next();
        }
    }
    lines.into_iter()
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

fn build_dictionary() -> Vec<(String, u32)> {
    println!("Read in dictionary");
    let mut dictionary: HashMap<String, u32> = HashMap::new();
    let mut word = String::new();
    process_text("./data/text/wiki-en.txt", |_, text| {
        word.clear();
        for ch in text.chars() {
            if ch.is_alphabetic() {
                word.push(ch);
            } else if !word.is_empty() {
                let count = dictionary
                    .entry(word.to_lowercase().to_string())
                    .or_insert(0);
                *count += 1;
                word.clear();
            }
        }
    });
    let mut dictionary: Vec<(String, u32)> = dictionary.into_iter().collect();
    dictionary.sort_by(|(_, count1), (_, count2)| count2.cmp(count1));

    // Print 10 random entries:
    println!("The dictionary has {} words", dictionary.len());
    print_dictionary_sample(&dictionary, 10);

    let mut count = 0;
    for (_, frequency) in &dictionary {
        if *frequency == 1 {
            count += 1;
        }
    }

    println!(
        "There are {} words with a frequency of 1, or {}%",
        count,
        ((count as f64 / dictionary.len() as f64) * 100.0) as u32
    );

    dictionary
}

fn print_dictionary_sample(dictionary: &Vec<(String, u32)>, count: usize) {
    println!("Sample of words in dictionary:");
    let mut rng = rand::thread_rng();
    let mut selection: Vec<usize> =
        std::iter::repeat_with(|| (rng.next_f64() * dictionary.len() as f64) as usize)
            .take(count)
            .collect();
    selection.push(0);
    selection.push(1);
    selection.push(2);
    selection.sort();

    for i in selection {
        let (word, frequency) = dictionary.get(i).unwrap();
        println!("  {}: {}", word, frequency);
    }
}

/// Count the number of characters in the entire text.
fn count_chars() {
    let mut char_counts: HashMap<char, usize> = HashMap::new();
    process_text("./data/text/wiki-en.txt", |_, text| {
        for ch in text.chars() {
            *char_counts.entry(ch).or_insert(0) += 1;
        }
    });
    let mut char_vec: Vec<(char, usize)> = char_counts.into_iter().collect();
    char_vec.sort_by(|(_, count1), (_, count2)| count2.cmp(count1));

    for (ch, frequency) in char_vec {
        println!("{}: {}", ch, frequency);
    }
}

fn build_symbols(
    dictionary: &Vec<(String, u32)>,
    iterations: usize,
) -> (Vec<(Symbol, u32)>, StringTable) {
    println!("Build symbols");
    let mut string_table = StringTable::new();

    let mut scratch_string = String::new();

    // Convert the dictionary to symbols.
    let mut dictionary: Vec<(Vec<Symbol>, u32)> = dictionary
        .iter()
        .map(|(word, frequency)| {
            let symbols: Vec<Symbol> = word
                .chars()
                .map(|ch| {
                    scratch_string.clear();
                    scratch_string.push(ch);
                    Symbol(string_table.index(&scratch_string))
                })
                .collect();
            (symbols, *frequency)
        })
        .collect();

    for _ in 0..iterations {
        // Count the symbol pairs.
        let mut symbol_pair_counts: HashMap<(Symbol, Symbol), u32> = HashMap::new();
        for (word, frequency) in &dictionary {
            for slice in word.windows(2) {
                let pair: (Symbol, Symbol) = (slice[0].clone(), slice[1].clone());
                *symbol_pair_counts.entry(pair).or_insert(0) += frequency;
            }
        }

        // Get the largest pair.
        let (pair_to_merge, _) = symbol_pair_counts
            .iter()
            .max_by_key(|(_, frequency)| *frequency)
            .unwrap();

        println!(
            "Merge: {}{}",
            pair_to_merge.0.string(&string_table),
            pair_to_merge.1.string(&string_table)
        );

        // Create the new merged symbol.
        let merged_symbol: Symbol = {
            let mut merged_string = pair_to_merge.0.string(&string_table).to_string();
            merged_string.push_str(pair_to_merge.1.string(&string_table));
            Symbol(string_table.take_string(merged_string))
        };

        // Merge the symbol in the dictionary.
        for (word, _) in &mut dictionary {
            let mut check_word = true;
            while check_word {
                check_word = false;
                for (index, slice) in word.windows(2).enumerate() {
                    if slice[0] == pair_to_merge.0 && slice[1] == pair_to_merge.1 {
                        check_word = true;
                        let mut new_word: Vec<Symbol> = Vec::new();
                        new_word.extend(&word[0..index]);
                        new_word.push(merged_symbol);
                        new_word.extend(&word[(index + 2)..]);
                        *word = new_word;
                        break;
                    }
                }
            }
        }
    }

    // Compute the final symbol frequency.
    let mut final_symbols = HashMap::new();
    for (symbols, frequency) in dictionary {
        for symbol in symbols {
            *final_symbols.entry(symbol).or_insert(0) += frequency;
        }
    }

    // Return the results sorted.
    let mut final_symbols: Vec<(Symbol, u32)> = final_symbols.into_iter().collect();
    final_symbols.sort_by(|(_, a), (_, b)| b.cmp(a));

    (final_symbols, string_table)
}

// Reports on the symbol pairs.
fn report_symbol_pairs(
    symbol_pair_counts: &HashMap<(Symbol, Symbol), u32>,
    string_table: &StringTable,
) {
    // Sort the pairs by frequency.
    let mut sorted: Vec<(&(Symbol, Symbol), &u32)> = symbol_pair_counts.iter().collect();
    sorted.sort_by(|(_, a), (_, b)| b.cmp(a));

    println!("Symbol pairs count: {:?}", sorted.len());
    for ((a, b), frequency) in &sorted[0..50] {
        println!(
            "  \"{}{}\" - {}",
            a.string(&string_table),
            b.string(&string_table),
            frequency
        );
    }
}

fn main() {
    set_cwd();
    // count_chars();
    let dictionary = build_dictionary()
        .into_iter()
        // Don't use things with only 1 entry.
        .filter(|entry| entry.1 > 1)
        .collect();
    let (symbols, string_table) = build_symbols(&dictionary, 5000);

    // Report an write out the symbols.
    println!("Symbols for the language: {}", symbols.len());
    for (symbol, count) in &symbols {
        println!("  {} - {:?}", count, symbol.string(&string_table));
    }

    {
        let path = "./data/text/en-dictionary-symbols.txt";
        println!("Writing out results to: {}", path);
        let mut file = File::create(path).unwrap();
        for (symbol, count) in &symbols {
            writeln!(file, "{: <12} {}", count, symbol.string(&string_table)).unwrap();
        }
    }

    println!("Done processing text.");
}
