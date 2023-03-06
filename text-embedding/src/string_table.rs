pub type StringIndex = usize;

/// Stores a unique list of strings, so that strings can be operated up via stable
/// indexes. This makes for cheap comparisons and storage of references to strings.
pub struct StringTable {
    strings: Vec<String>,
}

impl StringTable {
    /// Create a new StringTable.
    pub fn new() -> StringTable {
        StringTable {
            strings: Vec::new(),
        }
    }

    /// Interns a string if it exists, and returns the index. Otherwise it discards the
    /// string and returns an existing index. O(n)
    pub fn take_string(&mut self, string: String) -> StringIndex {
        match self.strings.iter().position(|s| *s == string) {
            Some(index) => index,
            None => {
                let index = self.strings.len();
                self.strings.push(string);
                index
            }
        }
    }

    /// Gets an index of the string. If the string doesn't exist, a copy is interned.
    pub fn index(&mut self, string: &String) -> StringIndex {
        match self.strings.iter().position(|s| s == string) {
            Some(index) => index,
            None => {
                let index = self.strings.len();
                self.strings.push(string.to_string());
                index
            }
        }
    }

    /// Returns a string from an index.
    pub fn string(&self, index: StringIndex) -> &str {
        match self.strings.get(index) {
            Some(string) => &**string,
            None => "",
        }
    }
}
