# Text Embedding

This project is for understanding how text embedding works with machine learning models.

## Byte Pair Encoding

This implementation is based off of the paper [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909). In the paper it suggests taking an existing vocabulary, and augmenting it with the byte pair encodings as well. This implementation is a bit different in that it is purely algorithmic. This way I can experiment on arbitrary data sets, and don't need to find pre-existing dictionaries. I also changed it to compute the byte pair encodings over the frequency in the training corpus, not the frequency in the dictionary.

### Running

```
cargo run -p text-embedding
```

This takes in the information in [`data/text/`](../data/text) and outputs a `{LANG}-dictionary-symbols.txt` into the same folder. The text contains a number first, which is the frequency of that symbol, and then the symbol itself.

The top symbols for english are:

```
119911       the
59475        of
53413        in
50205        and
45409        a
36366        to
```

While a sampling further down includes many word parts, and complete words:

```
518          sol
518          england
518          sent
517          che
517          dies
516          cast
515          da
514          que
514          ved
510          ming
510          ric
510          director
```
