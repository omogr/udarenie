
# Udarenie

Russian text accentuation library.

## Features

- Dictionary-based accentuation for unambiguous words
- BERT-based neural disambiguation for homographs
- Morphological enhancement for improved accuracy
- SSML tag support

## Installation


```bash
pip install git+https://github.com/omogr/udarenie.git
```


## Usage

```python

from udarenie import load_accentor

accentor = load_accentor(data_dir='data_plus')
print( accentor.accentuate('Стены замка').to_annotated_text() )

```

## Data

The library requires model data which is downloaded automatically on first use.



