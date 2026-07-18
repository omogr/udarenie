
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

print('run_accentuator udarenie', args.data_path)
accentor = load_accentor(data_dir=Path(args.data_path))

print( accentor.accentuate('Стены замка') )

```

## Data

The library requires model data which is downloaded automatically on first use.



