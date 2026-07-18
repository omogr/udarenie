from udarenie import load_accentor
data_path = 'udarenie_data'
print('run_accentuator udarenie', data_path)
accentor = load_accentor(data_dir=data_path)

print( accentor.accentuate('Стены замка').to_annotated_text() )

