from udarenie import load_accentor
data_path = '../data_plus'
print('run_accentuator udarenie', data_path)

user_dict = {
  "гурбуртур": "Гурб+уртур"
}

accentor = load_accentor(data_dir=data_path, user_dict=user_dict)

print( accentor.accentuate('Стены замка Гурбуртур').to_annotated_text() )

