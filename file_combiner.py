import os
import chardet

def read_file_with_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        if encoding is None:
            raise ValueError(f'Unable to detect encoding for file {file_path}')
        return raw_data.decode(encoding)

def combine_files(source_dir, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for root, _, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    content = read_file_with_encoding(file_path)
                    outfile.write(content)
                    outfile.write('\n')  # Добавим новую строку между содержимым файлов
                except Exception as e:
                    print(f'Error reading file {file_path}: {e}')

source_directory = './local-directory'
output_file_path = './output_file.txt'

combine_files(source_directory, output_file_path)

def remove_empty_lines(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    non_empty_lines = [line for line in lines if line.strip() != '']
    
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.writelines(non_empty_lines)

remove_empty_lines(output_file_path, output_file_path)