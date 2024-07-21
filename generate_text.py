import os
import torch
import numpy as np
import argparse
from dotenv import load_dotenv
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.metrics.pairwise import cosine_similarity

# Загрузка переменных окружения
load_dotenv()
model_path = os.getenv('MODEL_PATH')

# Загрузка токенизатора и модели
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2Model.from_pretrained(model_path)

# Проверка наличия GPU и перенос модели на GPU, если она доступна
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Добавление токена для заполнения
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def chunk_text_by_lines(text, lines_per_chunk=50):
    lines = text.split('\n')
    chunks = ['\n'.join(lines[i:i + lines_per_chunk]) for i in range(0, len(lines), lines_per_chunk)]
    return chunks

def get_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy())
    return embeddings

def embed_question(question):
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

def find_similar_chunks(question_vector, embeddings, chunks, top_n=5):
    similarities = cosine_similarity([question_vector], embeddings)[0]
    similar_indices = similarities.argsort()[-top_n:][::-1]
    similar_chunks = [chunks[i] for i in similar_indices]
    return similar_chunks

def main(file_path, question, top_n=5):
    text = read_file(file_path)
    chunks = chunk_text_by_lines(text)
    embeddings = get_embeddings(chunks)
    question_vector = embed_question(question)
    similar_chunks = find_similar_chunks(question_vector, embeddings, chunks, top_n)
    return similar_chunks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find similar chunks in a file based on a question.")
    parser.add_argument('file_path', type=str, help="Path to the file containing the text.")
    parser.add_argument('question', type=str, help="The question to find similar chunks for.")
    parser.add_argument('--top_n', type=int, default=10, help="Number of top similar chunks to return (default is 5).")

    args = parser.parse_args()
    
    similar_chunks = main(args.file_path, args.question, args.top_n)
    for idx, chunk in enumerate(similar_chunks):
        print(f"Similar chunk {idx}:\n{chunk}\n")
#import os
#import torch
#import numpy as np
#from dotenv import load_dotenv
#from transformers import GPT2Tokenizer, GPT2Model
#from sklearn.metrics.pairwise import cosine_similarity
#import logging
#import argparse
#import sys
#import io
#
## Настройка логирования
#logging.basicConfig(level=logging.INFO)
#logger = logging.getLogger(__name__)
#
## Изменение кодировки вывода на UTF-8
#sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
#sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
#
## Загрузка переменных окружения
#load_dotenv()
#model_path = os.getenv('MODEL_PATH')
#
## Проверка наличия переменной окружения
#if not model_path:
#    logger.error("MODEL_PATH environment variable not set")
#    exit(1)
#
## Загрузка токенизатора и модели
#try:
#    logger.info(f"Loading tokenizer and model from {model_path}")
#    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
#    model = GPT2Model.from_pretrained(model_path)
#except Exception as e:
#    logger.error(f"Error loading model: {e}")
#    exit(1)
#
## Проверка наличия GPU и перенос модели на GPU, если она доступна
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model.to(device)
#logger.info(f"Using device: {device}")
#
## Добавление токена для заполнения
#if tokenizer.pad_token is None:
#    tokenizer.pad_token = tokenizer.eos_token
#
#def read_file(file_path):
#    try:
#        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
#            text = file.read()
#        return text
#    except Exception as e:
#        logger.error(f"Error reading file {file_path}: {e}")
#        return ""  # Возвращать пустую строку в случае ошибки
#
#def chunk_text_by_lines(text, lines_per_chunk=50):
#    lines = text.split('\n')
#    chunks = ['\n'.join(lines[i:i + lines_per_chunk]) for i in range(0, len(lines), lines_per_chunk)]
#    return chunks
#
#def get_embeddings(chunks):
#    embeddings = []
#    for chunk in chunks:
#        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True).to(device)
#        with torch.no_grad():
#            outputs = model(**inputs)
#        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy())
#    return embeddings
#
#def embed_question(question):
#    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True).to(device)
#    with torch.no_grad():
#        outputs = model(**inputs)
#    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
#
#def find_similar_chunks(question_vector, embeddings, chunks, top_n=5):
#    similarities = cosine_similarity([question_vector], embeddings)[0]
#    similar_indices = similarities.argsort()[-top_n:][::-1]
#    similar_chunks = [chunks[i] for i in similar_indices]
#    return similar_chunks
#
#def main(directory_path, question, top_n=5):
#    logger.info(f"Reading files from directory: {directory_path}")
#    all_text = []
#    for root, _, files in os.walk(directory_path):
#        for file in files:
#            file_path = os.path.join(root, file)
#            all_text.append(read_file(file_path))
#    text = "\n".join(all_text)
#    logger.info("Splitting text into chunks")
#    chunks = chunk_text_by_lines(text)
#    logger.info(f"Total chunks created: {len(chunks)}")
#    logger.info("Generating embeddings for chunks")
#    embeddings = get_embeddings(chunks)
#    logger.info("Embedding question")
#    question_vector = embed_question(question)
#    logger.info("Finding similar chunks")
#    similar_chunks = find_similar_chunks(question_vector, embeddings, chunks, top_n)
#    return similar_chunks
#
#if __name__ == "__main__":
#    parser = argparse.ArgumentParser(description="Find similar chunks of text in a codebase")
#    parser.add_argument("directory_path", type=str, help="Path to the directory containing code files")
#    parser.add_argument("question", type=str, help="The question to find similar chunks for")
#    parser.add_argument("--top_n", type=int, default=5, help="Number of top similar chunks to return")
#    args = parser.parse_args()
#
#    similar_chunks = main(args.directory_path, args.question, args.top_n)
#    for idx, chunk in enumerate(similar_chunks):
#        logger.info(f"Similar chunk {idx}:\n{chunk}\n")
#        print(f"Similar chunk {idx}:\n{chunk}\n")
#
#
#import os
#import torch
#import numpy as np
#import argparse
#from dotenv import load_dotenv
#from transformers import GPT2Tokenizer, GPT2Model
#from sklearn.metrics.pairwise import cosine_similarity
#
## Загрузка переменных окружения
#load_dotenv()
#model_path = os.getenv('MODEL_PATH')
#
## Загрузка токенизатора и модели
#tokenizer = GPT2Tokenizer.from_pretrained(model_path)
#model = GPT2Model.from_pretrained(model_path)
#
## Проверка наличия GPU и перенос модели на GPU, если она доступна
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model.to(device)
#
## Добавление токена для заполнения
#if tokenizer.pad_token is None:
#    tokenizer.pad_token = tokenizer.eos_token
#
#def read_file(file_path):
#    with open(file_path, 'r', encoding='utf-8') as file:
#        text = file.read()
#    return text
#
#def read_all_files_in_directory(directory_path):
#    texts = []
#    for root, dirs, files in os.walk(directory_path):
#        for file in files:
#            if file.endswith('.js'):  # Фильтруем только файлы с расширением .js
#                file_path = os.path.join(root, file)
#                texts.append(read_file(file_path))
#    return texts
#
#def chunk_text_by_lines(text, lines_per_chunk=50):
#    lines = text.split('\n')
#    chunks = ['\n'.join(lines[i:i + lines_per_chunk]) for i in range(0, len(lines), lines_per_chunk)]
#    return chunks
#
#def get_embeddings(chunks):
#    embeddings = []
#    for chunk in chunks:
#        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True).to(device)
#        with torch.no_grad():
#            outputs = model(**inputs)
#        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy())
#    return embeddings
#
#def embed_question(question):
#    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True).to(device)
#    with torch.no_grad():
#        outputs = model(**inputs)
#    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
#
#def find_similar_chunks(question_vector, embeddings, chunks, top_n=5):
#    similarities = cosine_similarity([question_vector], embeddings)[0]
#    similar_indices = similarities.argsort()[-top_n:][::-1]
#    similar_chunks = [chunks[i] for i in similar_indices]
#    return similar_chunks
#
#def main(directory_path, question, top_n=5):
#    texts = read_all_files_in_directory(directory_path)
#    all_chunks = []
#    for text in texts:
#        all_chunks.extend(chunk_text_by_lines(text))
#    embeddings = get_embeddings(all_chunks)
#    question_vector = embed_question(question)
#    similar_chunks = find_similar_chunks(question_vector, embeddings, all_chunks, top_n)
#    return similar_chunks
#
#if __name__ == "__main__":
#    parser = argparse.ArgumentParser(description="Find similar chunks of code based on a given question.")
#    parser.add_argument('directory_path', type=str, help="Path to the directory containing the code files.")
#    parser.add_argument('question', type=str, help="The question to find similar chunks for.")
#    parser.add_argument('--top_n', type=int, default=5, help="Number of top similar chunks to return. Default is 5.")
#
#    args = parser.parse_args()
#
#    similar_chunks = main(args.directory_path, args.question, args.top_n)
#    for idx, chunk in enumerate(similar_chunks):
#        print(f"Similar chunk {idx}:\n{chunk}\n")
#
#
#
#import os
#import torch
#import numpy as np
#from dotenv import load_dotenv
#from transformers import GPT2Tokenizer, GPT2Model
#from sklearn.metrics.pairwise import cosine_similarity
#
## Загрузка переменных окружения
#load_dotenv()
#model_path = os.getenv('MODEL_PATH')
#
## Загрузка токенизатора и модели
#tokenizer = GPT2Tokenizer.from_pretrained(model_path)
#model = GPT2Model.from_pretrained(model_path)
#
## Проверка наличия GPU и перенос модели на GPU, если она доступна
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model.to(device)
#
## Добавление токена для заполнения
#if tokenizer.pad_token is None:
#    tokenizer.pad_token = tokenizer.eos_token
#
#def read_file(file_path):
#    with open(file_path, 'r', encoding='utf-8') as file:
#        text = file.read()
#    return text
#
#def chunk_text_by_lines(text, lines_per_chunk=50):
#    lines = text.split('\n')
#    chunks = ['\n'.join(lines[i:i + lines_per_chunk]) for i in range(0, len(lines), lines_per_chunk)]
#    return chunks
#
#def get_embeddings(chunks):
#    embeddings = []
#    for chunk in chunks:
#        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True).to(device)
#        with torch.no_grad():
#            outputs = model(**inputs)
#        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy())
#    return embeddings
#
#def embed_question(question):
#    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True).to(device)
#    with torch.no_grad():
#        outputs = model(**inputs)
#    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
#
#def find_similar_chunks(question_vector, embeddings, chunks, top_n=5):
#    similarities = cosine_similarity([question_vector], embeddings)[0]
#    similar_indices = similarities.argsort()[-top_n:][::-1]
#    similar_chunks = [chunks[i] for i in similar_indices]
#    return similar_chunks
#
#def main(file_path, question, top_n=5):
#    text = read_file(file_path)
#    chunks = chunk_text_by_lines(text)
#    embeddings = get_embeddings(chunks)
#    question_vector = embed_question(question)
#    similar_chunks = find_similar_chunks(question_vector, embeddings, chunks, top_n)
#    return similar_chunks
#
#if __name__ == "__main__":
#    file_path = "./codefiles/index.js"  # Укажите путь к вашему файлу с кодом
#    question = "How create function getMockSmtp() ?"  # Задайте ваш вопрос
#    top_n = 10  # Количество возвращаемых наиболее схожих чанков
#    similar_chunks = main(file_path, question, top_n)
#    for idx, chunk in enumerate(similar_chunks):
#        print(f"Similar chunk {idx}:\n{chunk}\n")
#
#
#import os
#import torch
#from dotenv import load_dotenv
#from transformers import GPT2Tokenizer, GPT2Model
#
## Загрузка переменных окружения
#load_dotenv()
#model_path = os.getenv('MODEL_PATH')
#
## Загрузка токенизатора и модели
#tokenizer = GPT2Tokenizer.from_pretrained(model_path)
#model = GPT2Model.from_pretrained(model_path)
#
## Добавление токена для заполнения
#if tokenizer.pad_token is None:
#    tokenizer.pad_token = tokenizer.eos_token
#
#def read_file(file_path):
#    with open(file_path, 'r', encoding='utf-8') as file:
#        text = file.read()
#    return text
#
#def chunk_text_by_lines(text, lines_per_chunk=50):
#    lines = text.split('\n')
#    chunks = ['\n'.join(lines[i:i + lines_per_chunk]) for i in range(0, len(lines), lines_per_chunk)]
#    return chunks
#
#def get_embeddings(chunks):
#    embeddings = []
#    for chunk in chunks:
#        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True)
#        with torch.no_grad():
#            outputs = model(**inputs)
#        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
#    return embeddings
#
#def main(file_path):
#    text = read_file(file_path)
#    chunks = chunk_text_by_lines(text)
#    embeddings = get_embeddings(chunks)
#    return embeddings
#
#if __name__ == "__main__":
#    file_path = "./codefiles/index.js"  # Укажите путь к вашему файлу с кодом
#    embeddings = main(file_path)
#    for idx, emb in enumerate(embeddings):
#        print(f"Embedding {idx}: {emb}")
#
#    # Сохранение эмбеддингов в файл (опционально)
#    import numpy as np
#    np.save("embeddings.npy", embeddings)
#