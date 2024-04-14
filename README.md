# Simple Retrieval Augmented Generation (RAG) Pipeline

### Компоненты
- **ChromaDB**: используется в качестве векторного хранилища для эффективного извлечения контекста.
- **Mistral 7B**: LLM, используемая для генерации текста на основе контекста, предоставляемого ChromaDB.
- **BGE-Small-EN-v1.5**: Embeddings модель

## Инструкции по настройке

### Предварительные
- Питон 3.8 или выше
- pip
- Доступ к вычислительным ресурсам, способным запускать модели глубокого обучения (желательно с поддержкой CUDA для ускорения графического процессора).

### Монтаж
1. **Клонируйте репозиторий**
     ``` bash
     git clone https://github.com/AlisherAmirbek/simple-rag-pipeline.git
     CD simple-rag-pipeline
     ```
2. **Активируйте виртуальную среду в python**
2. **Установить зависимости Python**
     ``` bash
     pip install -r requirements.txt
     ```
3. **Установите переменные среды** (только для работы с GPU)
     ``` bash
     CMAKE_ARGS="-DLLAMA_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir --verbose
     ```

## Использование

Чтобы запустить пайплайн RAG, перейдите в корневой каталог проекта и выполните:
``` bash
python main.py
