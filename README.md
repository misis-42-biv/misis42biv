# Запуск модели

1. Убедитесь, что у вас установлен Git LFS. Склонируйте куда-нибудь репозиторий.
2. `cd export`
3. `docker build . -t biv`
4. `docker run -it -v ./data:/data biv`. Вместо `./data` напишите путь к директории с данными. Файл со входными данными должен называться `input.tsv`, файл с выходными после работы контейнера будет называться `output.tsv`. 

# Обучение модели

Обучение выполняется на машине с GPU (RTX 3090 подойдёт).

1. Убедитесь, что у вас установлен Git LFS. Склонируйте куда-нибудь репозиторий.
2. Установите на машину pyenv и poetry.
3. Установите окружение: `poetry env use $(pyenv which python)`
4. Установите пакеты `poetry install`
5. Если вы хотите обучаться на одной GPU - пропустите этот шаг. Если вы хотите запускать распределённое обучение, то настройте accelerate: `poetry run accelerate config`. Настройка выполняется через удобный CLI-интерфейс.
6. Обучите LLM на размеченной выборке `poetry run accelerate launch -m biv.dev.train_llm`. Веса будут лежать в `checkpoint/train-vikhr`. 
7. Запустите процесс разметки неразмеченной выборки при помощи LLM `poetry run python -m biv.dev.label_llm --device cuda:0`. Вместо `cuda:0` напишите устройство, которое будет заниматься разметкой.
8. Обучите легковесную модель `poetry run accelerate launch -m biv.dev.train_encoder`. Веса будут лежать в `checkpoint/train-encoder`.
9. Экспортируйте легковесную модель в ONNX: `poetry run python -m biv.dev.export_encoder`. Модель будет лежать в `export/model/encoder.onnx`.
