# Russian subtitles dataset

Проект содержит уже обработанный и готовый к использованию [набор данных](https://github.com/Desklop/Russian_subtitles_dataset/blob/master/data/subtitles_ru.txt.zip) (запакован в .zip для уменьшения размера) из русских субтитров к 347 различным сериалам и обученную на этом наборе данных языковую [модель](https://github.com/Desklop/Russian_subtitles_dataset/blob/master/data/jamspell_ru_model_subtitles.bin.zip) (запакована в .zip для уменьшения размера) для корректировщика опечаток [JamSpell](https://github.com/bakwc/JamSpell). Так же имеется небольшой скрипт на Python для обработки исходного набора данных, взятого из корпуса [Taiga](https://tatianashavrina.github.io/taiga_site/downloads).

**Внимание!** Перед использованием, не забудьте распаковать из .zip [набор данных](https://github.com/Desklop/Russian_subtitles_dataset/blob/master/data/subtitles_ru.txt.zip) и языковую [модель](https://github.com/Desklop/Russian_subtitles_dataset/blob/master/data/jamspell_ru_model_subtitles.bin.zip) :)

# О наборе данных

За основу был взят корпус [Taiga](https://tatianashavrina.github.io/taiga_site/downloads), а точнее - корпус из всех субтитров. Он представляет собой архив, содержащий субтитры к 347 сериалам на различных языках (русский, английский, испанский и т.д.). Они находятся в архиве под кнопкой "All subtitles" из раздела "Our special collections for" по пути `/home/tsha/Subtitles/texts`.

Из этих субтитров были оставлены субтитры только на русском языке (с некоторыми изменениями скрипта для обработки исходного набора данных можно получить субтитры на других доступных языках), которые затем были обработаны и собраны в один файл - [`subtitles_ru.txt`](https://github.com/Desklop/Russian_subtitles_dataset/blob/master/data/subtitles_ru.txt.zip) (запакован в .zip для уменьшения размера).

Этот набор субтитров можно использовать для:
 - построения модели [word2vec](https://radimrehurek.com/gensim/models/word2vec.html) (предусмотрено в скрипте для обработки исходного набора данных)
 - создания языковой модели для [JamSpell](https://github.com/bakwc/JamSpell) (предусмотрено в скрипте для обработки исходного набора данных)
 - обучения нейронной сети
 - обучения своего чат-бота (т.к. все диалоги были правильно разобраны и исправлены, что бы на одной строке была законченная фраза только одного актёра)
 - расширения словаря в какой-либо задаче обработки естественного языка (например, классификация текста, определение эмоционального окраса)
 - или в любой другой задаче обработки естественного языка (natural language processing - NLP)

# Скрипт для обработки исходного набора данных

Скрипт реализован на языке Python, пока что без поддержки многопоточности (т.е. обработка выполняется на одном ядре). Исходный код находится в файле [`preprocessing.py`](https://github.com/Desklop/Russian_subtitles_dataset/blob/master/preprocessing.py) (для запуска необходимо вызвать метод `prepare_subtitles()` класса `Preparation`). Скрипт выполняет следующие операции:
 - поиск и сохранение в отдельную папку только русских субтитров в исходном наборе субтитров
 - очистка и исправление найденных русских субтитров
 - разбиение каждого предложения на отдельные слова и приведение их к одной длине (для возможности построения модели word2vec)
 - обучение языковой модели для корректировщика опечаток [JamSpell](https://github.com/bakwc/JamSpell#train)
 - построение гистограммы размеров предложений
 - предварительная обработка предложения 'на лету' (с использованием JamSpell, будет полезно при создании своего чат-бота, классификации текста или любой другой задаче NLP)

Скрипт для своей работы требует:
1. Для Python3.6: Matplotlib, NumPy, JamSpell.
2. Для Ubuntu: swig, cmake, python3, python3-pip, git.

Если вы используете Ubuntu 18.04, для установки всех пакетов можно воспользоваться [`install_packages.sh`](https://github.com/Desklop/Russian_subtitles_dataset/blob/master/install_packages.sh) (проверено в Ubuntu 18.04).

**Внимание!** В процессе обучения языковой модели для JamSpell скрипту необходимо **15 Гб** оперативной памяти. Если у вас столько нету, вы можете это исправить с помощью [swap-файла](https://ru.wikipedia.org/wiki/%D0%9F%D0%BE%D0%B4%D0%BA%D0%B0%D1%87%D0%BA%D0%B0_%D1%81%D1%82%D1%80%D0%B0%D0%BD%D0%B8%D1%86). При всех остальных операциях пиковое потребление оперативной памяти около 6 Гб.

Для упрощения работы со скриптом, он поддерживает аргументы командной строки, которые имеют следующую структуру: `[ключ] имя_папки/файла`.

Возможные комбинации аргументов командной строки и их описание:
1. без аргументов - обработка исходных субтитров с учётом промежуточных результатов от прошлых запусков (используются пути по умолчанию). Например: `python3 preprocessing.py`
2. `foldername` - задать имя папки с исходными субтитрами (должна содержать вложенные папки с `.txt` файлами субтитров). Например: `python3 preprocessing.py data/subtitles`
3. `modelname` - задать имя `.bin` файла для сохранения языковой модели для JamSpell. Например: `python3 preprocessing.py data/model.bin`
4. `foldername modelname` - задать имя папки с исходными субтитрами и имя `.bin` файла для сохранения языковой модели для JamSpell. Например: `python3 preprocessing.py data/subtitles data/model.bin`
5. `-f foldername` - обработка всех исходных субтитров из `foldername` без учёта промежуточных результатов. Например: `python3 preprocessing.py -f data/subtitles`
6. `-f modelname` - обработка всех исходных субтитров без учёта промежуточных результатов и сохранение языковой модели для JamSpell в `modelname`. Например: `python3 preprocessing.py -f data/model.bin`
7. `-f foldername modelname` - обработка всех исходных субтитров из `foldername` без учёта промежуточных результатов и сохранение языковой модели для JamSpell в `modelname`. Например: `python3 preprocessing.py -f data/subtitles data/model.bin`
8. `help` - вывести вышеописанную информацию. Например: `python3 preprocessing.py help`

Значения путей по умолчанию:
 - имя папки с исходными субтитрами: `data/subtitles`
 - имя папки с найденными русскими субтитрами: `data/subtitles_ru`
 - имя файла с собранными и очищенными русскими субтитрами: `data/subtitles_ru.txt`
 - имя файла с обработанными русскими субтитрами: `data/subtitles_ru_prepared.pkl`
 - имя файла с языковой моделью для JamSpell: `data/jamspell_ru_model_subtitles.bin`

---

### 1. Поиск русских субтитров.

При переводе сериалов на различные языки в названиях файлов с субтитрами принято указывать язык субтитров. Субтитры из корпуса [Taiga](https://tatianashavrina.github.io/taiga_site/downloads) полностью удовлетворяют этому требованию. По этому поиск русских субтитров заключается в поиске всех файлов, имеющих в названии конструкцию `.ru.`. Так же этот поиск можно легко дополнить другими языками (например, конструкция `.en.` в названии файла соответствует субтитрам на английском языке).

Реализация находится в методе `__search_ru_subtitles()` класса `Preparation` в файле `preprocessing.py`.

### 2. Очистка и исправление найденных русских субтитров.

Это самый длительный этап обработки, в идеале требующий распараллеливания на все доступные ядра процессора. 

Очистка субтитров заключается в следующем:
- удаление временной метки в начале каждой строки
- удаление временной метки внутри строки с разбиением строки на отдельные предложения по этой временной метке
- смена регистра всех строк на нижний
- замена ё на е
- удаление ссылок
- удаление квадратных и круглых скобок вместе с содержимым
- удаление html-тегов
- удаление всего, что не является русской/английской буквой, цифрой, '!', '?', ',', '.', ':', '*' или '-'
- удаление номера сезона и эпизода
- удаление ников переводчиков
- удаление пустых строк
- объединение строк, если текущая строка начинается на '…' или '-…', или предыдущая заканчивается на ',' (почти на 100% является признаком незавершённой фразы одного актёра)
- а так же удаление смайлов и многое другое

Процесс исправления субтитров не обязательный, но будет очень полезен при обучении, например, чат-бота. Этот процесс заключается в исправлении строк таким образом, что бы на одной строке располагалось только одно предложение, а точнее фраза одного актёра. Например, до исправления:
```
- привет, пап! - привет, доченька.
```
После исправления:
```
привет, пап!
привет, доченька.
```
Реализация находится в методе `__clear_ru_subtitles()` класса `Preparation` в файле `preprocessing.py`.

### 3. Разбиение каждого предложения на отдельные слова.

Тут всё просто. Каждая строка разбивается на отдельные слова (знаки препинания и '-' считаются отдельными словами), а затем приводится к одной длине (равной максимальной длине строки) с помощью слов-наполнителей `<PAD>`, т.е. в итоге список предложений становится двумерным списком слов.

### 4. Обучение языковой модели для корректировщика опечаток [JamSpell](https://github.com/bakwc/JamSpell#train).

На этом этапе скрипту требуется около 15 Гб оперативной памяти. Вначале происходит скачивание и сборка инструментов для обучения и валидации языковой модели для JamSpell с помощью скрипта [`download_jamspell.sh`](https://github.com/Desklop/Russian_subtitles_dataset/blob/master/download_jamspell.sh), затем запуск обучения языковой модели и в конце её оценка.

Результаты оценки модели (часть информации вырезана для уменьшения размера изображения):
![result_evaluation](https://github.com/Desklop/Russian_subtitles_dataset/blob/master/result_evaluation_jamspell_model.png)

Реализация находится в методе `__train_jamspell_model()` класса `Preparation` в файле `preprocessing.py`.

### 5. Построение гистограммы размеров предложений.

С помощью модуля matplotlib строится гистограмма размеров предложений для визуальной оценки набора данных и какой-либо его корректировке, по желанию.
![size_histogram](https://github.com/Desklop/Russian_subtitles_dataset/blob/master/data/size_histogram.png)

Реализация находится в методе `__build_histogram()` класса `Preparation` в файле `preprocessing.py`.

### 6. Предварительная обработка предложения 'на лету'.

Позволяет уже после обучения какой-либо NLP-модели или нейронной сети выполнять предварительную обработку приходящего извне предложения. Обработка заключается в удалении всего, что не является русскими буквами, цифрами, '!', '?', ',', '.', ':', '*' или '-', исправлении опечаток с помощью JamSpell (необходима созданная ранее языковая [модель](https://github.com/Desklop/Russian_subtitles_dataset/blob/master/data/jamspell_ru_model_subtitles.bin.zip)), разбиении предложения на отдельные слова и приведении полученной последовательности к необходимой длине с помощью слов-наполнителей `<PAD>`. Например, до обработки:
```
Привет прекрасный мир!
```
После обработки:
```
['<PAD>', '<PAD>', ..., '<PAD>', '<PAD>', 'привет', 'прекрасный', 'мир', '!']
```

Реализация находится в методе `prepare_sentence()` класса `Preparation` в файле `preprocessing.py`.

---

Если у вас возникнут вопросы или вы хотите сотрудничать, можете написать мне на почту: vladsklim@gmail.com или в [LinkedIn](https://www.linkedin.com/in/vladklim/).
