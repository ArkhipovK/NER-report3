# NER-report3 : Прототип для модуля распознавания сущностей

Данный прототип способен преобрабатывать текстовые корпуса получаемые из дампов статей википедии, автоматически маркировать слова основываясь на экзеплярах полученных из заголовком статей, обучаться на полученном корпусе для дальнейшего распознавания неизвестных статей.
В качестве эмбеннигов используется модель обученная с помощью fastText
Для обучения используется нейросети библиотека Keras

Работоспособность протестирована только на Linux (В Windows работоспособность должна сохраняться за исключеним использования fastText )
---
## Установка

Для установки необходимо перейти в корневую папку проекта на выполнить команду 

```shell
python setup.py install
```

---
## использование

В данный момент у модуля 3 функции: 

1. Приведение данных из статей Википедии к пригодному для обучения вида (Избавление от лишних символов, разделов, приведение к нижнему регистру)
Для использования данной функции необходимо вызвать файл с параметрам файла (папкой) с дампом. По умолчанию полученные результаты заносятся в папку text.

```shell
python ner_report3/preprocess.py 
```
```shell
positional arguments:
  path                  Path of a file or a folder of files.

optional arguments:
  -h, --help            show this help message and exit
  -output OUTPUT        Path of a file or a folder of files.
  -e EXTENSION, --extension EXTENSION
                        File extension to filter by.
  -wikiText             Process texts from wikidump
  -toLower              Lowercase all words
  -wipeChars WIPECHARS  Regexp for pattern to be wiped
```

2. Обучение эмбеддингам, для этого используется уже скомпилированная для Linux версия fastText. Включена в проект для удобства ( на Windows  не работает)


3. Обучение модели нейронной сети. На данный момент используется архитектура двунаправленной LSTM с выводом в CRF (один из зарекомендовавших себя методов), в архитектуре используется 1 слой с dropout (0.1) и один плотный слой, на полученный результат обрабатывается CRF. Функция активации на срытых слоях - "ReLU" По умолчанию полученные результаты заносятся в файл models/keras/animals.h5 .
Для запуска необходимо передать параметр с именем файла для сохранения модели 
```shell
python ner_report3/learnSequences.py models/keras/animals.h5
```
```shell
usage: learnSequences.py [-h] [-epoch EPOCH] [-batchSize BATCHSIZE]
                         [-arch ARCH] [-wordsModel WORDSMODEL]
                         [-textData TEXTDATA] [-labelData LABELDATA]
                         [-dim DIM] [-tokenizer TOKENIZER]
                         output

Text preprocessing tool

positional arguments:
  output                Path to save model

optional arguments:
  -h, --help            show this help message and exit
  -epoch EPOCH          Epochs number
  -batchSize BATCHSIZE  Size of batches for learning
  -arch ARCH            Name of architecture to use
  -wordsModel WORDSMODEL
                        Path for words representation model.
  -textData TEXTDATA    Path for word corpus.
  -labelData LABELDATA  Path for folder with files class instances lists
  -dim DIM              Size of word vectors.
  -tokenizer TOKENIZER  Tokenizer type (sentence or abstract based)
```
