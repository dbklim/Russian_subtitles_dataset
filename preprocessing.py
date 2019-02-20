#!/usr/bin/python3
# -*- coding: utf-8 -*-
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#       OS : GNU/Linux Ubuntu 18.04 
# COMPILER : Python 3.6.7
#   AUTHOR : Klim V. O.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

'''
Предназначен для предварительной обработки наборов данных для обучения нейронной сети.
'''

import sys
import os
import platform
import signal
import re
import subprocess
import curses
import pickle
import matplotlib.pyplot as plt
from shutil import copyfile
import numpy as np
import jamspell


curses.setupterm()


class Preparation:
    ''' Предназначен для предварительной обработки наборов данных для построения модели word2vec и обучения нейронной сети.
    Поддерживается набор данных на основе русских субтитров к 347 сериалам.
    1. f_name_jamspell_model - имя .bin файла с языковой моделью для JamSpell 
    (Необходимо в случае, когда объект класса используется для обработки одиночных последовательностей с помощью prepare_sentence()) '''
    def __init__(self, f_name_jamspell_model=None):
        self.max_sequence_length = None
        self.corrector = jamspell.TSpellCorrector()

        if f_name_jamspell_model != None:
            if os.path.isfile(f_name_jamspell_model) == False:
                print('[E] Языковая модель %s для JamSpell не найдена.' % f_name_jamspell_model)
                return
            self.corrector.LoadLangModel(f_name_jamspell_model)
        

    def prepare_subtitles(self, folder_all_subtitles, folder_ru_subtitles=None, f_name_ru_subtitles=None, f_name_prepared_subtitles=None, f_name_jamspell_model=None, force=False):
        ''' Предварительная обработка набора русских субтитров к 347 сериалам (для обучения модели word2vec и jamspell).
        
        Обработка включает: поиск русских субтитров в folder_all_subtitles и сохранение их в отдельную папку, удаление всего, что не является русской/английской 
        буквой, цифрой, '!', '?', ',', '.', ':', '*' или '-' (очистка), разбиение предложений из них на отдельные слова и сохранение результата в f_name_prepared_subtitles.
        1. folder_all_subtitles - имя папки со всеми субтитрами
        2. folder_ru_subtitles - имя папки с только русскими субтитрами (по умолчанию data/subtitles_ru)
        3. f_name_ru_subtitles - имя .txt файла с очищенными русскими субтитрами (по умолчанию data/subtitles_ru.txt)
        4. f_name_prepared_subtitles - имя .pkl файла с разбитыми на отдельные слова субтитрами (по умолчанию data/prepared_subtitles.pkl)
        5. force - True: если необходимо выполнить полную обработку всех данных (требуется folder_all_subtitles), False: если часть данных ранее была обработана
        6. f_name_jamspell_model - имя .bin файла с языковой моделью для JamSpell (по умолчанию data/jamspell_ru_model_subtitles.bin) '''

        if folder_ru_subtitles == None:
            if os.path.exists('data') == False:
                os.makedirs('data')
            folder_ru_subtitles = 'data/subtitles_ru'
        if f_name_ru_subtitles == None:
            if os.path.exists('data') == False:
                os.makedirs('data')
            f_name_ru_subtitles = 'data/subtitles_ru.txt'
        if f_name_prepared_subtitles == None:
            if os.path.exists('data') == False:
                os.makedirs('data')
            f_name_prepared_subtitles = 'data/prepared_subtitles.pkl'
        if f_name_jamspell_model == None:
            if os.path.exists('data') == False:
                os.makedirs('data')
            f_name_jamspell_model = 'data/jamspell_ru_model_subtitles.bin'

        if force:
            self.__search_ru_subtitles(folder_all_subtitles, folder_ru_subtitles)
            self.__clear_ru_subtitles(folder_ru_subtitles, f_name_ru_subtitles)
            self.__split_subtitles(f_name_ru_subtitles, f_name_prepared_subtitles)
            self.__train_jamspell_model(f_name_ru_subtitles, f_name_jamspell_model)
        else:
            if os.path.isfile(f_name_prepared_subtitles) == False:
                if os.path.isfile(f_name_ru_subtitles) == False:
                    if os.path.exists(folder_ru_subtitles) == False:
                        self.__search_ru_subtitles(folder_all_subtitles, folder_ru_subtitles)
                    self.__clear_ru_subtitles(folder_ru_subtitles, f_name_ru_subtitles)
                self.__split_subtitles(f_name_ru_subtitles, f_name_prepared_subtitles)
            if os.path.isfile(f_name_jamspell_model) == False:
                self.__train_jamspell_model(f_name_ru_subtitles, f_name_jamspell_model)
        print('[i] Готово')


    def __search_ru_subtitles(self, folder_all_subtitles, folder_ru_subtitles):
        ''' Поиск русских субтитров в folder_all_subtitles и сохранение их в folder_ru_subtitles.
        1. folder_all_subtitles - имя папки, которая содержит папки с сериалами, в кажой из которых находятся субтитры к этим сериалам
        2. folder_ru_subtitles - имя папки, которая содержит папки с сериалами, в каждой из которых находятся только русские субтитры к этим сериалам '''

        print('[i] Поиск русских субтитров в %s' % folder_all_subtitles)

        # Получение имён всех папок с сериалами, находящимися в folder_all_subtitles
        f_all_series = os.listdir(path=folder_all_subtitles)

        folder_all_subtitles += '/'

        # Получение имён всех .txt файлов с субтитрами из сериалов (все языки, кроме русского и английского игнорируются)
        f_all_ru_subtitles = []
        f_all_en_subtitles = []
        for f_one_series in f_all_series:
            f_all_ru_subtitles_one_series = []
            f_all_en_subtitles_one_series = []
            f_all_subtitles_one_series = os.listdir(path=folder_all_subtitles+f_one_series)
            i = 0
            while i < len(f_all_subtitles_one_series):
                if f_all_subtitles_one_series[i].find('.en.') != -1:
                    f_all_en_subtitles_one_series.append(f_all_subtitles_one_series[i])
                    del f_all_subtitles_one_series[i]
                    continue
                elif f_all_subtitles_one_series[i].find('.ru.') != -1:
                    f_all_ru_subtitles_one_series.append(f_all_subtitles_one_series[i])
                    del f_all_subtitles_one_series[i]
                    continue
                i += 1
            f_all_ru_subtitles.append([f_one_series, f_all_ru_subtitles_one_series])
            f_all_en_subtitles.append([f_one_series, f_all_en_subtitles_one_series])

        # Копирование .txt файлов с русскими субтитрами в folder_ru_subtitles с сохранением структуры папок и названий сериалов
        if os.path.exists(folder_ru_subtitles) == False:
            os.makedirs(folder_ru_subtitles)

        folder_ru_subtitles += '/'

        print('[i] Копирование найденных русских субтитров в %s' % folder_ru_subtitles)
        for f_ru_subtitles_one_series in f_all_ru_subtitles:
            if os.path.exists(folder_ru_subtitles + f_ru_subtitles_one_series[0]) == False:
                os.makedirs(folder_ru_subtitles + f_ru_subtitles_one_series[0])
            for f_ru_subtitles_one_series_part in f_ru_subtitles_one_series[1]:
                copyfile(folder_all_subtitles + f_ru_subtitles_one_series[0] + '/' + f_ru_subtitles_one_series_part, folder_ru_subtitles + f_ru_subtitles_one_series[0] + '/' + f_ru_subtitles_one_series_part)
        '''
        folder_en_subtitles = 'subtitles_en/'
        
        # Копирование .txt файлов с английскими субтитрами в folder_ru_subtitles с сохранением структуры папок и названий сериалов
        if os.path.exists(folder_en_subtitles) == False:
            os.makedirs(folder_en_subtitles)

        for f_en_subtitles_one_series in f_all_en_subtitles:
            if os.path.exists(folder_en_subtitles + f_en_subtitles_one_series[0]) == False:
                os.makedirs(folder_en_subtitles + f_en_subtitles_one_series[0])
            for f_en_subtitles_one_series_part in f_en_subtitles_one_series[1]:
                copyfile(folder_all_subtitles + f_en_subtitles_one_series[0] + '/' + f_en_subtitles_one_series_part, folder_en_subtitles + f_en_subtitles_one_series[0] + '/' + f_en_subtitles_one_series_part)
        '''


    def __clear_ru_subtitles(self, folder_ru_subtitles, f_name_ru_subtitles):
        ''' Считывание субтитров из folder_ru_subtitles, их очистка и сохранение в f_name_ru_subtitles.
        1. folder_ru_subtitles - имя папки, которая содержит папки с сериалами, в каждой из которых находятся русские субтитры к этим сериалам
        2. f_name_ru_subtitles - имя .txt файла со всеми очищенными субтитрами '''

        print('[i] Считывание субтитров из %s' % folder_ru_subtitles)

        # Получение имён всех папок с сериалами, находящимися в folder_ru_subtitles
        f_all_series = sorted(os.listdir(path=folder_ru_subtitles))

        folder_ru_subtitles += '/'
        
        # Получение имён всех .txt файлов с субтитрами из сериалов
        f_all_subtitles = []
        for f_one_series in f_all_series:
            f_all_subtitles.append([f_one_series, sorted(os.listdir(path=folder_ru_subtitles+f_one_series))])

        # Считывание всех .txt файлов с субтитрами из сериалов
        all_subtitles = []
        for f_subtitles_one_series in f_all_subtitles:
            for f_subtitles_one_series_part in f_subtitles_one_series[1]:                
                with open(folder_ru_subtitles+f_subtitles_one_series[0]+'/'+f_subtitles_one_series_part, 'r') as f_subtitles:
                    all_subtitles += f_subtitles.readlines()

        print('[i] Очистка субтитров... ')
        i = 0
        while i < len(all_subtitles):
            if i % 1000 == 0 or i == len(all_subtitles) - 1:
                os.write(sys.stdout.fileno(), curses.tigetstr('cuu1'))
                print('[i] Очистка субтитров... %i из %i' % (i, len(all_subtitles)))
            sentence = all_subtitles[i]
            sentence = re.sub(r'(\ufeff)?\d+\t?\d{1,2}:\d{1,2}:\d{1,2},\d{1,5}\t?\d{1,3}:\d{1,2}:\d{1,2},\d{1,5}\t?', '', sentence) # Удаление временной метки
            
            # Удаление временной метки внутри строки с разбиением строки на отдельные предложения
            coincidence = re.findall(r'\d{1,3}\s*\t?\d{1,2}:\d{1,2}:\d{1,2},\d{1,5}\s*-?-?>?\s*\d{1,2}:\d{1,2}:\d{1,2},\d{1,5}', sentence) 
            if len(coincidence) != 0:
                index = sentence.find(coincidence[0])
                all_subtitles.insert(i + 1, sentence[index + len(coincidence[0]):])
                sentence = sentence[:index]
            
            sentence = sentence.lower().replace('ё', 'е')
            sentence += ' ' # Что бы смайлы в конце строки удалялись нормально (если они есть)
            sentence = ' ' + sentence # Что бы дефис в самом начале корректно обрабатывался (если он есть)
            
            # Удаление ссылок
            sentence = re.sub(r'(www\.[^\s]+)|(https?://[^\s]+)|(\.com)|(\.ru)|(\.org)', ' ', sentence)

            # Удаление скобок вместе с их содержимым
            sentence = re.sub(r'\([^)]*\)', '', sentence)
            sentence = re.sub(r'\[.*?\]', '', sentence)
            
            # Добавление пробелов перед тире и замена его на дефис
            sentence = re.sub(r'-+', '-', sentence)
            sentence = re.sub(r'[\d\s\.,]-', ' - ', sentence)
            sentence = re.sub(r'[\s\.,]—', ' - ', sentence)
            sentence = re.sub(r'—', '-', sentence)
            
            # Удаление html-тегов
            sentence = re.sub(r'<[^>]*>', '', sentence)

            # Удаление всего, что не является русской/английской буквой, цифрой, '!', '?', ',', '.', ':', '*' или '-'
            sentence = re.sub(r'[^a-zA-Zа-яА-Я0-9!\?,\.:\*-]+', ' ', sentence)

            # Удаление номера сезона и эпизода
            sentence = re.sub(r'сезон\s\d+\sэпизод\s\d+', '', sentence)
            
            # Замена нескольких подряд идущих ','  '!' и '?' на одиночные
            sentence = re.sub(r',+', ',', sentence)
            sentence = re.sub(r'!+', '!', sentence)
            sentence = re.sub(r'\?+', '?', sentence)

            # Удаление пробелов перед и/или после '…'
            sentence = re.sub(r'\s?(\.{2,10}\s?)+', '… ', sentence)
            sentence = re.sub(r'\s…', '…', sentence)

            # Замена конструкций '?...' и '!...' на '?' и '!' 
            sentence = re.sub(r'\?…', '?', sentence)
            sentence = re.sub(r'!…', '!', sentence)

            # Удаление пробелов перед и/или после ',', '.', '!' и '?'
            sentence = re.sub(r'\s?,\s?', ', ', sentence)
            sentence = re.sub(r'\s?\.\s?', '. ', sentence)
            sentence = re.sub(r'\s?!\s?', '! ', sentence)
            sentence = re.sub(r'\s?\?\s?', '? ', sentence)

            # Исправление конструкций '!,', '!.', '?,', '?.', ',.', '.,', ',!'. '.!', ',?', '.?'
            sentence = re.sub(r'!\s*,', '!', sentence)
            sentence = re.sub(r'!\s*\.', '!', sentence)
            sentence = re.sub(r'\?\s*,', '?', sentence)
            sentence = re.sub(r'\?\s*[\.…]', '?', sentence)
            sentence = re.sub(r',\s*[\.…]', ',', sentence)
            sentence = re.sub(r'\.\s*,', '.', sentence)
            sentence = re.sub(r'…\s*,', '…', sentence)
            sentence = re.sub(r',\s*!', ',', sentence)
            sentence = re.sub(r'\.\s*!', '.', sentence)
            sentence = re.sub(r',\s*\?', ',', sentence)
            sentence = re.sub(r'\.\s*\?', '.', sentence)

            # Удаление пробела после '.' или ',' в дробных числах
            sentence = re.sub(r'(\d+)[,.]\s(\d+)', r'\1,\2', sentence)

            # Удаление нескольких подряд идущих одинаковых букв (замена, например, 'мдаааааа' на 'мдаа')
            sentence = re.sub(r'([а-я])\1{2,100}', r'\1\1', sentence)
            
            # Удаление нескольких подряд идущих пробелов
            sentence = re.sub(r'\s+', ' ', sentence)

            # Удаление пробелов в начале и конце строки
            sentence = sentence.strip() 

            # Удаление ников переводчиков
            coincidence = re.match(r'(перевод)|(перевели)|(переведено)', sentence) 
            if coincidence != None:
                del all_subtitles[i]
                continue
                
            # Удаление пустых строк
            if sentence == '':
                del all_subtitles[i]
                continue

            # Если текущая строка начинается на '…' или '-…', или предыдущая заканчивается на ',', то объединить текущую строку с предыдущей
            if sentence[0] == '…' or sentence[:2] == '-…' or all_subtitles[i-1][-1] == ',':
                all_subtitles[i-1] += ' ' + sentence
                all_subtitles[i-1] = re.sub(r'-?…+', '', all_subtitles[i-1])
                all_subtitles[i-1] = re.sub(r'\s+', ' ', all_subtitles[i-1])
                all_subtitles[i-1] = all_subtitles[i-1].strip()
                del all_subtitles[i]
                continue

            all_subtitles[i] = sentence
            i += 1
        
        # Если строка содержит две реплики (больше не встречалось), то она разбивается на две отдельные строки
        # Было: - привет, пап! - привет, доченька.
        # Стало: привет, пап!
        #        привет, доченька.
        print('[i] Исправление диалогов... ')
        i = 0
        while i < len(all_subtitles):
            if i % 1000 == 0 or i == len(all_subtitles) - 1:
                os.write(sys.stdout.fileno(), curses.tigetstr('cuu1'))
                print('[i] Исправление диалогов... %i из %i' % (i, len(all_subtitles)))
            all_subtitles[i] = re.sub(r'\n', '', all_subtitles[i])
            index = all_subtitles[i].find(' - ')        
            if index != -1:                
                temp = all_subtitles[i][index + 1:]                
                all_subtitles.insert(i + 1, temp)                
                all_subtitles[i] = all_subtitles[i][:index]
                all_subtitles[i] = re.sub(r'- ', '', all_subtitles[i])
                all_subtitles[i + 1] = re.sub(r'- ', '', all_subtitles[i + 1])
                i += 2
            else:
                all_subtitles[i] = re.sub(r'- ', '', all_subtitles[i])
                i += 1

        print('[i] Сохранение очищенных субтитров в %s' % f_name_ru_subtitles)
        with open(f_name_ru_subtitles, 'w') as f_ru_subtitles:
            for subtitles in all_subtitles:
                f_ru_subtitles.write(subtitles + '\n')


    def __split_subtitles(self, f_name_ru_subtitles, f_name_prepared_subtitles):
        ''' Считывание очищенных субтитров из f_name_ru_subtitles, разбиение предложений из них на слова, построение гистограммы размеров предложений
        и сохранение в f_name_prepared_subtitles.
        1. f_name_ru_subtitles - имя .txt файла с очищенными субтитрами
        2. f_name_prepared_subtitles - имя .pkl файла с обработанными субтитрами '''

        print('[i] Считывание очищенных субтитров из %s' % f_name_ru_subtitles)
        with open(f_name_ru_subtitles, 'r') as f_ru_subtitles:
            subtitles = f_ru_subtitles.readlines()

        print('[i] Разбиение предложений из субтитров на отдельные слова... ')
        i = 0
        subtitles_words = []
        while i < len(subtitles):
            if i % 1000 == 0 or i == len(subtitles) - 1:
                os.write(sys.stdout.fileno(), curses.tigetstr('cuu1'))
                print('[i] Разбиение предложений из субтитров на отдельные слова... %i из %i' % (i, len(subtitles)))
            words = re.split(r'(\W)', subtitles[i]) # разбиение строки на последовательность из слов и знаков препинания
            words = [ word for word in words if word.strip() ] # удаление пустых элементов из последовательности
            if len(words) < 135:
                subtitles_words.append(words)
            i += 1

        self.max_sequence_length = (np.asarray([ len(words) for words in subtitles_words ])).max()
        print('[i] Максимальная длина предложения: ' + str(self.max_sequence_length))
        
        print('[i] Сохранение предобработанных субтитров в %s' % f_name_prepared_subtitles)
        with open(f_name_prepared_subtitles, 'wb') as file:
            pickle.dump(subtitles_words, file)

        self.__build_histogram(subtitles_words)

    
    def __train_jamspell_model(self, f_name_ru_training_sample, f_name_jamspell_model):
        ''' Загружает инструменты для обучения JamSpell с https://github.com/bakwc/JamSpell и выполняет обучение языковой модели. Для загрузки
        инструментов используется download_jamspell.sh.
        1. f_name_ru_training_sample - имя .txt файла с обучающей выборкой на русском языке 
        2. f_name_jamspell_model - имя .bin файла с обученной языковой моделью '''

        if os.path.exists('JamSpell') == False:
            subprocess.call('./download_jamspell.sh', shell=True)
        
        command_line = 'JamSpell/build/main/jamspell train JamSpell/test_data/alphabet_ru.txt ' + f_name_ru_training_sample + ' ' + f_name_jamspell_model
        subprocess.call(command_line, shell=True)

        command_line = 'python3 JamSpell/evaluate/evaluate.py -a JamSpell/test_data/alphabet_ru.txt -jsp ' + f_name_jamspell_model + ' -mx 5000000 ' + f_name_ru_training_sample
        subprocess.call(command_line, shell=True)


    def __build_histogram(self, dataset, f_name_hist=None):
        ''' Построение гистограммы размеров предложений.
        1. dataset - list с очищенными предложениями
        2. f_name_hist - имя .png файла с построенной гистограммой '''

        if f_name_hist == None:
            f_name_hist = 'data/size_histogram.png'

        # Колчиество слов в каждом предложении
        number_words = np.asarray([ np.asarray(len(sentence)) for sentence in dataset ]) 

        print('[i] Размеры предложений:')
        print('\tмаксимальный: ', number_words.max()) # Максимальная длинна
        print('\tминимальный: ', number_words.min()) # Минимальная длинна
        print('\tмeдиана: ', np.median(number_words).astype(int)) # Медианная длинна

        # Гистограмма размеров предложений
        print('[i] Построение гистограммы размеров предложений...')
        plt.figure()
        plt.hist(x = number_words, label = ['предложение'])
        plt.title('Гистограмма размеров предложений') 
        plt.ylabel('количество')
        plt.xlabel('длинна')
        plt.legend()
        print('[i] Сохранение гистограммы размеров предложений в %s' % f_name_hist)
        plt.savefig(f_name_hist, dpi=100)

        
    def prepare_sentence(self, sentence):
        ''' Предварительная обработка предложения: удаление всего, что не является русскими буквами или '-', исправление опечаток, 
        разбиение предложения на отдельные слова и приведение полученной последовательности к необходимой длине (max_sequence_length).
        1. sentence - строка с предложением
        2. возвращает list с обработанным предложением 
        
        !!!Если не был вызван prepare_subtitles(), необходимо задать max_sequence_length!!! '''

        if self.max_sequence_length == None:
            print('[E] Перед использованием необходимо задать максимальную длину итоговой последовательности!')
            return ['error']
        sentence = sentence.lower()

        # Удаление всего, что не является русской буквой, цифрой, '!', '?', ',', '.', ':', '*' или '-'
        sentence = re.sub(r'[^а-яА-Я0-9!\?,\.:\*-]+', ' ', sentence)

        # Исправление опечаток и простых грамматических ошибок
        sentence = self.corrector.FixFragment(sentence)
        
        # Разбиение сообщения на отдельные слова
        sentence = re.split(r'(\W)', sentence) # разбиение строки на последовательность из слов и знаков препинания
        sentence = [ word for word in sentence if word.strip() ] # удаление пустых элементов из последовательности
        #sentence = [ word for word in sentence if len(word) > 1 or word == '-' ] # удаление слов длиной 1 символ

        # Обрезка сообщения
        if len(sentence) > self.max_sequence_length:
            sentence = sentence[:self.max_sequence_length]

        # Приведение сообщения к необходимой длине
        sentence = ['<PAD>'] * (self.max_sequence_length - len(sentence)) + sentence
        return sentence




def main():
    folder_all_subtitles = 'data/subtitles'
    folder_ru_subtitles = 'data/subtitles_ru'
    f_name_ru_subtitles = 'data/subtitles_ru.txt'
    f_name_prepared_subtitles = 'data/subtitles_ru_prepared.pkl'

    f_name_jamspell_model = 'data/jamspell_ru_model_subtitles.bin'

    # Максимальное потребление оперативной памяти (в основном во время построения языковой модели): 15 Гб
    # Аргументы командной строки имеют следующую структуру: [ключ] имя_папки/файла
    # 1. foldername - задать имя папки с исходными субтитрами (должна содержать вложенные папки с .txt файлами субтитров)
    # 2. modelname - задать имя .bin файла для сохранения языковой модели для JamSpell
    # 3. foldername modelname - задать имя папки с исходными субтитрами и имя .bin файла для сохранения языковой модели для JamSpell
    # 4. -f foldername - обработка всех исходных субтитров из foldername без учёта промежуточных результатов
    # 5. -f modelname - обработка всех исходных субтитров без учёта промежуточных результатов и сохранение языковой модели для 
    # JamSpell в modelname
    # 6. -f foldername modelname - обработка всех исходных субтитров из foldername без учёта промежуточных результатов и сохранение 
    # языковой модели для JamSpell в modelname

    test_prepare_sentence = False
    force = False
    if len(sys.argv) > 1:
        if sys.argv[1].find('.bin') != -1:  # задать имя .bin файла для сохранения языковой модели для JamSpell
            f_name_jamspell_model = sys.argv[1]
        elif os.path.exists(sys.argv[1]):  # задать имя папки с исходными субтитрами
            if len(sys.argv) > 2:
                if sys.argv[2].find('.bin') != -1:  # задать имя .bin файла для сохранения языковой модели для JamSpell
                    f_name_jamspell_model = sys.argv[2]
                else:
                    print("\n[E] Неверный аргумент командной строки '" + sys.argv[2] + "'. Введите help для помощи.\n")
                    return
            folder_all_subtitles = sys.argv[1]
            folder_ru_subtitles = folder_all_subtitles + '_ru'
            f_name_ru_subtitles = folder_ru_subtitles + '.txt'
            f_name_prepared_subtitles = folder_ru_subtitles + '_prepared.pkl'
        elif sys.argv[1] == '-f':  # обработка всех исходных субтитров без учёта промежуточных результатов
            if len(sys.argv) > 2:
                if os.path.exists(sys.argv[2]):  # задать имя папки с исходными субтитрами
                    if len(sys.argv) > 3:
                        if sys.argv[3].find('.bin') != -1:  # задать имя .bin файла для сохранения языковой модели для JamSpell
                            f_name_jamspell_model = sys.argv[3]
                        else:
                            print("\n[E] Неверный аргумент командной строки '" + sys.argv[3] + "'. Введите help для помощи.\n")
                            return
                    folder_all_subtitles = sys.argv[2]
                    folder_ru_subtitles = folder_all_subtitles + '_ru'
                    f_name_ru_subtitles = folder_ru_subtitles + '.txt'
                    f_name_prepared_subtitles = folder_ru_subtitles + '_prepared.pkl'
                elif sys.argv[2].find('.bin') != -1:  # задать имя .bin файла для сохранения языковой модели для JamSpell
                    f_name_jamspell_model = sys.argv[2]
                else:
                    print("\n[E] Неверный аргумент командной строки '" + sys.argv[2] + "'. Введите help для помощи.\n")
                    return
                force = True
            else:
                print("\n[E] Неверный аргумент командной строки '" + sys.argv[1] + "'. Введите help для помощи.\n")
                return
        elif sys.argv[1] == 'help':
            print('\nПоддерживаемые аргументы командной строки:')
            print('\tбез аргументов - обработка исходных субтитров с учётом промежуточных результатов от прошлых запусков (используются пути по умолчанию)')
            print('\tfoldername - задать имя папки с исходными субтитрами (должна содержать вложенные папки с .txt файлами субтитров)')
            print('\tmodelname - задать имя .bin файла для сохранения языковой модели для JamSpell')
            print('\tfoldername modelname - задать имя папки с исходными субтитрами и имя .bin файла для сохранения языковой модели для JamSpell')
            print('\t-f foldername - обработка всех исходных субтитров из foldername без учёта промежуточных результатов')
            print('\t-f modelname - обработка всех исходных субтитров без учёта промежуточных результатов и сохранение языковой модели для JamSpell в modelname')
            print('\t-f foldername modelname - обработка всех исходных субтитров из foldername без учёта промежуточных результатов и сохранение языковой модели для JamSpell в modelname')

            print('\nЗначения путей по умолчанию:')
            print('\tимя папки с исходными субтитрами - %s' % folder_all_subtitles)
            print('\tимя папки с найденными русскими субтитрами - %s' % folder_ru_subtitles)
            print('\tимя файла с собранными и очищенными русскими субтитрами - %s' % f_name_ru_subtitles)
            print('\tимя файла с обработанными русскими субтитрами - %s' % f_name_prepared_subtitles)
            print('\tимя файла с языковой моделью для JamSpell - %s\n' % f_name_jamspell_model)
            return
        else:
            print("\n[E] Неверный аргумент командной строки '" + sys.argv[1] + "'. Введите help для помощи.\n")
            return
    else:
        test_prepare_sentence = True
    
    pr = Preparation()    
    pr.prepare_subtitles(folder_all_subtitles, folder_ru_subtitles, f_name_ru_subtitles, f_name_prepared_subtitles, f_name_jamspell_model, force)
    
    if test_prepare_sentence:
        pr.max_sequence_length = 134
        while True:
            sentence = input('Введите предложение: ')
            result = pr.prepare_sentence(sentence)
            print(result)


def on_stop(*args):
    print('\n[i] Остановлено')
    os._exit(0)


if __name__ == '__main__':
    # При нажатии комбинаций Ctrl+Z, Ctrl+C либо закрытии терминала будет вызываться функция on_stop() (Работает только на linux системах!)
    if platform.system() == 'Linux':
        for sig in (signal.SIGTSTP, signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, on_stop)
    main()