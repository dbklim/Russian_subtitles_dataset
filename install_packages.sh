#!/bin/bash

apt-get -y update
apt-get -y dist-upgrade

# Установка пакетов Ubuntu
apt-get -y install swig cmake python3 python3-pip git

# Установка пакетов Python3
yes | pip3 install --upgrade pip
yes | pip3 install matplotlib numpy jamspell

# Очистка кеша
apt-get -y autoremove
apt-get -y autoclean
apt-get -y clean

