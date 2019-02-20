#!/bin/bash

git clone https://github.com/bakwc/JamSpell.git
cd JamSpell
mkdir build
cd build
cmake ..
make
cd -
yes | rm -r .git
