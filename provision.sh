#!/bin/sh

export LC_ALL="en_US.UTF-8"

sudo apt update
sudo apt upgrade -y
sudo apt install -y unzip python3 python3-pip

sudo pip3 install --upgrade pip
sudo pip3 install jupyter pandas joblib sklearn matplotlib seaborn plotly xgboost==0.4a30
