# Where are they looking?

This repository contains an implementation of the "Where are they looking?" paper by A. Recasens*, A. Khosla*, C. Vondrick and A. Torralba.

## Introduction

A deep neural network-based approach for gaze-following automated using a SSD face detector.

## Installation

- [Pytorch 1.1 & Torchvision](https://pytorch.org/)

## Usage

* First, download pretrained Places365 AlexNet model: https://urlzs.com/ytKK3

* Then run: python3 main.py --data_dir=`location to gazefollow dataset` --placesmodelpath=`location to places365 alexnet model`


* Please check out opts.py for other parameter changing.

## Contact

Please do get in touch with us by email for any questions, comments, suggestions you have!

* rohit.gajawada@gmail.com
* haard.panchal@students.iiit.ac.in

## References

* sfzhang15's SFD detector is used for face detection (https://github.com/sfzhang15/SFD).
* Link to the NIPS 2015 paper from MIT: http://people.csail.mit.edu/khosla/papers/nips2015_recasens.pdf. Please cite them if you decide to use this project for your research.


## Cody's Part of the readme 
I used this with Pytorch 1.5

add the following folders 

model_outputs 
savedmodels

extract the data.zip file  to a folder named data

run using this 
python modeltester.py --data_dir ./data/ --placesmodelpath ./whole_alexnet_places365.pth

Notes on updates:
Implementations updates to suport python 3.X from 2.x
Updated network to train by fixing layer sizing
Updated the conversion of torch.transforms to images in modeltester.py
Didn't get the modeltester_withssd.py updated and working yet.

TODOs: 
Add a requirments.txt for python packages used
clean up unused imports in python files. 