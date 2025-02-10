# Precog_Task

## Overview

This repository contains the codebase for the Precog Task. The project is divided into two main parts: PART1 and PART2, each handling different aspects of the task.

## Directory Structure

```
Precog_Task/
│
├── PART1/
│   ├── eng_news_2024_300K-sentences
│   └── part1.py
|   |___ compare_pretrained.py
│
├── PART2/
│   └── part2.py
|
└── README.md
```

## Models

You have to download the pre-trained models from the following links:
- [GoogleNews-vectors-negative300.bin.gz](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g)
- [wiki.hi](https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.hi.zip)

Place them in root folder itself

## Commands to Run the Project

### PART1

To run the PART1 of the project, navigate to the `PART1` directory and execute the following command:

```bash
cd PART1
python part1.py
python compare_pretrained.py
```

### PART2

To run the PART2 of the project, navigate to the `PART2` directory and execute the following command:

```bash
cd PART2
python part2.py
```

### Libraries Used
Detailed libraries for each part in their respective README
- numpy
- pandas
- scikit-learn
- torch

## Approach

For this project, we followed a systematic approach to solve the task at hand. Each part of the project (PART1 and PART2) has its own specific approach, which is detailed in their respective README files.