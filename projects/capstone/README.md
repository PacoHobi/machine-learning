# Dataset

The dataset is to big to include, so it has to be to downloaded manually from:

> https://www.kaggle.com/c/pubg-finish-placement-prediction/data

We only need the dataset `train_V2.csv`, and it should be located in `data/train_V2.csv`.

The final structure should be:

```
.
├── README.md
├── code
│   ├── config.py
│   ├── data_exploration.ipynb
│   └── utils
│       ├── model.py
│       ├── preprocessing.py
│       └── visualization.py
└── data
    └── train_V2.csv
```

# Requirements

This project requires **python 3.6** or higher. The required packages are:

```
scikit-learn==0.20.2
numpy==1.16.0
scipy==1.2.0
pandas==0.23.4
SALib==1.2
seaborn==0.9.0
jupyter==1.0.0
```

And they can be installed doing:

```
$ pip install -r code/requirements.txt
```

# Running notebook

To run the notebook correctly we first have to navigate to the `code` folder:

```
$ cd code/
$ jupyter notebook capstone.ipynb
```
