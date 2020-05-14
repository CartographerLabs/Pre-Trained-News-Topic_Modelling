# Pre-trained LDA Topic Modelling For Online Data

A pre-trained model and Python library for LDA topic modelling. This can be re-trained, however, has been trained by default for online data (Specifically news articles). 

## Installation 
Install the library with  ```pip``` and the GitHub link.
```
python -m pip install git+https://github.com/user1342/Topic-Modelling-For-Online-Data.git
```

## Usage

```python 
from topic_modelling.topic_modelling import topic_modelling

modeller = topic_modelling()

print(modeller.identify_topic("hello world"))
print(modeller.get_topics())
```

## Dataset
This dataset has been gathered from the global most used news websites (written in English), where they're most recent pages have been identified as news (using:  [Website-Category-Identification-Tool](https://github.com/user1342/Website-Category-Identification-Tool)) then these articles have been added to the dataset.

- Kaggle Dataset and Kernel: www.kaggle.com/dataset/061b55bb510ebb7c484ef2c9ed5f5ddc474239d5952d9a75e7cc587923bec7df
