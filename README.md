# Pre-trained LDA Topic Modelling For Online Data 🌐

A pre-trained model and Python library for LDA topic modelling. This can be re-trained, however, has been trained by default for online data (Specifically news articles). 

## Instalation 
Install the library with  ```pip``` and the Github link.
```
python -m pip install git+https://github.com/user1342/Topic-Modelling-For-Online-Data.git
```

## Using

```python 
from topic_modelling.topic_modelling import topic_modelling

modeller = topic_modelling()

print(modeller.identify_topic("hello world"))
print(modeller.get_topics())
```

## Dataset
This dataset has been gathered from he global most used news websites (written in English), where they're most recent pages have been identified as news (using:  [Website-Category-Identification-Tool](https://github.com/user1342/Website-Category-Identification-Tool)) then these articles have been added to the dataset.
- Kaggle Dataset and Kernel: https://www.kaggle.com/jamessteve/news-websites 
