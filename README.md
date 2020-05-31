# Pre-trained LDA Topic Modelling For News Website Topics

A pre-trained model and Python library for LDA topic modelling. This can be re-trained, however, has been trained by default for identifying topics for online data (Specifically news articles).

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

### Retraining The Model
The number of topics, passes, and dataset (seperated by ```\n``` with one line per article) can be defined. Topics defaults to 4, passes to 15, and the dataset defaults to a list of 22545 news articles located in the ```data``` directory of the library called ```dataset.txt```. Once retrained the new model files will be created in the libraries ```models``` folder and will be used for later use.

```python
modeller = topic_modelling()
modeller.re_train(number_of_topics=4, number_of_passes=70, dataset="new_data.txt")
```

### Visualising Groups
When using a Jupyter Notebook you can use the ```get_lda_display``` function to return the LDA data to visualise the model.

```python
modeller = topic_modelling()
pyLDAvis.display(modeller.get_lda_display())
```
## Groups
The model is pre-trained for 4 'news-related' groups. These group descriptions are detailed below:

| Group ID | Group Description | Example Keywords                                   |
|----------|-------------------|----------------------------------------------------|
| 0        | People            | Player, People, Family, League                     |
| 1        | Government        | President, Government, Company, National, Minister |
| 2        | Technology        | Google, Apple, Phone, Technology, Microsoft        |
| 3        | Residential       | Bedroom, Garden, House, Water                      |

## Dataset
This dataset has been gathered from the global most used news websites (written in English), where they're most recent pages have been identified as news (using:  [Website-Category-Identification-Tool](https://github.com/user1342/Website-Category-Identification-Tool)) then these articles have been added to the dataset.

- Kaggle Dataset and Kernel: www.kaggle.com/dataset/061b55bb510ebb7c484ef2c9ed5f5ddc474239d5952d9a75e7cc587923bec7df
