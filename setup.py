import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Online-Data-Topic-Modelling",
    version="0.0.1",
    author="James Stevenson",
    author_email="hi@jamesstevenson.me",
    description="A pre-trained model for topic modelling online data (Primarily online news articles).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/user1342/Topic-Modelling-For-Online-Data",
    packages=["topic_modelling"],
    include_package_data=True,
    package_data={"":["topic_modelling/data","topic_modelling/models"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GPL-3.0",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['spacy', 'nltk', 'gensim','pyLDAvis'],
)
