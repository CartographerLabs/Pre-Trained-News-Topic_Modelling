from topic_modelling.topic_modelling import topic_modelling

modeller = topic_modelling()

print(modeller.identify_topic("hello world"))
print(modeller.get_topics())
