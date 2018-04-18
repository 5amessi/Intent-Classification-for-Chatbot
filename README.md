## Intent Classification for Chat-bot 

Intent Classification is a text classification task ot understand user intent and take an action.
Dataset is kvert dataset.

#Dateset
kvert data contains 3 labels {wethear, location, reminders}

#preproccessing
Remove Puncituation and stop words
Lemmatization 
Stemming 

#Embedding 
Embedding with Glove500dim , you can use word2vec too

#classifier  
We used Many to one Recurrent neural network with GRU Cell (one layer) then fed last hidden layer to fully conected neural network with one hidden layer and softmax in output layer
