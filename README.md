## Intent Classification for Chat-bot 

Intent Classification is a text classification task to understand user intent and take an action.
Dataset is kvert dataset.

# Dataset
kvert data contains 3 labels {wethear, location, reminders}

# Preprocessing
Remove Punctuation and stop words
Lemmatization 
Stemming 

# Embedding 
Embedding with Glove500dim, you can use word2vec too

# Classification  
We used Many to one Recurrent neural network with GRU Cell (one layer) then fed last hidden layer to fully connected neural network with one hidden layer and softmax in the output layer
