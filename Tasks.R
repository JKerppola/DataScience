# For the basic graph descriptive measures we use the following libraries

library('igraph')
library('igraphdata')

# Create adjency matrix
adj_matrix = matrix(c(0,1,1,1,0,0,1,0,
                    1,0,1,1,0,0,0,0,
                    1,1,0,1,0,0,0,0,
                    1,1,1,0,1,0,1,1,
                    0,0,0,1,0,0,1,1,
                    0,1,1,0,0,0,0,1,
                    1,0,0,1,1,0,0,1,
                    0,0,0,1,1,1,1,0), 
                    nrow = 8,
                    ncol = 8,
                    byrow = TRUE)

# Create the graph
graph_ex = graph.adjacency(adj_matrix, mode = 'undirected')

# Visualize
tkplot(graph_ex, vertex.color = 'blue')

# Calculate the most important descriptive measures 

# degree
graph_degree = degree(graph_ex)
# Degree distribution
graph_dd = degree.distribution(graph_ex)
# Graph density
graph_density = graph.density(graph_ex)
# Graph shortest path
graph_paths = shortest.paths(graph_ex)

# Results of descriptive measures

cat("Graph degree: ", graph_degree)
cat("Graph degree distribution: ", graph_dd)
cat("Graph density: ", graph_density)
graph_paths





# Create random graphs 

set.seed(1990)

# First random network where we have a set probability for an edge (Type = gnp)
random_plot_erdos1 <- erdos.renyi.game(n=8,0.1,type='gnp')
random_plot_erdos2 <- erdos.renyi.game(n=8,0.3,type='gnp')
random_plot_erdos3 <- erdos.renyi.game(n=8,0.6,type='gnp')
random_plot_erdos4 <- erdos.renyi.game(n=8,0.9,type='gnp')

# Calculate descriptive measures of the random graph
#Degree
degree_erdos1 = degree(random_plot_erdos1)
degree_erdos2 = degree(random_plot_erdos2)
degree_erdos3 = degree(random_plot_erdos3)
degree_erdos4 = degree(random_plot_erdos4)


# Degree distribution
graph_dd_erdos1 = degree.distribution(random_plot_erdos1)
graph_dd_erdos2 = degree.distribution(random_plot_erdos2)
graph_dd_erdos3 = degree.distribution(random_plot_erdos3)
graph_dd_erdos4 = degree.distribution(random_plot_erdos4)


# Graph density
graph_density_erdos1 = graph.density(random_plot_erdos1)
graph_density_erdos2 = graph.density(random_plot_erdos2)
graph_density_erdos3 = graph.density(random_plot_erdos3)
graph_density_erdos4 = graph.density(random_plot_erdos4)



# Graph shortest path
graph_paths_erdos1 = shortest.paths(random_plot_erdos1)
graph_paths_erdos2 = shortest.paths(random_plot_erdos2)
graph_paths_erdos3 = shortest.paths(random_plot_erdos3)
graph_paths_erdos4 = shortest.paths(random_plot_erdos4)




cat("Graph degree 1: ", degree_erdos1)
cat("Graph degree 2: ", degree_erdos2)
cat("Graph degree 3: ", degree_erdos3)
cat("Graph degree 4: ", degree_erdos4)


cat("Graph degree distribution 1: ", graph_dd_erdos1)
cat("Graph degree distribution 2: ", graph_dd_erdos2)
cat("Graph degree distribution 3: ", graph_dd_erdos3)
cat("Graph degree distribution 4: ", graph_dd_erdos4)


cat("Graph density 1: ", graph_density_erdos1)
cat("Graph density 2: ", graph_density_erdos2)
cat("Graph density 3: ", graph_density_erdos3)
cat("Graph density 4: ", graph_density_erdos4)


graph_paths


# --> Most similar to the original graph is the third random erdos graph

# degree
graph_degree 
degree_erdos3

# Degree distribution
graph_dd 
graph_dd_erdos3

# Graph density
graph_density 
graph_density_erdos3

# Graph shortest path
graph_paths 
graph_paths_erdos3




# Create another random graph and calculate discriptive measurements

# Create a small-world network
random_plot_SW1 <- watts.strogatz.game(dim=1, size=8, nei=2, p=0.1)
random_plot_SW2 <- watts.strogatz.game(dim=1, size=8, nei=2, p=.3)
random_plot_SW3 <- watts.strogatz.game(dim=1, size=8, nei=2, p=.5)
random_plot_SW4 <- watts.strogatz.game(dim=1, size=8, nei=2, p=.8)


# Calculate descriptive measures

# degree
graph_degree_SW1 = degree(random_plot_SW1)
graph_degree_SW2 = degree(random_plot_SW2)
graph_degree_SW3 = degree(random_plot_SW3)
graph_degree_SW4 = degree(random_plot_SW4)


# Degree distribution
graph_dd_SW1 = degree.distribution(random_plot_SW1)
graph_dd_SW2 = degree.distribution(random_plot_SW2)
graph_dd_SW3 = degree.distribution(random_plot_SW3)
graph_dd_SW4 = degree.distribution(random_plot_SW4)


# Graph density
graph_density_SW1 = graph.density(random_plot_SW1)
graph_density_SW2 = graph.density(random_plot_SW2)
graph_density_SW3 = graph.density(random_plot_SW3)
graph_density_SW4 = graph.density(random_plot_SW4)

# Graph shortest path
graph_paths_SW1 = shortest.paths(random_plot_SW1)
graph_paths_SW2 = shortest.paths(random_plot_SW2)
graph_paths_SW3 = shortest.paths(random_plot_SW3)
graph_paths_SW4 = shortest.paths(random_plot_SW4)


# Results of the small world graph

cat("Graph degree 1: ", graph_degree_SW1)
cat("Graph degree 2: ", graph_degree_SW2)
cat("Graph degree 3: ", graph_degree_SW3)
cat("Graph degree 4: ", graph_degree_SW4)


cat("Graph degree distribution 1: ", graph_dd_SW1)
cat("Graph degree distribution 2: ", graph_dd_SW2)
cat("Graph degree distribution 3: ", graph_dd_SW3)
cat("Graph degree distribution 4: ", graph_dd_SW4)


cat("Graph density 1: ", graph_density_SW1)
cat("Graph density 2: ", graph_density_SW2)
cat("Graph density 3: ", graph_density_SW3)
cat("Graph density 4: ", graph_density_SW4)



# -> By looking at the descriptive measures the most similar to the original graph is the

# degree
graph_degree 
graph_degree_SW3

# Degree distribution
graph_dd 
graph_dd_SW3

# Graph density
graph_density 
graph_density_SW3

# Graph shortest path
graph_paths 
graph_paths_SW3





# Plot the different graphs
tkplot(graph_ex, vertex.color = 'blue')
tkplot(random_plot_SW3, vertex.color = 'green')
tkplot(random_plot_erdos3, vertex.color = 'red')


# ----------------------------------------- #
# ---------------- TASK 2 ----------------- #
# ----------------------------------------- #


# Clear workspace/environment
rm(list = ls(all=T))


# Function for calculating closeness of the graph

calc_closseness = function(adj_graph){
  # Create a graph using the adjency matrix
  graph = graph.adjacency(adj_graph, mode = 'undirected')
  
  # Form the shortest paths
  paths = shortest.paths(graph)
  
  # Calculate the normalized closeness for each node 
  cl = (rowSums(paths)/(length(paths[1,])-1))^-1
  
  return(cl)
  
}

# Function for pageRank

pg_alg <- function(m, n){
  ## m is the matrix, n is the number of steps in the iteration
  temp <- m
  for (i in 1:n){
    # %*% is the matrix product
    temp <- temp %*% m
  }
  temp %*% prob_d
}




 # Test the function
graph_adj = matrix(c(0,1,1,1,1,
                     1,0,0,0,0,
                     1,0,0,0,0,
                     1,0,0,0,0,
                     1,0,0,0,0),
                   nrow = 5,
                   ncol = 5,
                   byrow = TRUE)

graph10 = graph.adjacency(graph_adj, mode = 'undirected')
tkplot(graph10, vertex.color = 'green' )
test = calc_closseness(graph_adj)


# Load star wars data
sw = read.csv('star-wars.csv', sep=";")

# Transform data to matrix form
sw_m = as.matrix(sw)

# Convert to a graph
sw_graph = graph.edgelist(sw_m, directed = FALSE)

# Visualize with a plot
tkplot(sw_graph, vertex.color = 'green')

# Calculate different centrality measures

# Degree centrality
sw_degree = degree(sw_graph)


# Closeness
sw_closeness = calc_closseness(get.adjacency(sw_graph))


# Betweennes
sw_betweenness = betweenness(sw_graph)


# PageRank

# Create transformation matrix
sw_adj_m = as.matrix(get.adjacency(sw_graph))
col_s <- colSums(sw_adj_m)

col_s

tf = c()
for(i in 1:ncol(sw_adj_m)){
  tf <- cbind(tf, sw_adj_m[,i] / col_s[i])
}

# Starting point is taken from an equal distribution
prob_d <- rep(1/nrow(sw_adj_m), nrow(sw_adj_m))

prob_d


# Calculate pageRank values with the custom function
sw_pageRank = pg_alg(tf, 150)


# Compare values with in-built method
sw_pageRank2 = page.rank(sw_graph)



# Compare the results

# First create a dataframe with three most central characters according to each method of centrality
df = data.frame(sort(sw_degree, decreasing = TRUE)[1:3],
                sort(sw_closeness, decreasing = TRUE)[1:3],
                sort(sw_betweenness, decreasing = TRUE)[1:3],
                sort(sw_pageRank, decreasing = TRUE)[1:3])

colnames(df) <- c("degree", "closeness", "betweenness", "pageRank")


# Compare the methods using the Pearson correlation matrix

df_all = data.frame(sw_degree,  sw_closeness,sw_betweenness, sw_pageRank)
colnames(df_all) <- c("degree", "closeness", "betweenness", "pageRank")

# Create the Pearson correlation matrix
PCM = cor(df_all)

pairs(~sw_degree + sw_closeness + sw_betweenness + sw_pageRank,
      data = df_all, main = "Scatterplot Matrix")





# --------------------------------- #
# ------------ TASK 3 ------------- #
# --------------------------------- #


# Clear workspace
rm(list = ls(all=T))


# Load customer file
dataTrain = read.csv("data_assign.csv", sep = ";")



# Create a neural network for making the predictions
# For this we use caret-library
library(caret)
library(caTools) # For splitting the training data


# Exclude X-column
dataTrain = dataTrain[ , !(names(dataTrain) %in% "X")]

# Normalize numeric features
library(dplyr)
dataTrain = dataTrain %>% mutate_each_(funs(scale(.) %>% as.vector), 
                           vars=c("age", "salary"))


# factorize the output class
dataTrain$recommendation = factor(dataTrain$recommendation) 

set.seed(100) # For reproducibility



split = sample.split(dataTrain$recommendation, SplitRatio = 0.75)

# Create subsets for testing and training the neural network implementation
train_data = subset(dataTrain, split == TRUE)
test_data = subset(dataTrain, split == FALSE)


# Fit a neural network for predictions
nn_fit = train(recommendation ~ ., data = train_data, method = 'nnet')


pred = predict(nn_fit, test_data)
confusionMatrix(pred, test_data$recommendation)


# Load test data
testData = read.csv("data_assign_new.csv", sep= ";")
colnames(testData) = c("X","gender","age", "edu", "salary", "previous.orders","previous.purchase", "favourite.genre" )

# Exclude X-column
testData = testData[ , !(names(testData) %in% "X")]

# Normalize numeric features
library(dplyr)
testData_norm = testData %>% mutate_each_(funs(scale(.) %>% as.vector), 
                                       vars=c("age", "salary"))

# Create predictions for testData
testPred = predict(nn_fit, testData_norm)

# Bind predictions to the testData data frame
testData = cbind(testData, testPred)



# --------------------------------- #
# ------------ TASK 4 ------------- #
# --------------------------------- #

# Clear workspace
rm(list = ls(all=T))



# Load libraries
library(qdap)
library(tm)
library(wordcloud)
library(dendextend)
library(ggplot2)
library(ggthemes)

# Function for cleaning and creating a corpus 

create_corpus <- function(text){
  # Convert text data to corpus using tm-library
  # First we need to convert text to a vector source
  source = VectorSource(text)
  
  # Next we convert the source to a corpus (using VCorpus ~ volatile corpus)
  corpus = VCorpus(source)
  
  # Clean the corpus
  corpus <- tm_map(corpus, removePunctuation)# remove punctuations
  corpus <- tm_map(corpus, stripWhitespace) # remove white spaces 
  corpus <- tm_map(corpus, content_transformer(tolower)) #changing everything to lowercase
  corpus <- tm_map(corpus, removeWords, c(stopwords("en"))) # Remove stopwords (words frequently used but have small amount of information)
  corpus <- tm_map(corpus, removeNumbers) # remove numbers
  return(corpus)
}



# Load text data
text = read.csv("News-Final.csv", stringsAsFactors = FALSE)

# Using only headlines for the text analysis
headlines = text$Headline[1000:2500] # USe only a proportion of the entries to manage the size


# Create a clean corpus using create_corpus-function
head_cleanCorpus = create_corpus(headlines)



# Analyze term frequencies using a document-term matrix
head_dtm = DocumentTermMatrix(head_cleanCorpus)

# Convert this to matrix 
head_m <- as.matrix(head_dtm)

# Analyze the frequency of each term by looking at the total amount term is used in the 
# documents. We can easily calculate this by taking the sum of each column from the 
# document-term matrix

freq = colSums(head_m)
freq_sorted = sort(freq, decreasing = T)


# Create a bar plot of the 30 most common words

barplot(freq_sorted[1:30], col = "tan", las = 2)



# For the next iteration we'll use the knowledge gained to stem the words
# and to remove stopwords that are not relevant or create little information

# Start with word stemming

# Stem document by first creating a list of included words
rm_punc = removePunctuation(headlines) # Remove punctuations from text
head_list = unlist(strsplit(rm_punc, split = ' '))# Creates a list of words

# Library SnowballC is needed for word stemming
library(SnowballC)

# Stem the list of words created previously
head_stemmed = stemDocument(head_list)


# USe stemCompletion to complete the word endings, as the dictionary we use a copy of the original list of words
head_completed = stemCompletion(head_stemmed, head_list, type = "prevalent")

# Create a corpus of the stemmed words
corpus_stemmed = create_corpus(head_completed)

# On the basis of the last analysis we can remove some frequently occurring words that convey little information
# These added words are taken from the 250 most frequently occurring words
corpus_stemmed <- tm_map(corpus_stemmed, removeWords, c(stopwords("en"), "said", "one", "last", "percent", 
                         "monday", "tuesday", "first","year", "now","november", "wednesday", "week", "friday", "today", 
                         "two", "since", "still", "may", "back", "country", "house", "also", "month", "three", "day",
                         "use", "october", "see","say", "saturday","week", "small", "another","will"))


# NExt analyze the term frequencies with these modifications
stemmed_dtm = DocumentTermMatrix(corpus_stemmed)

# Convert this to matrix 
stemmed_m <- as.matrix(stemmed_dtm)

# Analyze the frequency of each term by looking at the total amount term is used in the 
# documents. We can easily calculate this by taking the sum of each column from the 
# document-term matrix

freq_stemmed = colSums(stemmed_m)
freq_stem_sorted = sort(freq_stemmed, decreasing = T)


freq_stem_sorted[1:50]

# Create a bar plot of the 30 most common words

barplot(freq_stem_sorted[1:30], col = "tan", las = 2)




# Make second round of iteration based on the results from the last round
# REmove some frequent words that are not relevant



corpus_stemmed2 <- tm_map(corpus_stemmed, removeWords, c(stopwords("en"), "users", "quarter", "cup", 
                                                         "qualifier", "next", "time", "start", "play", "released", 
                                                         "take", "according", "thursday", "can", "make", "says", "new",
                                                         "users", "showed", "announced", "centers", "report", "nov", "countries",
                                                         "plans", "store", "third", "rate", "just"))




# Create additional control feature to limit the amount of word length and count only words that are present
# in 5 to 1000 documents
stemmed_dtm2 = DocumentTermMatrix(corpus_stemmed2, control=list(wordLengths=c(3, 20),
                                                              bounds = list(global = c(5,1000))))

# Same steps as before

stemmed_m2 <- as.matrix(stemmed_dtm2)
freq_stemmed2 = colSums(stemmed_m2)
freq_stem_sorted2 = sort(freq_stemmed2, decreasing = T)
freq_stem_sorted2[1:50]
barplot(freq_stem_sorted2[1:30], col = "tan", las = 2)



# Create a unified plot of the three different cases
par(mfrow = c(3,1))
barplot(freq_sorted[1:10], col = "tan", las = 2, cex.names=1.5, main = "Basic cleaning and stopwords")
barplot(freq_stem_sorted[1:10], col = "tan", las = 2,cex.names=1.5, main = "Iteration 1: stemming")
barplot(freq_stem_sorted2[1:10], col = "tan", las = 2,cex.names=1.5, main = "Iteration 2: Added stopwords")



# Create a wordcloud
set.seed(1000)
dev.off() # Erase the par-setting for plots
wordcloud(names(freq_stemmed2),freq_stemmed2, min.freq=35,max.words = 30, colors=brewer.pal(6,"Dark2"))




# Topic modelling
library("topicmodels")



# Topic modelling data
  # head_m / only cleaned data and stopword removal
  # head_removed / Create a dataset where the extra stopwords are removed

head_removed <- tm_map(head_cleanCorpus, removeWords, c(stopwords("en"), "said", "one", "last", "percent", 
                                                        "monday", "tuesday", "first","year", "now","november", "wednesday", "week", "friday", "today", 
                                                        "two", "since", "still", "may", "back", "country", "house", "also", "month", "three", "day",
                                                        "use", "october", "see","say", "saturday","week", "small", "another", "will", "users", "quarter",
                                                        "cup","qualifier", "next", "time", "start", "play", "released", "take", "according", "thursday",
                                                        "can", "make", "says", "new", "third", "announced", "book", "people", "expected", 
                                                        "just"))

head_removed_dtm = DocumentTermMatrix(head_removed)
head_removed_m = as.matrix(head_removed_dtm)


head_topic = LDA(head_m, k = 3, control = list(seed = 1985))
as.matrix(terms(head_topic,10))

head_removed_topic = LDA(head_removed_m, k = 4, control = list(seed = 1985))
as.matrix(terms(head_removed_topic,10))
