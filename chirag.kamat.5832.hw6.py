from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from collections import Counter
import numpy as np



lines = [line.strip() for line in open('wsj00-18.tag')] #Reading the Penn Tree bank line by line and adding them to lines list

words = [(line.split('\t')[0]).lower() for line in lines if '\t' in line] #Contains list of words from PTB excluding the tags
unigrams = Counter(words) # This will give all the unique words and their counts
bigrams = Counter() #This is to calculate the count for prev word and current word

common_word_tuple_dict = {}
top_1000 = [tuple[0] for tuple in unigrams.most_common(1000)] #Taking the top 1000 frequently occuring words

for word,next_word in zip(words,words[1:]):
    bigrams[(word,next_word)] += 1 #Computing the counts for all the bigrams in the PTB

vocab = unigrams.keys() # vocab contains all the unique words required for creating the left and right vectors


vector_list = []

'''This is the main loop to create the left and right vectors for each most common word
. Left vector is created by getting the bigram counts of all those items which have words appearing to the left of common word.
Similarly for the right vector. And then these two vectors are concatenated and a final vector list is built where each element
of this final list contains a concatenated left and right vector having a total length of 77,148'''
for common_word in top_1000:
    left = []
    right = []
    for word in vocab:
        left.append(bigrams.get((word,common_word),0.0))
        right.append(bigrams.get((word,common_word),0.0))

    vector = np.concatenate((left,right))
    vector_list.append(vector)


print "vector created\n"
arr = normalize(vector_list,norm='l2') #normalizing the final vector with l2 norm. Can be changed to l1, but I
# was getting better clusters for l2
kmeans = KMeans(n_clusters=25, random_state=0).fit(arr) #Passing the vector as an input to the Kmeans algorithm
print "kmeans labels are {} \n".format(kmeans.labels_) # Printing all the kmeans labels for each of the 1000 common words


'''Finally the below two loops will help print all the words belonging to indiividual cluster based on the kmeans labels list'''
clusters = {}
for cluster,word in zip(kmeans.labels_,top_1000):
    if not clusters.has_key(cluster):
        clusters.update({cluster:word})
    else:
        clusters[cluster] += ' ' + word

for i in clusters.keys():
    print i,clusters[i]