setwd('..\\Desktop\\DSI Jatim Camp 3 Sentiment\\dataset\\Dataset')
################################################################

library(tm)
df=read.csv('garuda.csv', sep = ';')

#####Merubah tipe data
corpus=Corpus(VectorSource(df$text))

#####Preprocessing
#case_folding
case_folding=tm_map(corpus, tolower)
#menghapus angka
no_number=tm_map(case_folding, removeNumbers)
#menghapus tanda baca
no_punctuations=tm_map(no_number, removePunctuation)
#menghapus whitespace
no_whitespace=tm_map(no_punctuations, stripWhitespace)

#stopwords
no_stopwords=tm_map(no_whitespace, removeWords, stopwords('english'))
no_stopwords=tm_map(no_stopwords, removeWords, c('flight','garuda'))

#Stemming
stemmed=tm_map(no_stopwords, stemDocument)

#tokenizing
tokenizer=function(x) strsplit(x, split = ' ')
token=tm_map(stemmed, tokenizer)

#CountVectorizer
dtm=DocumentTermMatrix(token)
inspect(dtm)

#TFIDF
weight_tfidf=function(x) weightTfIdf(x)
dtmTFIDF=DocumentTermMatrix(token, control = list(weighting=weight_tfidf))
inspect(dtmTFIDF)

#Visualisasi###
library(ggplot2)

##Kata paling sering muncul
freq=sort(colSums(as.matrix(dtm)), decreasing = T)
head(freq,20)


wf=data.frame(word=names(freq), freq=freq)

##bar plot
ggplot(subset(wf, freq>300), aes(x=reorder(word, -freq), y=freq))+
    geom_bar(stat='identity')

##Wordcloud
library(wordcloud)
win.graph()
wordcloud(names(freq), freq = freq, min.freq = 100, 
          random.order = T, colors = brewer.pal(8, 'Dark2'))

###wordcloud sentimen
m=as.matrix(dtm)
reviewPositif=m[which(df$sentiment=='POSITIVE'),]
reviewNegatif=m[which(df$sentiment=='NEGATIVE'),]

freqPositif=sort(colSums(reviewPositif), decreasing = TRUE)
freqNegatif=sort(colSums(reviewNegatif), decreasing = TRUE)

win.graph()
wordcloud(names(freqPositif), freq = freqPositif, min.freq = 100, 
          random.order = T, colors = brewer.pal(8, 'Dark2'))
win.graph()
wordcloud(names(freqNegatif), freq = freqNegatif, min.freq = 10,
          random.order = T, colors = brewer.pal(8, 'Dark2'))



##bar plot positif
wfpositif=data.frame(word=names(freqPositif), freq=freqPositif)
ggplot(subset(wfpositif, freq>300), aes(x=reorder(word, -freq), y=freq))+
  geom_bar(stat='identity')

##bar plot negatif
wfnegatif=data.frame(word=names(freqNegatif), freq=freqNegatif)
ggplot(subset(wfnegatif, freq>30), aes(x=reorder(word, -freq), y=freq))+
  geom_bar(stat='identity')


#Train-test split
library(caret)
library(e1071)

TrainIndex=createDataPartition(y=df$sentiment, p=0.7, list = FALSE)
TrainData=cbind(as.data.frame(m[TrainIndex,]), sentiment=df$sentiment[TrainIndex])
XTest=as.data.frame(m[-TrainIndex,])
YTest=as.factor(df$sentiment[-TrainIndex])

#Model
model=naiveBayes(formula=as.factor(TrainData$sentiment)~., data = TrainData)
prediksi=predict(model,XTest)
confusionMatrix(prediksi, YTest, mode='everything')

##buang kata2 jarang muncul, avoid overfitting
dtm1=removeSparseTerms(dtm, 0.80)
inspect(dtm1)
m1=as.matrix(dtm1)


TrainIndex=createDataPartition(y=df$sentiment, p=0.7, list = FALSE)
TrainData=cbind(as.data.frame(m1[TrainIndex,]), sentiment=df$sentiment[TrainIndex])
XTest=as.data.frame(m1[-TrainIndex,])
YTest=as.factor(df$sentiment[-TrainIndex])

model=naiveBayes(formula=as.factor(TrainData$sentiment)~., data = TrainData)
prediksi=predict(model,XTest)
confusionMatrix(prediksi, YTest, mode='everything')

###COba Cluster
library(cluster)
distance=dist(t(dtm1),method = 'euclidian')
fit=hclust(distance, method = 'complete')
plot(fit)
rect.hclust(fit, k=6, border = 'red')

library(fpc)   
kfit <- kmeans(distance, 5)   
clusplot(as.matrix(distance), kfit$cluster, color=T, shade=T, labels=2, lines=0)   

##Menggabungkan semua file csv dalam satu folder jadi satu dataframe
file_list=list.files( pattern = ".csv")
file_list
ALLAirlines=do.call(rbind, lapply(file_list, function(x) read.csv(x, sep=';')))
dim(ALLAirlines)
