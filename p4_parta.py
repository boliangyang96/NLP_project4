import csv
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk

def fromFile(filename):
    sentences = []
    options = []
    labels = []
    with open(filename, encoding='latin-1') as f:
        lines = csv.reader(f)
        i = 0
        # line[0] = id, line[1-4] = 4-sentences, line[5] = option 1, line[6] = option 2, line[7] = label
        for line in lines:
            i += 1
            if i == 1: continue
            sentences.append([line[1],line[2],line[3],line[4]])
            options.append([line[5].lower(),line[6].lower()])
            labels.append(int(line[7]))
    return sentences, options, labels

def fromTestFile(filename):
    id_ = []
    sentences = []
    options = []
    with open(filename, encoding='latin-1') as f:
        lines = csv.reader(f)
        i = 0
        for line in lines:
            i += 1
            if i == 1: continue
            id_.append(line[0])
            sentences.append([line[1],line[2],line[3],line[4]])
            options.append([line[5].lower(),line[6].lower()])
    return id_, sentences, options

# features: length of the sentence, sentiment_score of the sentnece, part_of_speech tag sequence, word sequence
def createFeatures(train_options):
    features = []
    analyser = SentimentIntensityAnalyzer()
    for i in range(len(train_options)):
        option1 = train_options[i][0]
        option2 = train_options[i][1]
        temp_feature = dict()
        temp_feature["length"] = len(option1)
        temp_feature["sentiment_score"] = analyser.polarity_scores(option1)['pos']
        pos = nltk.pos_tag(nltk.word_tokenize(option1))

        for i in range(len(pos)):
            temp_feature["pos=%s"%(pos[i][1])] = True
            temp_feature["word=%s"%(pos[i][0])] = True
        features.append(temp_feature)

        temp_feature = dict()
        temp_feature["length"] = len(option2)
        temp_feature["sentiment_score"] = analyser.polarity_scores(option2)['pos']
        pos = nltk.pos_tag(nltk.word_tokenize(option2))

        for i in range(len(pos)):
            temp_feature["pos=%s"%(pos[i][1])] = True
            temp_feature["word=%s"%(pos[i][0])] = True
        features.append(temp_feature)
        # print(features)
        # break
    return features

def processLabels(train_labels):
    labels = []
    for i in range(len(train_labels)):
        if (train_labels[i] == 1):
            labels.append(1)
            labels.append(0)
        else:
            labels.append(0)
            labels.append(1)
    return labels
    

if __name__ == "__main__":
    train_sentences, train_options, train_labels = fromFile('train.csv')
    train_features = createFeatures(train_options)
    labels = processLabels(train_labels)
    ## use logistic regression model
    model = LogisticRegression(multi_class='auto')

    ## use dictvectorizer to transform features
    vector = DictVectorizer(sparse=False)
    train_features_trans = vector.fit_transform(train_features)
    ## train model
    clf = model.fit(train_features_trans, labels)

    dev_sentences, dev_options, dev_labels = fromFile('dev.csv')
    dev_features = createFeatures(dev_options)
    dev_features_trans = vector.transform(dev_features)

    predict_labels = []
    for i in range(len(dev_labels)):
        score1 = clf.predict_proba(dev_features_trans[2*i].reshape(1, -1))
        score2 = clf.predict_proba(dev_features_trans[2*i+1].reshape(1, -1))
        if (score1[0][1] > score2[0][1]): predict_labels.append(1)
        else: predict_labels.append(2)
    
    count = 0
    for i in range(len(dev_labels)):
        if predict_labels[i] == dev_labels[i]:
            count += 1
    print("Validation accuracy: ", count/len(dev_labels))
    print("Weights: ", clf.coef_)

    '''
    test_id, test_sentences, test_options = fromTestFile('test.csv')
    test_features = createFeatures(test_options)
    test_features_trans = vector.transform(test_features)
    predict_test_labels = []
    for i in range(len(test_id)):
        score1 = clf.predict_proba(test_features_trans[2*i].reshape(1, -1))
        score2 = clf.predict_proba(test_features_trans[2*i+1].reshape(1, -1))
        if (score1[0][1] > score2[0][1]): predict_test_labels.append(1)
        else: predict_test_labels.append(2)
    outputFile = open('test_pred.csv', 'w')
    outputFile.write('Id,Prediction\n')
    for i in range(len(test_id)):
        outputFile.write(test_id[i]+','+str(predict_test_labels[i])+'\n')
    outputFile.close()'''
    
