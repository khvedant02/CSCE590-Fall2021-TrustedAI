Data Used:

1. Tweet-IDs: These tweet IDs are u.sed to hydrate tweets related to COVID-19. These tweet Ids are obtained from [1] which contains an ongoing collection of tweet ids since 28th Jan 2020.
2. Government Policies: These are data related to several government policies implemented during the pandemic [2,3,4,5]
3. Diagnostic and Statistical Manual of Mental Disorders (DSM-5): Also known as the bible for psychologists, this resource is used to obtain mental health and drug abuse-related entities.[6]
4. Drug Abuse Ontology (DAO): This ontology is used to obtain mental health and drug abuse-related entities and their usage in social media data and other platforms. [7]
5. GeoNames Ontology: This ontology is used to obtain granular location information about different US states, further used to tag different tweets with location data. [8]
6. Dbpedia: This is a general-purpose knowledge graph that enriches the lexicon of the mental health and drug abuse entities. [9]
7. Subreddit: These are posts and their top comments from Depression, Addiction and Anxiety subreddits. [10]

Files and their details:

1. FilteringLocationSemantic.py : This file contains the code to perform the semantic and location based filtering on the input tweets objects and return and save the filtered in tweet objects in a file. 
2. IndexValues.py : This file contains the code to perform the calcualtion of Index scores for Depression, Addiction and Anxiety. The index scores so calculated are added to the input tweet objects and saved in a file. 
3. NGramTopicModelsSubreddir.py : This file contains the code to perform the NGram topic modelling for Depression Addiction and Anxiety Subreddit. These models are further used for the index score calculations. 
4. SEDOMatrix.py : This file contains the code to obtain the weight matrix from the process of Semantic Encoding and Decoding Optimization. The matrix obtained if further used to modulate the word embeddings. 
5. SubRedditLanguageModels.py : This file contains the code to perform language modelling (word2vec) on ngrams of Depression, Addiction and Anxiety Subreddit. These models are further used for the index score calculations. 
6. SubredditTopicModels.py : This file contains the code to perform topic modelling for Depression, Addiction and Anxiety Subreddit. These models are further used for the index score calculations. 
7. TopicModelsTweets.py : This file contains the code to perform topic modelling on tweets for categories Depression, Addiction and Anxiety. These models are further used for the index score calculations. 
8. TrainingAllClassifieris.py : This file contains the code to perform the process of modulating word embeddings of the tweets based on the SEDO weight matrix and further give those embeddings as input to various classifiers with target as the index scores.

References:

[1] echen102,  “Echen102/covid-19-tweetids:  The  repository  contains  anongoing  collection  of  tweets  ids  associated  with  the  novel  coronaviruscovid-19   (sars-cov-2),   which   commenced   on   january   28,   2020.”[Online]. Available: https://github.com/echen102/COVID-19-TweetIDs. <br />
[2] OxCGRT,  “Oxcgrt/covid-policy-tracker:  Systematic  dataset  of  covid-19    policy,    from    oxford    university.”    [Online].    Available:    https://github.com/OxCGRT/covid-policy-tracker. <br />
[3] MultiState,   “Covid-19   policy   tracker.”   [Online].   Available:   https://www.multistate.us/issues/covid-19-policy-tracker. <br />
[4] Published:   Dec   02,   “State   covid-19   data   and   policy   actions,”Dec  2021.  [Online].  Available:  https://www.kff.org/coronavirus-covid-19/issue-brief/state-covid-19-data-and-policy-actions/. <br />
[5] Covid-19  recovery  -  united  states  department  of  state,”  Nov  2021.[Online]. Available: https://www.state.gov/covid-19-recovery/. <br />
[6] Diagnostic   and   statistical   manual   of   mental   disorders   (dsm–5).”[Online].   Available:   https://www.psychiatry.org/psychiatrists/practice/dsm. <br />
[7] U.   Lokala,   R.   Daniulaityte,   F.   Lamy,   M.   Gaur,   K.   Thirunarayan,U.  Kursuncu,  and  A.  P.  Sheth,  “Dao:  An  ontology  for  substance  useepidemiology on social media and dark web,”JMIR Public Health andSurveillance, 2020. <br />
[8] Online]. Available: https://www.geonames.org/ontology/documentation.html. <br />
[9] S. Auer, C. Bizer, G. Kobilarov, J. Lehmann, R. Cyganiak, and Z. Ives,“Dbpedia:  A  nucleus  for  a  web  of  open  data,”  inThe  semantic  web.Springer, 2007, pp. 722–735. <br />
[10] “Home  feed  subreddits  (0).”  [Online].  Available:  https://www.reddit.com/subreddits/. <br />
