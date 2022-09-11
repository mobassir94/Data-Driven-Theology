# Data-Driven-Theology


The main aim of this repository is to find similarities between different religious scriptures and make AI driven theological similarity search engine app. specially for Holy Quran and Bible.

column names starting with t_ means those columns belongs to Torah and q_ means those columns are for Holy Quran.

the final processed similarity pair dataset is public here for further research -> https://www.kaggle.com/datasets/mobassir/quran-vs-bible


# Motivation

Finding Similar Verses among different religious scriptures is very important so that we all can follow at least what is common religious laws that we all need to follow properly. but to the best of our knowledge even in today's age No such data driven research project has been developed yet for public use (At least we couldn't find similar work that is publicly available). For that reason,Multilingual Data Driven theology project has been created here as a very basic step to find out which verses are there in the Quran and which are in Old testament that matches with each other and also closest to users Query(in terms of word similarity/not verse similarity always, we are yet to figure out verse similarity (in terms of meaning) as it is a very complex problem that needs to be solved).

And due to the absence of this search engine, even in today's age sectarian hatred keeps on increasing and IMHO it is like "Drinking the poison and expecting someone else to die". This is why this project is undertaken and hoping to mitigate sectarian hatred as much as we can.The aim of this project is to create a search engine that will take input queries for Holy Quran and Bible(Torah) from any language like Arabic, Hindi, Bengali, English etc. (It also supports code mixed language input i.e. mix of English and Arabic input.) and Provide output in  English language that shows top verse pairs among Quran and Bible. This is an "asymmetric semantic similarity search" problem.Meaning, you usually have a short question (like a question or some keywords) and you want to find a long paragraph answering the Question/Query.


There are a few things to keep in mind when using the project in its current state.For example and they are discussed in very next section.


# limitations/ cautions

1. This is not a question answering system,hence it won't give explicit answer for questions (Question-Answering System needs labeled dataset,which is currently 
missing for Holy Quran and Old Testament.)

2. The system tries to predict closest verses from Quran or Torah  for your query using some mathematical/statistical process. So predicted answer won't be always right.

3. If the user query contains spelling mistake or incorrect words then predictions of the system is expected to be wrong for most of the times.

4. if the users query contains irrelevant words and doesn't come closer to any verse in terms of semantic similarity then the system will provide wrong results.

5. Most of the times Pairwise Euclidean metric provides better results than dot product metric.

6. This system can't consider the domain of the Query.

7. Higher Prediction Score indicates higher similarity,if prediction score for your query is small,then try to use different keywords/queries instead

8. The meaning of selected verse pairs won't be same/similar for most of the times but mostly they are expected to contain same or similar words.

9. Predicted results might not always give you appropriate results for your input query,but most of the times it will try to output something that contains words or
keywords that are very close to your input query.

10. This is a zero Shot learning approach because we are lacking labeled dataset for this task,to improve performance of this system,one need to further finetune LASER
and improve the Ensemble Pipeline



# Supported Input Languages

Afrikaans, Albanian, Amharic, Arabic, Armenian, Aymara, Azerbaijani, Basque, Belarusian, Bengali, Berber languages, Bosnian, Breton, Bulgarian, Burmese, Catalan, Central/Kadazan Dusun, Central Khmer, Chavacano, Chinese, Coastal Kadazan, Cornish, Croatian, Czech, Danish, Dutch, Eastern Mari, English, Esperanto, Estonian, Finnish, French, Galician, Georgian, German, Greek, Hausa, Hebrew, Hindi, Hungarian, Icelandic, Ido, Indonesian, Interlingua, Interlingue, Irish, Italian, Japanese, Kabyle, Kazakh, Korean, Kurdish, Latvian, Latin, Lingua Franca Nova, Lithuanian, Low German/Saxon, Macedonian, Malagasy, Malay, Malayalam, Maldivian (Divehi), Marathi, Norwegian (Bokmål), Occitan, Persian (Farsi), Polish, Portuguese, Romanian, Russian, Serbian, Sindhi, Sinhala, Slovak, Slovenian, Somali, Spanish, Swahili, Swedish, Tagalog, Tajik, Tamil, Tatar, Telugu, Thai, Turkish, Uighur, Ukrainian, Urdu, Uzbek, Vietnamese, Wu Chinese and Yue Chinese.

# Supported Output Languages

English only

# Solution / Pipeline

1. from [bibleverses](https://www.kaggle.com/datasets/phyred23/bibleverses) dataset we first separate old testament

2. for each verse in old testament we compare similarity of each and every verse of the Holy Quran.

3. for [Finding Similar Verses between Quran and Torah](https://github.com/mobassir94/Data-Driven-Theology/tree/main/Quran%20vs%20Bible) we used popular [SentenceTransformers](https://www.sbert.net/) models and they are all-MiniLM-L6-v2,msmarco-distilbert-base-tas-b,all-mpnet-base-v2,all-MiniLM-L12-v2 and google's [Universal Sentence Encoder Large](https://tfhub.dev/google/universal-sentence-encoder-large/5) model. (NOTE : we used different data processing with all-MiniLM-L12-v2 model).

4. ENSEMBLE : we pick the verses pairs from each dataframe (output of each model discussed in step 3) with highest confident score (we take the decision based on histogram score distribution), then we merge all the filtered dataset and drop duplicates except first. then we sort the final filtered dataset and save that dataset for next level research. check the ensemble notebook called [find similar verses between quran and bible](https://github.com/mobassir94/Data-Driven-Theology/blob/main/Quran%20vs%20Bible/find-similar-verses-between-quran-and-bible.ipynb)  for better understanding.

5. we use final ensembled dataset (verses pairs) for creating Multilingual Laser embedding/Corpus embedding.

6. we convert users queries into Multilingual Laser embedding/Query Embedding and find similarity with Corpus embedding using l2 distance or dot product

[COLAB DEMO](https://github.com/mobassir94/Data-Driven-Theology/blob/main/Quran%20vs%20Bible/Inference_Multilingual_Data_Driven_Theology.ipynb)

# How to install the app?

1. go to terminal and cd to the root directory of this project,then

2. !pip install -r requirements.txt (make sure no error occured) then,

3. python app.py

4. Now,go to browser and hit -> http://127.0.0.1:33507/

demo video tutorial is available here -> https://www.youtube.com/watch?v=sSnsOePvy24&t=12s

# Few Words

To the best of our knowledge this is the first approach ever taken for finding similar verse pairs among different religious scriptures using cross lingual data driven theological approach.This is a research project that was developed and shared publicly instantly without having the mission or vision of publishing it or updated version of it in any journal or conference because we don't care about having name on any research paper and Allah is The Most Watcher On us In Every Moments Of our Breath.
here are few ideas that we would like to share for you to take this project even further and if you are doing similar research project,you can let us know and we are ready to help as much as we can WITHOUT NEEDING TO HAVE OUR NAMES IN ANY RESEARCH PAPER.

idea1 -> we used single english translated version of Holy Quran and Torah, try to merge translations from different sources like sahih international,al bayan etc. more translated verses created better diversity and should help improve ensemble pipeline.

idea2 -> try laser3 as discussed in No language left behing(nllb) -> https://ai.facebook.com/blog/nllb-200-high-quality-machine-translation/

idea3 -> Include Multilingual Embeddings like meta's Laser and multilingual USE(universal sentence embedding) in ensemble pipeline,again it should add diversity in ensemble and should help us find strong similar verse pairs

idea4 -> once you make ensemble pipeline robust, try to replace laser with powerful english only embedder like google's USE large variant for doing similarity search and observe if it's better than multilingual Laser or not for this task?

idea 5 -> finding appropriate dataset is the key here,see if you can manually label few verses that are similar in both quran and torah.

idea 6 -> extend this research for other religious scriptures as well and observe the differences

idea 7 -> maybe models pretrained only on religious scriptures will do much better job? If You Never Try You'll Never Know. 

# References

1. https://arxiv.org/pdf/1812.10464v2.pdf
2. https://www.sbert.net/
3. https://github.com/mobassir94/Multilingual-NLP-for-Islamic-Theology/blob/main/README.md

# Acknowledgements

1. Apsis Solutions Ltd.
2. bengali.ai


