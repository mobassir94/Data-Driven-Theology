# Data-Driven-Theology

# Motivation

The main aim of this repository is to find similarities between different religious scriptures and make AI driven theological similarity search engine app. specially for Holy Quran and Bible.

column names starting with t_ means those columns belongs to Torah and q_ means those columns are for Holy Quran.

the final processed similarity pair dataset is public here for further research -> https://www.kaggle.com/datasets/mobassir/quran-vs-bible

# Supported Input Languages

Afrikaans, Albanian, Amharic, Arabic, Armenian, Aymara, Azerbaijani, Basque, Belarusian, Bengali, Berber languages, Bosnian, Breton, Bulgarian, Burmese, Catalan, Central/Kadazan Dusun, Central Khmer, Chavacano, Chinese, Coastal Kadazan, Cornish, Croatian, Czech, Danish, Dutch, Eastern Mari, English, Esperanto, Estonian, Finnish, French, Galician, Georgian, German, Greek, Hausa, Hebrew, Hindi, Hungarian, Icelandic, Ido, Indonesian, Interlingua, Interlingue, Irish, Italian, Japanese, Kabyle, Kazakh, Korean, Kurdish, Latvian, Latin, Lingua Franca Nova, Lithuanian, Low German/Saxon, Macedonian, Malagasy, Malay, Malayalam, Maldivian (Divehi), Marathi, Norwegian (Bokmål), Occitan, Persian (Farsi), Polish, Portuguese, Romanian, Russian, Serbian, Sindhi, Sinhala, Slovak, Slovenian, Somali, Spanish, Swahili, Swedish, Tagalog, Tajik, Tamil, Tatar, Telugu, Thai, Turkish, Uighur, Ukrainian, Urdu, Uzbek, Vietnamese, Wu Chinese and Yue Chinese.

# Supported Output Languages

English only

# Solution / Pipeline

1. from [bibleverses](https://www.kaggle.com/datasets/phyred23/bibleverses) dataset we first separate old testament

2. for each verse in old testament we compare similarity of each and every verse of the Holy Quran.

3. for [Finding Similar Verses between Quran and Torah](https://github.com/mobassir94/Data-Driven-Theology/tree/main/Quran%20vs%20Bible) we used popular [SentenceTransformers](https://www.sbert.net/) models and they are all-MiniLM-L6-v2,msmarco-distilbert-base-tas-b,all-mpnet-base-v2,all-MiniLM-L12-v2 and google's [Universal Sentence Encoder Large](https://tfhub.dev/google/universal-sentence-encoder-large/5) model. (NOTE : we used different data processing with all-MiniLM-L12-v2 model).

4. ENSEMBLE : we pick the verses pairs from each dataframe (output of each model discussed in step 3) with highest confident score (we take the decision based on histogram score distribution), then we merge all the filtered dataset and drop duplicates except first. then we sort the final filtered dataset and save that dataset for next level research. check the ensemble notebook called [find similar verses between quran and bible](https://github.com/mobassir94/Data-Driven-Theology/blob/main/Quran%20vs%20Bible/find-similar-verses-between-quran-and-bible.ipynb)  for better understanding.

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


# Acknowledgements

1. Apsis Solutions Ltd.
2. bengali.ai


