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

