from flask import Flask, render_template, request
import pandas as pd
import re
import profanity_check
import transformers
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, col, size

app = Flask(__name__)
spark = SparkSession.builder.getOrCreate()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No file selected.')

# Read the Excel file into a Pandas DataFrame
        df = pd.read_excel(file)

        # Apply the program and store the counts in a new column
        df['text1'] = df['SENTENCE'].apply(lambda text: re.sub(r"[^a-zA-Z0-9]", " ", str(text)))
        df['censored_words'] = df['text1'].apply(lambda text: [word for word in text.split() if profanity_check.predict([word])[0] == 1])
        from transformers import pipeline 
        model=pipeline('sentiment-analysis')
        df['sentiment']=df['SENTENCE'].apply(lambda text: model(str(text))[0]['label'])
        df['sentimentscore']=df['SENTENCE'].apply(lambda text: model(text)[0]['score'])
        # Convert Pandas DataFrame to PySpark DataFrame
        spark_df = spark.createDataFrame(df)

        # Calculate the count of words in the 'censored_words' column
        countdf = spark_df.withColumn("count_words", size(col("censored_words")))

 # Show the PySpark DataFrame
        result = countdf.toPandas().to_html(index=False)

        return render_template('index.html', result=result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)