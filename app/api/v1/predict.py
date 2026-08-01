import io
import re
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from wordcloud import WordCloud
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import joblib
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv
import pickle
import os

# Load environment variables from .env file
load_dotenv()

# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token



dagshub_url = "https://dagshub.com"
repo_owner = "iamdebasishdas123"
repo_name = "YouTube-Mood-Tracker"

app = FastAPI(title="YouTube Comment Sentiment Analysis API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Schemas for Data Validation ---
class CommentItem(BaseModel):
    text: str
    timestamp: str

class PredictWithTimestampsInput(BaseModel):
    comments: List[CommentItem]

class PredictInput(BaseModel):
    comments: List[str]

class ChartInput(BaseModel):
    sentiment_counts: Dict[str, int]

class TrendInput(BaseModel):
    sentiment_data: List[Dict[str, Any]]


# --- Preprocessing & Model Loading ---
def preprocess_comment(comment: str) -> str:
    try:
        comment = comment.lower().strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)
        
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])
        
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])
        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment

def load_model_and_vectorizer(model_name: str, model_version: str, vectorizer_path: str):
    mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
    client = MlflowClient()
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

# Initialize model and vectorizer
model, vectorizer = load_model_and_vectorizer("yt_chrome_plugin_model", "1", "./tfidf_vectorizer.pkl")


# --- Routes ---
@app.get('/')
def home():
    return {"message": "Welcome to our FastAPI"}

@app.post('/predict_with_timestamps')
def predict_with_timestamps(data: PredictWithTimestampsInput):
    try:
        comments = [item.text for item in data.comments]
        timestamps = [item.timestamp for item in data.comments]
        
        preprocessed_comments = [preprocess_comment(c) for c in comments]
        transformed_comments = vectorizer.transform(preprocessed_comments)
        
        predictions = model.predict(transformed_comments).tolist()
        predictions = [str(pred) for pred in predictions]
        
        response = [
            {"comment": c, "sentiment": s, "timestamp": t} 
            for c, s, t in zip(comments, predictions, timestamps)
        ]
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post('/predict')
def predict(data: PredictInput):
    try:
        preprocessed_comments = [preprocess_comment(c) for c in data.comments]
        transformed_comments = vectorizer.transform(preprocessed_comments)
        
        predictions = model.predict(transformed_comments).tolist()
        predictions = [str(pred) for pred in predictions]
        
        response = [{"comment": c, "sentiment": s} for c, s in zip(data.comments, predictions)]
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post('/generate_chart')
def generate_chart(data: ChartInput):
    try:
        counts = data.sentiment_counts
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [int(counts.get('1', 0)), int(counts.get('0', 0)), int(counts.get('-1', 0))]
        
        if sum(sizes) == 0:
            raise HTTPException(status_code=400, detail="Sentiment counts sum to zero")
            
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']
        
        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, textprops={'color': 'w'})
        plt.axis('equal')
        
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()
        
        return StreamingResponse(img_io, media_type='image/png')
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chart generation failed: {str(e)}")

@app.post('/generate_wordcloud')
def generate_wordcloud(data: PredictInput):
    try:
        preprocessed_comments = [preprocess_comment(c) for c in data.comments]
        text = ' '.join(preprocessed_comments)
        
        wordcloud = WordCloud(
            width=800, height=400, background_color='black', 
            colormap='Blues', stopwords=set(stopwords.words('english')), 
            collocations=False
        ).generate(text)
        
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)
        
        return StreamingResponse(img_io, media_type='image/png')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Word cloud generation failed: {str(e)}")

@app.post('/generate_trend_graph')
def generate_trend_graph(data: TrendInput):
    try:
        df = pd.DataFrame(data.sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['sentiment'] = df['sentiment'].astype(int)
        
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100
        
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0
        
        monthly_percentages = monthly_percentages[[-1, 0, 1]]
        
        plt.figure(figsize=(12, 6))
        colors = {-1: 'red', 0: 'gray', 1: 'green'}
        
        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index, monthly_percentages[sentiment_value], 
                marker='o', linestyle='-', label=sentiment_labels[sentiment_value], 
                color=colors[sentiment_value]
            )
            
        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
        plt.legend()
        plt.tight_layout()
        
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()
        
        return StreamingResponse(img_io, media_type='image/png')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trend graph generation failed: {str(e)}")
