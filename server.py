import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv')

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        if environ["REQUEST_METHOD"] == "GET":
            # Create the response body from the reviews and convert to a JSON byte string
            # response_body = json.dumps(reviews, indent=2).encode("utf-8")
            
            # Write your code here
            # sentiment=[self.analyze_sentiment(x['ReviewBody']) for x in reviews]
            query_string = environ.get('QUERY_STRING', '')
            params = parse_qs(query_string)
            location = params.get('location', [None])[0]
            start_date = params.get('start_date', [None])[0]
            end_date = params.get('end_date', [None])[0]
            filtered_reviews = reviews.copy()

            if location:
                filtered_reviews = filtered_reviews[filtered_reviews['Location'] == location]
            if start_date and not end_date:
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
                filtered_reviews['Timestamp'] = pd.to_datetime(filtered_reviews['Timestamp'])
                filtered_reviews = filtered_reviews[(filtered_reviews['Timestamp'] >= start_date)]
                filtered_reviews['Timestamp']=filtered_reviews['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            if not start_date and end_date:
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
                filtered_reviews['Timestamp'] = pd.to_datetime(filtered_reviews['Timestamp'])
                filtered_reviews = filtered_reviews[(filtered_reviews['Timestamp'] <= end_date)]
                filtered_reviews['Timestamp']=filtered_reviews['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            if start_date and end_date:
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
                filtered_reviews['Timestamp'] = pd.to_datetime(filtered_reviews['Timestamp'])
                filtered_reviews = filtered_reviews[(filtered_reviews['Timestamp'] >= start_date) & (filtered_reviews['Timestamp'] <= end_date)]
                filtered_reviews['Timestamp']=filtered_reviews['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            # Analyze sentiment and add to reviews
            filtered_reviews['sentiment'] = filtered_reviews['ReviewBody'].apply(self.analyze_sentiment) 

            # Sort by compound sentiment score in descending order
            filtered_reviews = filtered_reviews.sort_values(by=['sentiment'], key=lambda x: x.apply(lambda y: y['compound']), ascending=False)

            # Convert to list of dictionaries
            filtered_reviews = filtered_reviews.to_dict(orient='records')

            # Create the response body from the reviews and convert to a JSON byte string
            response_body = json.dumps(filtered_reviews, indent=2).encode("utf-8")
            # Set the appropriate response headers
            start_response("200 OK", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(response_body)))
             ])
            
            return [response_body]


        if environ["REQUEST_METHOD"] == "POST":
            # Write your code here
            try:
                # Read and parse POST data
                content_length = int(environ.get('CONTENT_LENGTH', 0))
                post_data = environ['wsgi.input'].read(content_length).decode('utf-8')
                params = parse_qs(post_data)
                
                review_body = params.get('ReviewBody', [None])[0]
                location = params.get('Location', [None])[0]

                if review_body is None or location is None:
                    raise ValueError("Missing ReviewBody or Location")

                # Generate ReviewId and Timestamp
                review_id = str(uuid.uuid4())
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # Create new review entry
                new_review = {
                    "ReviewId": review_id,
                    "ReviewBody": review_body,
                    "Location": location,
                    "Timestamp": timestamp
                }

                # Create response body
                response_body = json.dumps(new_review, indent=2).encode('utf-8')

                # Set the appropriate response headers
                start_response("201 OK", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])
                
                return [response_body]
            except Exception as e:
                error_response = json.dumps({"error": str(e)}).encode('utf-8')
                start_response("400 Bad Request", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(error_response)))
                ])
                return [error_response]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8080)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()