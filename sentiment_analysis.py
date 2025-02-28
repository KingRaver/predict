import requests
import re
import pandas as pd
import numpy as np
from textblob import TextBlob
from datetime import datetime, timedelta
import time

class SentimentAnalyzer:
    def __init__(self, db_connection):
        self.conn = db_connection
        self.setup_database()
        
    def setup_database(self):
        """Create necessary tables for sentiment storage"""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                source TEXT NOT NULL,
                symbol TEXT NOT NULL,
                sentiment_score REAL NOT NULL,
                volume INTEGER NOT NULL,
                raw_data JSON
            )
        """)
        self.conn.commit()
        
    def analyze_twitter_sentiment(self, symbol="KAITO", count=100):
        """Analyze sentiment from Twitter (simplified mock version)"""
        try:
            # In a real implementation, you would:
            # 1. Use Twitter API to fetch recent tweets about the symbol
            # 2. Process and analyze those tweets
            
            # For this example, we'll create mock data
            mock_tweets = self._generate_mock_tweets(symbol, count)
            
            # Analyze sentiment for each tweet
            sentiments = []
            for tweet in mock_tweets:
                blob = TextBlob(tweet['text'])
                sentiment = blob.sentiment.polarity
                sentiments.append(sentiment)
                
            # Calculate overall metrics
            avg_sentiment = np.mean(sentiments)
            sentiment_volume = len(sentiments)
            
            # Store in database
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO sentiment_data (
                    timestamp, source, symbol, sentiment_score, volume, raw_data
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                'twitter',
                symbol,
                avg_sentiment,
                sentiment_volume,
                None
            ))
            self.conn.commit()
            
            return {
                'symbol': symbol,
                'source': 'twitter',
                'sentiment_score': avg_sentiment,
                'volume': sentiment_volume,
                'timestamp': datetime.now()
            }
        except Exception as e:
            print(f"Error analyzing Twitter sentiment: {str(e)}")
            return None
            
    def analyze_reddit_sentiment(self, symbol="KAITO", subreddits=['CryptoCurrency'], count=100):
        """Analyze sentiment from Reddit (simplified mock version)"""
        try:
            # Mock Reddit data
            mock_posts = self._generate_mock_reddit(symbol, count, subreddits)
            
            # Analyze sentiment
            sentiments = []
            for post in mock_posts:
                blob = TextBlob(post['title'] + " " + post['text'])
                sentiment = blob.sentiment.polarity
                sentiments.append(sentiment)
                
            # Calculate overall metrics
            avg_sentiment = np.mean(sentiments)
            sentiment_volume = len(sentiments)
            
            # Store in database
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO sentiment_data (
                    timestamp, source, symbol, sentiment_score, volume, raw_data
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                'reddit',
                symbol,
                avg_sentiment,
                sentiment_volume,
                None
            ))
            self.conn.commit()
            
            return {
                'symbol': symbol,
                'source': 'reddit',
                'sentiment_score': avg_sentiment,
                'volume': sentiment_volume,
                'timestamp': datetime.now()
            }
        except Exception as e:
            print(f"Error analyzing Reddit sentiment: {str(e)}")
            return None
    
    def get_combined_sentiment(self, symbol="KAITO", hours=24):
        """Get combined sentiment from all sources"""
        try:
            query = f"""
                SELECT source, AVG(sentiment_score) as avg_sentiment, 
                       SUM(volume) as total_volume
                FROM sentiment_data
                WHERE symbol = '{symbol}'
                AND timestamp >= datetime('now', '-{hours} hours')
                GROUP BY source
            """
            sentiment_df = pd.read_sql(query, self.conn)
            
            if sentiment_df.empty:
                return {
                    'symbol': symbol,
                    'combined_sentiment': 0,
                    'total_volume': 0,
                    'sources': []
                }
                
            # Calculate weighted sentiment
            total_volume = sentiment_df['total_volume'].sum()
            if total_volume > 0:
                weighted_sentiment = (sentiment_df['avg_sentiment'] * sentiment_df['total_volume']).sum() / total_volume
            else:
                weighted_sentiment = sentiment_df['avg_sentiment'].mean()
                
            # Create source breakdown
            sources = []
            for _, row in sentiment_df.iterrows():
                sources.append({
                    'source': row['source'],
                    'sentiment': row['avg_sentiment'],
                    'volume': row['total_volume']
                })
                
            return {
                'symbol': symbol,
                'combined_sentiment': weighted_sentiment,
                'total_volume': total_volume,
                'sources': sources
            }
        except Exception as e:
            print(f"Error getting combined sentiment: {str(e)}")
            return None
    
    def get_sentiment_trend(self, symbol="KAITO", days=7, interval_hours=4):
        """Get sentiment trend over time"""
        try:
            query = f"""
                SELECT 
                    strftime('%Y-%m-%d %H:00:00', timestamp) as time_bucket,
                    AVG(sentiment_score) as sentiment,
                    SUM(volume) as volume
                FROM sentiment_data
                WHERE symbol = '{symbol}'
                AND timestamp >= datetime('now', '-{days} days')
                GROUP BY time_bucket
                ORDER BY time_bucket ASC
            """
            trend_df = pd.read_sql(query, self.conn)
            
            if trend_df.empty:
                return []
                
            # Convert to list of points
            trend = []
            for _, row in trend_df.iterrows():
                trend.append({
                    'timestamp': row['time_bucket'],
                    'sentiment': row['sentiment'],
                    'volume': row['volume']
                })
                
            return trend
        except Exception as e:
            print(f"Error getting sentiment trend: {str(e)}")
            return []
    
    def _generate_mock_tweets(self, symbol, count):
        """Generate mock Twitter data for testing"""
        positive_templates = [
            f"{symbol} looking bullish today! #crypto",
            f"Just bought more {symbol}! To the moon! 🚀",
            f"{symbol} technical analysis shows strong support levels",
            f"Loving the {symbol} project roadmap. Great dev team!",
            f"{symbol} breaking out! This could be the start of something big"
        ]
        
        neutral_templates = [
            f"What's everyone's thoughts on {symbol} right now?",
            f"{symbol} trading sideways today",
            f"New to crypto, is {symbol} a good investment?",
            f"Just heard about {symbol}, researching it now",
            f"{symbol} volume seems average today"
        ]
        
        negative_templates = [
            f"{symbol} dropping hard. Should I sell?",
            f"Not convinced about the sustainability of {symbol}",
            f"Bears taking control of {symbol} price action",
            f"Dumped my {symbol} bags. Project isn't delivering",
            f"{symbol} support level broken, could see more downside"
        ]
        
        # Sentiment distribution (slightly bullish)
        sentiment_probabilities = [0.45, 0.30, 0.25]  # positive, neutral, negative
        
        tweets = []
        for _ in range(count):
            sentiment_type = np.random.choice(['positive', 'neutral', 'negative'], p=sentiment_probabilities)
            
            if sentiment_type == 'positive':
                template = np.random.choice(positive_templates)
            elif sentiment_type == 'neutral':
                template = np.random.choice(neutral_templates)
            else:
                template = np.random.choice(negative_templates)
                
            # Add some randomization
            tweet = {
                'text': template,
                'created_at': (datetime.now() - timedelta(hours=np.random.randint(1, 24))).isoformat(),
                'user': f"crypto_user_{np.random.randint(1000, 9999)}",
                'retweets': np.random.randint(0, 50),
                'likes': np.random.randint(0, 100)
            }
            
            tweets.append(tweet)
            
        return tweets
    
    def _generate_mock_reddit(self, symbol, count, subreddits):
        """Generate mock Reddit data for testing"""
        positive_templates = [
            f"{symbol} - My Technical Analysis for this week: BULLISH [DD]",
            f"Why {symbol} is undervalued right now",
            f"{symbol} just announced a major partnership! Implications inside",
            f"I've been using {symbol}'s product for a month now - Here's my review",
            f"{symbol} showing strong on-chain metrics lately"
        ]
        
        neutral_templates = [
            f"{symbol} - What am I missing?",
            f"Newbie here: Can someone explain {symbol} to me?",
            f"{symbol} price discussion thread",
            f"Comparing {symbol} with its competitors",
            f"Historical analysis of {symbol} price movements"
        ]
        
        negative_templates = [
            f"Warning: {symbol} showing bearish signals",
            f"Just uncovered potential red flags with {symbol}",
            f"Why did {symbol} drop so hard today?",
            f"{symbol} Technical Analysis: Prepare for more downside",
            f"Disappointed with {symbol}'s latest update"
        ]
        
        # Sentiment distribution (slightly bullish)
        sentiment_probabilities = [0.45, 0.30, 0.25]  # positive, neutral, negative
        
        posts = []
        for _ in range(count):
            sentiment_type = np.random.choice(['positive', 'neutral', 'negative'], p=sentiment_probabilities)
            
            if sentiment_type == 'positive':
                title_template = np.random.choice(positive_templates)
                sentiment_mod = 1
            elif sentiment_type == 'neutral':
                title_template = np.random.choice(neutral_templates)
                sentiment_mod = 0
            else:
                title_template = np.random.choice(negative_templates)
                sentiment_mod = -1
                
            # Generate mock post text based on sentiment
            text_length = np.random.randint(100, 500)
            
            if sentiment_mod > 0:
                text = f"I'm really optimistic about {symbol}. " * (text_length // 30)
            elif sentiment_mod == 0:
                text = f"Looking at {symbol} objectively. " * (text_length // 30)
            else:
                text = f"Concerned about {symbol}'s direction. " * (text_length // 30)
                
            # Add some randomization
            post = {
                'title': title_template,
                'text': text[:text_length],
                'created_at': (datetime.now() - timedelta(hours=np.random.randint(1, 72))).isoformat(),
                'subreddit': np.random.choice(subreddits),
                'user': f"reddit_user_{np.random.randint(1000, 9999)}",
                'upvotes': np.random.randint(1, 500),
                'comments': np.random.randint(0, 100)
            }
            
            posts.append(post)
            
        return posts
