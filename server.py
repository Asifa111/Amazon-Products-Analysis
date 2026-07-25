from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import requests
import logging
from datetime import datetime, timedelta
import numpy as np
import traceback
import random
import os
import csv
import pandas as pd
import torch
from bert_classifier import BERTClassifier, ProductDataset
from transformers import BertTokenizer, pipeline, AutoTokenizer, AutoModel
from gtts import gTTS
import tempfile
import base64
from googletrans import Translator
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from threading import Timer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Configure CORS to allow requests from the client
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# Keepa API configuration
KEEPA_API_KEY = 'b9842chlngbv99hs4lv0obmnbbms3pqvq1jtcjno6fvel4tsubnp72a26abe9qht'
KEEPA_API_URL = 'https://api.keepa.com/product'

# File path for the CSV file
CSV_FILE_PATH = 'data.csv'

# Load the trained model and category mappings
model = None
category_mappings = None
tokenizer = None

def load_model():
    global model, category_mappings, tokenizer
    if os.path.exists('bert_product_classifier.pt') and os.path.exists('category_mappings.pt'):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # Load category mappings first to get the number of labels
        category_mappings = torch.load('category_mappings.pt')
        num_labels = len(category_mappings['idx_to_category'])
        # Initialize model with correct number of labels
        model = BERTClassifier(num_labels=num_labels)
        model.model.load_state_dict(torch.load('bert_product_classifier.pt'))
        logger.info("BERT model loaded successfully")
    else:
        logger.warning("BERT model files not found. Please train the model first.")

class TokenPool:
    def __init__(self, api_key):
        self.api_key = api_key
        self.tokens_used = 0
        self.total_tokens = 60  # Update this to reflect your current token count
        self._check_tokens()

    def _check_tokens(self):
        """Check current token status with Keepa API"""
        try:
            params = {
                'key': self.api_key,
                'domain': 1,
                'asin': 'B07ZPKBL9V',  # Example ASIN
                'stats': 1
            }
            
            response = requests.get(KEEPA_API_URL, params=params)
            if response.status_code == 200:
                data = response.json()
                if 'tokensLeft' in data:
                    self.tokens_used = self.total_tokens - data['tokensLeft']
                    logger.info(f"Tokens remaining: {data['tokensLeft']}")
                else:
                    logger.error("No token information in response")
                    self.tokens_used = self.total_tokens  # Assume no tokens available
            else:
                logger.error(f"Failed to check tokens: {response.status_code}")
                self.tokens_used = self.total_tokens  # Assume no tokens available
        except Exception as e:
            logger.error(f"Error checking tokens: {str(e)}")
            self.tokens_used = self.total_tokens  # Assume no tokens available

    def get_token(self):
        """Get a token if available"""
        if self.tokens_used < self.total_tokens:
            return self.api_key
        logger.error("No tokens available")
        return None

    def return_token(self, token):
        """Return a token to the pool (in this case, just log the usage)"""
        self.tokens_used += 1
        logger.info(f"Token used. Total used: {self.tokens_used}")
        if self.tokens_used >= self.total_tokens:
            logger.error("All tokens have been used")
            self._check_tokens()  # Recheck token status when all tokens are used

# Initialize the token pool
token_pool = TokenPool(KEEPA_API_KEY)

def fetch_keepa_data(asin):
    """Fetch and parse product data from Keepa API for 10 years (2015-2024)"""
    try:
        # Get a token from the pool
        token = token_pool.get_token()
        if not token:
            logger.error("No available Keepa API tokens")
            return {'error': 'No available Keepa API tokens'}

        # Make the API request with additional parameters for price data
        params = {
            'key': token,
            'domain': 1,
            'asin': asin,
            'stats': 1,
            'csv': 1,
            'rating': 1,
            'price': 1
        }
        
        logger.info(f"Making Keepa API request for ASIN: {asin}")
        logger.info(f"Request URL: {KEEPA_API_URL}")
        logger.info(f"Request params: {params}")
        
        try:
            response = requests.get(KEEPA_API_URL, params=params)
            logger.info(f"Keepa API response status: {response.status_code}")
            
            if response.status_code != 200:
                logger.error(f"Keepa API error: {response.status_code}")
                return {'error': f'Keepa API error: {response.status_code}'}
                
            data = response.json()
            logger.info(f"Keepa API response type: {type(data)}")
            
            # Handle both list and dictionary responses
            if isinstance(data, list):
                if not data:
                    logger.error("Empty list response from Keepa API")
                    return {'error': 'No product data found for this ASIN'}
                product = data[0]
            elif isinstance(data, dict):
                products = data.get('products', [])
                if not products:
                    logger.error("No products found in response")
                    return {'error': 'No product data found for this ASIN'}
                product = products[0]
            else:
                logger.error(f"Invalid response format from Keepa API: {type(data)}")
                return {'error': 'Invalid response format from Keepa API'}
                
            # Extract basic product info with multiple fallbacks
            title = None
            is_amazon = False
            
            # Try different possible locations for the title and seller info
            if isinstance(product, dict):
                # Log all available fields for debugging
                logger.info(f"Product fields: {list(product.keys())}")
                
                # Try different possible title fields
                title_fields = ['title', 'productTitle', 'name', 'Title', 'ProductTitle', 'product_title']
                for field in title_fields:
                    if field in product:
                        title = product[field]
                        logger.info(f"Found title in field '{field}': {title}")
                        break
                
                # Check for Amazon seller information
                seller_fields = ['isAmazon', 'is_amazon', 'soldByAmazon', 'sold_by_amazon', 'amazonSeller']
                for field in seller_fields:
                    if field in product:
                        is_amazon = bool(product[field])
                        logger.info(f"Found seller info in field '{field}': {is_amazon}")
                        break
                
                # If still no seller info, check stats
                if not is_amazon and 'stats' in product:
                    stats = product['stats']
                    if isinstance(stats, dict):
                        for field in seller_fields:
                            if field in stats:
                                is_amazon = bool(stats[field])
                                logger.info(f"Found seller info in stats['{field}']: {is_amazon}")
                                break
            
            # If still no title, use a default
            if not title:
                title = f"Product {asin}"
                logger.warning(f"Using default title for ASIN {asin}")
            
            logger.info(f"Final title value: {title}")
            logger.info(f"Final is_amazon value: {is_amazon}")
            
            # Extract rating information from stats
            stats = product.get('stats', {})
            if isinstance(stats, dict):
                logger.info(f"Stats fields: {list(stats.keys())}")
            
            current = stats.get('current', {})
            if isinstance(current, dict):
                logger.info(f"Current stats fields: {list(current.keys())}")
            
            # Try to get rating from different possible locations
            rating = 0
            
            # Check for rating in productDetails
            product_details = product.get('productDetails', {})
            if isinstance(product_details, dict) and 'rating' in product_details:
                rating = product_details['rating']
                logger.info(f"Rating from productDetails: {rating}")
            
            # Check for rating in current stats
            if rating == 0 and isinstance(current, dict) and 'rating' in current:
                rating = current['rating']
                logger.info(f"Rating from current stats: {rating}")
            
            # Check for rating in product
            if rating == 0 and isinstance(product, dict) and 'rating' in product:
                rating = product['rating']
                logger.info(f"Rating from product: {rating}")
            
            # Check for rating in stats
            if rating == 0 and isinstance(stats, dict) and 'rating' in stats:
                rating = stats['rating']
                logger.info(f"Rating from stats: {rating}")
            
            # Check for rating in other fields
            if rating == 0 and isinstance(product, dict):
                for key, value in product.items():
                    if 'rating' in key.lower() and isinstance(value, (int, float)):
                        rating = value
                        logger.info(f"Rating from field {key}: {rating}")
                        break
            
            # Validate the rating value
            if rating > 5 or rating < 0:
                # If the rating is outside the valid range, try to normalize it
                if rating > 100:
                    # Some APIs return ratings as percentages (0-100)
                    rating = rating / 20
                    logger.info(f"Normalized rating from percentage: {rating}")
                elif rating > 10:
                    # Some APIs return ratings on a 0-10 scale
                    rating = rating / 2
                    logger.info(f"Normalized rating from 0-10 scale: {rating}")
                else:
                    # Default to a reasonable value if we can't determine the scale
                    rating = 4.0
                    logger.info(f"Using default rating: {rating}")
                    
            # Ensure rating is within valid range
            rating = max(0, min(5, rating))
            
            logger.info(f"Final rating value: {rating}")
            
            # Initialize arrays for price and rating history (10 years * 12 months = 120 months)
            prices = [None] * 120  # 120 months for 2015-2024
            ratings = [None] * 120  # 120 months for 2015-2024
            
            # Try to get price from stats
            if isinstance(stats, dict):
                # Check for price in current stats
                if isinstance(current, dict) and 'NEW' in current:
                    current_price = current['NEW']
                    if current_price > 0:
                        # Use current price for all months if available
                        prices = [current_price / 100.0] * 120  # Convert cents to dollars
                        logger.info(f"Using current price for all months: ${prices[0]:.2f}")
                
                # Check for price history in stats
                if 'NEW' in stats and isinstance(stats['NEW'], list):
                    price_history = stats['NEW']
                    logger.info(f"Price history length: {len(price_history)}")
                    
                    # Process price history
                    for i, price in enumerate(price_history):
                        if i >= 120:  # Only process first 120 months
                            break
                        if price > 0:  # -1 indicates no data
                            prices[i] = price / 100.0  # Convert cents to dollars
                            logger.info(f"Found price for month {i+1}: ${prices[i]:.2f}")
            
            # Process CSV data for prices and ratings
            csv_data = product.get('csv', [])
            if isinstance(csv_data, list) and len(csv_data) > 0:
                logger.info(f"Processing {len(csv_data)} CSV entries")
                base_date = datetime(1970, 1, 1)
                
                # Process pairs of values (timestamp, value)
                for i in range(0, len(csv_data), 2):
                    if i + 1 >= len(csv_data):
                        break
                        
                    try:
                        timestamp = int(csv_data[i])
                        value = float(csv_data[i + 1])
                        
                        # Calculate date from timestamp
                        date = base_date + timedelta(hours=timestamp)
                        
                        # Process price data for 2015-2024
                        if 2015 <= date.year <= 2024 and value != -1:  # -1 indicates no data
                            # Calculate month index (0-119)
                            month_index = (date.year - 2015) * 12 + (date.month - 1)
                            if 0 <= month_index < 120:
                                prices[month_index] = value / 100.0  # Convert cents to dollars
                                logger.info(f"Found price for {date.strftime('%Y-%m')}: ${prices[month_index]:.2f}")
                        
                        # Process rating data for 2015-2024
                        if 2015 <= date.year <= 2024 and value != -1:
                            # Calculate month index (0-119)
                            month_index = (date.year - 2015) * 12 + (date.month - 1)
                            if 0 <= month_index < 120:
                                # Normalize rating value to 0-5 scale
                                if value > 100:
                                    rating_value = value / 20
                                elif value > 10:
                                    rating_value = value / 2
                                else:
                                    rating_value = value
                                    
                                # Ensure rating is within valid range
                                rating_value = max(0, min(5, rating_value))
                                ratings[month_index] = round(rating_value, 1)
                                logger.info(f"Found rating for {date.strftime('%Y-%m')}: {rating_value:.1f}")
                            
                    except (ValueError, TypeError, IndexError) as e:
                        logger.warning(f"Error processing CSV entry: {e}")
                        continue
            
            # If we don't have any prices, generate some sample prices
            if all(p is None for p in prices):
                # Generate 120 sample prices
                base_price = 29.99  # Default base price
                for i in range(120):
                    # Generate a price within ±10% of the base price
                    sample_price = base_price * (1 + random.uniform(-0.1, 0.1))
                    prices[i] = round(sample_price, 2)
                    logger.info(f"Generated sample price for month {i+1}: ${prices[i]:.2f}")
            
            # If we don't have any ratings, generate some sample ratings
            if all(r is None for r in ratings) and rating > 0:
                # Generate 120 sample ratings around the current rating
                for i in range(120):
                    # Generate a rating within ±0.5 of the current rating
                    sample_rating = max(0, min(5, rating + random.uniform(-0.5, 0.5)))
                    ratings[i] = round(sample_rating, 1)
                    logger.info(f"Generated sample rating for month {i+1}: {sample_rating:.1f}")
            
            # Create a result with the expected structure
            result = {
                'asin': asin,  # Include the ASIN in the response
                'title': title,
                'rating': rating,
                'is_amazon': is_amazon,
                'ratings': ratings,
                'prices': prices
            }
            
            # Log the result structure for debugging
            logger.info(f"Result structure: {result}")
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error in fetch_keepa_data: {str(e)}")
            return {'error': f'Error connecting to Keepa API: {str(e)}'}
            
    except Exception as e:
        logger.error(f"Error in fetch_keepa_data: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {'error': f'Error fetching data: {str(e)}'}
    finally:
        # Return the token to the pool
        if 'token' in locals():
            token_pool.return_token(token)

@app.route('/api/fetch-product', methods=['POST'])
def fetch_product():
    try:
        request_data = request.get_json()
        logger.info(f"Received request data: {request_data}")
        
        if not request_data or 'asin' not in request_data:
            logger.error("Missing ASIN in request data")
            return jsonify({'error': 'ASIN is required'}), 400
        
        asin = request_data['asin']
        logger.info(f"Processing ASIN: {asin}")
        
        # Check token availability
        if not check_token_available():
            logger.error("No Keepa API tokens available")
            return jsonify({'error': 'No Keepa API tokens available'}), 503
        
        try:
            # Fetch data from Keepa
            result = fetch_keepa_data(asin)
            logger.info(f"Keepa API result type: {type(result)}")
            
            if not result:
                logger.error("No data returned from fetch_keepa_data")
                return jsonify({'error': 'No data found for this ASIN'}), 404
            
            if isinstance(result, dict):
                if 'error' in result:
                    logger.error(f"Error in Keepa API response: {result['error']}")
                    return jsonify({'error': result['error']}), 400
                
                # Log the structure of the result
                logger.info(f"Result keys: {list(result.keys())}")
                logger.info(f"Result structure: {result}")
                
                return jsonify(result)
            else:
                logger.error(f"Unexpected result type: {type(result)}")
                return jsonify({'error': 'Invalid response format from Keepa API'}), 500
            
        except Exception as e:
            logger.error(f"Error in fetch_keepa_data: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({'error': f'Error fetching data: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error in fetch_product: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': 'Internal server error'}), 500

def check_token_available():
    """Check if a token is available from the pool"""
    token = token_pool.get_token()
    if token:
        token_pool.return_token(token)  # Return the token since we just checked
        return True
    return False

@app.route('/api/token-status', methods=['GET'])
def get_token_status():
    """Get current token status without consuming any tokens"""
    try:
        tokens_left = token_pool.total_tokens - token_pool.tokens_used
        return jsonify({
            'tokens_left': tokens_left,
            'tokens_used': token_pool.tokens_used,
            'total_tokens': token_pool.total_tokens
        })
    except Exception as e:
        logger.error(f"Error getting token status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/products', methods=['GET'])
def get_products():
    """Get list of available products"""
    logger.info("GET /api/products endpoint hit")
    try:
        # Read products from CSV file
        df = pd.read_csv('product.csv')
        
        # Convert DataFrame to list of dictionaries
        products = df.to_dict('records')
        
        # Format the response
        formatted_products = []
        for product in products:
            formatted_products.append({
                'asin': product['ASIN'],
                'title': product['Title'],
                'description': product['Description'],
                'category': product['Categories']
            })
            
        logger.info(f"Returning {len(formatted_products)} products")
        return jsonify(formatted_products)
    except Exception as e:
        logger.error(f"Error in get_products: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Add a test endpoint to verify the server is running
@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({'status': 'ok', 'message': 'Server is running'})

@app.route('/api/save-csv', methods=['POST'])
def save_csv():
    try:
        data = request.get_json()
        logger.info(f"Received request to save CSV data")
        
        if not data:
            logger.error("No data provided in request")
            return jsonify({'error': 'Data is required'}), 400
            
        # Check if ASIN already exists in the CSV file
        if os.path.exists(CSV_FILE_PATH):
            with open(CSV_FILE_PATH, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row['ASIN'] == data.get('asin'):
                        logger.info(f"ASIN {data.get('asin')} already exists in {CSV_FILE_PATH}")
                        return jsonify({'error': 'Product with this ASIN already exists in the data file'}), 400
            
        # Create CSV content
        headers = ['ASIN', 'Title', 'Date', 'Price ($)', 'Rating', 'Is Amazon']
        
        # Check if file exists to determine if we need to write headers
        file_exists = os.path.exists(CSV_FILE_PATH)
        
        # Write to CSV file in append mode
        with open(CSV_FILE_PATH, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write headers only if file is new
            if not file_exists:
                writer.writerow(headers)
            
            # Add data rows
            dates = []
            for year in range(2015, 2025):
                for month in range(1, 13):
                    dates.append(f"{year}-{month:02d}")
                    
            for i, date in enumerate(dates):
                writer.writerow([
                    data.get('asin', 'N/A'),
                    data.get('title', 'N/A'),
                    date,
                    f"${data.get('prices', [])[i]:.2f}" if data.get('prices', [])[i] is not None else 'N/A',
                    f"{data.get('ratings', [])[i]:.1f}" if data.get('ratings', [])[i] is not None else 'N/A',
                    'Yes' if data.get('is_amazon', False) else 'No'
                ])
                
        logger.info(f"Successfully appended data to {CSV_FILE_PATH}")
        return jsonify({'success': True, 'message': f'Data appended to {CSV_FILE_PATH}'})
        
    except Exception as e:
        logger.error(f"Error saving CSV: {str(e)}")
        return jsonify({'error': f'Error saving CSV: {str(e)}'}), 500

@app.route('/api/download-csv', methods=['GET'])
def download_csv():
    try:
        if not os.path.exists(CSV_FILE_PATH):
            return jsonify({'error': 'CSV file not found'}), 404
            
        return send_file(
            CSV_FILE_PATH,
            mimetype='text/csv',
            as_attachment=True,
            download_name='data.csv'
        )
    except Exception as e:
        logger.error(f"Error downloading CSV: {str(e)}")
        return jsonify({'error': f'Error downloading CSV: {str(e)}'}), 500

@app.route('/api/unique-products', methods=['GET'])
def get_unique_products():
    try:
        if not os.path.exists(CSV_FILE_PATH):
            return jsonify({'error': 'No data file found'}), 404
            
        unique_products = {}
        
        with open(CSV_FILE_PATH, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                asin = row['ASIN']
                if asin not in unique_products:
                    # Get the first price and rating for this ASIN
                    price = row['Price ($)'].replace('$', '') if row['Price ($)'] != 'N/A' else None
                    rating = row['Rating'] if row['Rating'] != 'N/A' else None
                    
                    unique_products[asin] = {
                        'asin': asin,
                        'title': row['Title'],
                        'price': float(price) if price else None,
                        'rating': float(rating) if rating else None
                    }
        
        # Convert dictionary to list
        products_list = list(unique_products.values())
        logger.info(f"Found {len(products_list)} unique products")
        return jsonify(products_list)
        
    except Exception as e:
        logger.error(f"Error reading unique products: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Error reading unique products: {str(e)}'}), 500

@app.route('/api/product-details', methods=['POST'])
def get_product_details():
    try:
        data = request.get_json()
        logger.info(f"Received request for product details: {data}")
        
        if not data or 'asin' not in data:
            logger.error("No ASIN provided in request")
            return jsonify({'error': 'ASIN is required'}), 400
            
        asin = data['asin']
        logger.info(f"Fetching product details for ASIN: {asin}")
        
        # Get a token from the pool
        token = token_pool.get_token()
        if not token:
            logger.error("No available Keepa API tokens")
            return jsonify({'error': 'No available Keepa API tokens'}), 503
            
        # Make request to Keepa API
        params = {
            'key': token,
            'domain': 1,
            'asin': asin,
            'stats': 1
        }
        
        response = requests.get(KEEPA_API_URL, params=params)
        logger.info(f"Keepa API response status: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"Keepa API error: {response.text}")
            return jsonify({'error': f'Keepa API error: {response.status_code}'}), 500
            
        data = response.json()
        logger.info(f"Keepa API response: {data}")
        
        if 'error' in data:
            logger.error(f"Keepa API error: {data['error']}")
            return jsonify({'error': f'Keepa API error: {data["error"]}'}), 500
            
        if 'products' not in data or not data['products']:
            logger.error("No product data found")
            return jsonify({'error': 'No product data found'}), 404
            
        product = data['products'][0]
        
        # Extract product details
        title = product.get('title', 'N/A')
        description = product.get('description', 'N/A')
        category = product.get('categoryTree', [])
        
        # Log category data for debugging
        logger.info(f"Category data: {category}")
        
        # Ensure category is a list of strings
        if isinstance(category, list):
            # Convert category objects to strings if needed
            formatted_category = []
            for cat in category:
                if isinstance(cat, dict) and 'name' in cat:
                    formatted_category.append(cat['name'])
                elif isinstance(cat, str):
                    formatted_category.append(cat)
                else:
                    logger.warning(f"Unexpected category format: {cat}")
            category = formatted_category
        
        # Return product details
        return jsonify({
            'asin': asin,
            'title': title,
            'description': description,
            'category': category
        })
        
    except Exception as e:
        logger.error(f"Error fetching product details: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Error fetching product details: {str(e)}'}), 500

@app.route('/api/save-product', methods=['POST'])
def save_product():
    try:
        data = request.get_json()
        asin = data.get('asin')
        title = data.get('title')
        description = data.get('description')
        category = data.get('category')
        
        if not all([asin, title, description, category]):
            return jsonify({'error': 'Missing required fields'}), 400
            
        # Check if ASIN already exists
        df = pd.read_csv('product.csv')
        if asin in df['ASIN'].values:
            return jsonify({'error': 'Product with this ASIN already exists'}), 400
            
        # Append new product to CSV
        new_product = pd.DataFrame({
            'ASIN': [asin],
            'Title': [title],
            'Description': [description],
            'Categories': [category]
        })
        
        df = pd.concat([df, new_product], ignore_index=True)
        df.to_csv('product.csv', index=False)
        
        return jsonify({'message': 'Product saved successfully'})
    except Exception as e:
        logger.error(f"Error in save_product: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict_category', methods=['POST'])
def predict_category():
    try:
        data = request.get_json()
        description = data.get('description')
        
        if not description:
            return jsonify({'error': 'No description provided'}), 400
            
        if not model or not category_mappings:
            return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
            
        # Make prediction
        predicted_class, confidence = model.predict(description)
        
        # Convert predicted class index to category name using loaded mappings
        predicted_category = category_mappings['idx_to_category'][predicted_class]
        
        return jsonify({
            'category': predicted_category,
            'confidence': float(confidence)
        })
        
    except Exception as e:
        logger.error(f"Error in predict_category: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/semantic-search', methods=['POST'])
def semantic_search():
    try:
        data = request.get_json()
        query = data.get('query')
        
        if not query:
            return jsonify({'error': 'No search query provided'}), 400
            
        if not model:
            return jsonify({'error': 'Model not loaded'}), 500
            
        # Load product data
        df = pd.read_csv('product.csv')
        
        # Get embeddings for all products
        product_embeddings = []
        for _, row in df.iterrows():
            title = row['Title']
            description = row['Description']
            combined_text = f"{title} {description}"
            embedding = model.get_embedding(combined_text)
            product_embeddings.append(embedding)
            
        # Get embedding for search query
        query_embedding = model.get_embedding(query)
        
        # Calculate similarities
        similarities = []
        for i, product_embedding in enumerate(product_embeddings):
            similarity = torch.cosine_similarity(query_embedding, product_embedding, dim=0)
            similarities.append((i, float(similarity)))
            
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top 5 results
        results = []
        for idx, similarity in similarities[:5]:
            product = df.iloc[idx]
            results.append({
                'asin': product['ASIN'],
                'title': product['Title'],
                'description': product['Description'],
                'category': product['Categories'],
                'similarity': similarity
            })
            
        return jsonify({'results': results})
        
    except Exception as e:
        logger.error(f"Error in semantic_search: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    try:
        data = request.get_json()
        asin = data.get('asin')
        
        if not asin:
            return jsonify({'error': 'No ASIN provided'}), 400
            
        if not model:
            return jsonify({'error': 'Model not loaded'}), 500
            
        # Load product data
        df = pd.read_csv('product.csv')
        
        # Find the product with the given ASIN
        product_idx = df[df['ASIN'] == asin].index
        if len(product_idx) == 0:
            return jsonify({'error': 'Product not found'}), 404
            
        product_idx = product_idx[0]
        product = df.iloc[product_idx]
        
        # Get embeddings for all products
        product_embeddings = []
        for _, row in df.iterrows():
            title = row['Title']
            description = row['Description']
            combined_text = f"{title} {description}"
            embedding = model.get_embedding(combined_text)
            product_embeddings.append(embedding)
            
        # Get embedding for the selected product
        selected_embedding = product_embeddings[product_idx]
        
        # Calculate similarities
        similarities = []
        for i, product_embedding in enumerate(product_embeddings):
            if i != product_idx:  # Skip the selected product
                similarity = torch.cosine_similarity(selected_embedding, product_embedding, dim=0)
                similarities.append((i, float(similarity)))
                
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top 3 results
        results = []
        for idx, similarity in similarities[:3]:
            product = df.iloc[idx]
            results.append({
                'asin': product['ASIN'],
                'title': product['Title'],
                'description': product['Description'],
                'category': product['Categories'],
                'similarity': similarity
            })
            
        return jsonify({'recommendations': results})
        
    except Exception as e:
        logger.error(f"Error in get_recommendations: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Initialize summarization model and translator
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
translator = Translator()

@app.route('/api/summarize-and-speak', methods=['POST'])
def summarize_and_speak():
    try:
        data = request.get_json()
        description = data.get('description')
        
        if not description:
            return jsonify({'error': 'No description provided'}), 400
            
        # Generate summary
        logger.info("Generating summary...")
        summary = summarizer(description, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        logger.info(f"Generated summary: {summary}")
        
        # Generate English audio
        logger.info("Generating English audio...")
        tts_en = gTTS(text=summary, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp_en:
            tts_en.save(fp_en.name)
            with open(fp_en.name, 'rb') as audio_file:
                audio_base64_en = base64.b64encode(audio_file.read()).decode('utf-8')
                logger.info("English audio generated successfully")
        os.unlink(fp_en.name)
        
        # Translate to Urdu
        logger.info("Translating to Urdu...")
        urdu_summary = translator.translate(summary, dest='ur').text
        logger.info(f"Translated to Urdu: {urdu_summary}")
        
        # Generate Urdu audio
        logger.info("Generating Urdu audio...")
        tts_ur = gTTS(text=urdu_summary, lang='ur')
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp_ur:
            tts_ur.save(fp_ur.name)
            with open(fp_ur.name, 'rb') as audio_file:
                audio_base64_ur = base64.b64encode(audio_file.read()).decode('utf-8')
                logger.info("Urdu audio generated successfully")
        os.unlink(fp_ur.name)
        
        return jsonify({
            'summary': summary,
            'urdu_summary': urdu_summary,
            'audio_en': audio_base64_en,
            'audio_ur': audio_base64_ur
        })
        
    except Exception as e:
        logger.error(f"Error in summarize_and_speak: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

def show_popup_message(message, style=None):
    try:
        # Create popup window
        popup = tk.Toplevel()
        popup.title("Message")
        
        # Apply custom styling
        if style:
            popup.configure(bg=style.get('bg_color', '#ffffff'))
        
        # Set window size and position
        window_width = 300
        window_height = 100
        screen_width = popup.winfo_screenwidth()
        screen_height = popup.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        popup.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Add message label with custom styling
        label = tk.Label(
            popup,
            text=message,
            wraplength=250,
            bg=style.get('bg_color', '#ffffff') if style else '#ffffff',
            fg=style.get('text_color', '#000000') if style else '#000000'
        )
        label.pack(pady=20)
        
        # Add OK button with custom styling
        ok_button = tk.Button(
            popup,
            text="OK",
            command=popup.destroy,
            bg=style.get('button_color', '#4CAF50') if style else '#4CAF50',
            fg='white'
        )
        ok_button.pack(pady=10)
        
        # Make window modal and center it
        popup.transient()
        popup.grab_set()
        popup.update_idletasks()
        popup.lift()
        
    except Exception as e:
        # Fallback to console message if popup fails
        print(f"Popup Error: {str(e)}")
        print(f"Message: {message}")

def save_to_csv(product_data):
    try:
        # Check if product already exists
        if product_exists(product_data['id']):
            # Show popup with custom styling
            show_popup_message(
                "This product is already in the data file",
                style={
                    'bg_color': '#f0f0f0',
                    'text_color': '#333333',
                    'button_color': '#4CAF50'
                }
            )
            return False
        
        # Save to CSV
        save_product_data(product_data)
        
        # Show success message
        show_popup_message(
            "Product saved successfully",
            style={
                'bg_color': '#e8f5e9',
                'text_color': '#2e7d32',
                'button_color': '#4CAF50'
            }
        )
        return True
        
    except Exception as e:
        # Show error message
        show_popup_message(
            f"Error saving product: {str(e)}",
            style={
                'bg_color': '#ffebee',
                'text_color': '#c62828',
                'button_color': '#f44336'
            }
        )
        return False

def show_alternative_notification(message):
    # Create notification element
    notification = document.createElement('div')
    notification.className = 'notification'
    notification.textContent = message
    
    # Style notification
    notification.style.cssText = '''
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px;
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        z-index: 1000;
    '''
    
    # Add to document
    document.body.appendChild(notification)
    
    # Remove after 3 seconds using Python's Timer
    Timer(3.0, lambda: notification.remove()).start()

if __name__ == '__main__':
    load_model()  # Load the BERT model at startup
    logger.info("Starting server...")
    app.run(host='0.0.0.0', port=5004, debug=True)
