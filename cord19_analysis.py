
# cord19_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
from collections import Counter
import wordcloud
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

class CORD19Analyzer:
    def __init__(self, file_path='metadata.csv'):
        self.file_path = file_path
        self.df = None
        self.df_clean = None
        
    def load_data(self):
        """Load the metadata CSV file"""
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Dataset loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            return True
        except FileNotFoundError:
            print(f"File {self.file_path} not found. Please check the file path.")
            return False
    
    def basic_exploration(self):
        """Perform basic data exploration"""
        if self.df is None:
            print("Please load data first")
            return
            
        print("=== BASIC DATA EXPLORATION ===")
        print(f"Dataset shape: {self.df.shape}")
        print("\nFirst 5 rows:")
        print(self.df.head())
        
        print("\nData types:")
        print(self.df.dtypes)
        
        print("\nMissing values per column:")
        missing_data = self.df.isnull().sum()
        print(missing_data[missing_data > 0])
        
        print("\nBasic statistics for numerical columns:")
        print(self.df.describe())
    
    def clean_data(self):
        """Clean and prepare the data for analysis"""
        if self.df is None:
            print("Please load data first")
            return
            
        # Create a copy for cleaning
        self.df_clean = self.df.copy()
        
        # Handle publication dates
        self.df_clean['publish_time'] = pd.to_datetime(self.df_clean['publish_time'], errors='coerce')
        self.df_clean['year'] = self.df_clean['publish_time'].dt.year
        
        # Fill missing years with 2020 (most common year for COVID papers)
        self.df_clean['year'] = self.df_clean['year'].fillna(2020)
        
        # Create abstract word count
        self.df_clean['abstract_word_count'] = self.df_clean['abstract'].apply(
            lambda x: len(str(x).split()) if pd.notnull(x) else 0
        )
        
        # Clean journal names
        self.df_clean['journal_clean'] = self.df_clean['journal'].str.lower().str.strip()
        
        print("Data cleaning completed!")
        print(f"Cleaned dataset shape: {self.df_clean.shape}")
        
    def analyze_publications_over_time(self):
        """Analyze publication trends over time"""
        if self.df_clean is None:
            print("Please clean data first")
            return
            
        yearly_counts = self.df_clean['year'].value_counts().sort_index()
        
        plt.figure(figsize=(12, 6))
        yearly_counts.plot(kind='bar', color='skyblue')
        plt.title('Number of COVID-19 Publications by Year')
        plt.xlabel('Year')
        plt.ylabel('Number of Publications')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return yearly_counts
    
    def analyze_top_journals(self, top_n=10):
        """Analyze top journals publishing COVID-19 research"""
        if self.df_clean is None:
            print("Please clean data first")
            return
            
        top_journals = self.df_clean['journal_clean'].value_counts().head(top_n)
        
        plt.figure(figsize=(12, 8))
        top_journals.plot(kind='barh', color='lightgreen')
        plt.title(f'Top {top_n} Journals Publishing COVID-19 Research')
        plt.xlabel('Number of Publications')
        plt.tight_layout()
        plt.show()
        
        return top_journals
    
    def create_title_wordcloud(self):
        """Create a word cloud from paper titles"""
        if self.df_clean is None:
            print("Please clean data first")
            return
            
        # Combine all titles
        titles = ' '.join(self.df_clean['title'].dropna().astype(str))
        
        # Remove common stopwords and clean text
        stopwords = set(['the', 'of', 'and', 'in', 'to', 'a', 'for', 'with', 'on', 'at', 'from',
                        'by', 'an', 'be', 'that', 'this', 'is', 'are', 'as', 'or', 'was', 'were',
                        'covid', '19', 'sars', 'cov', '2', 'coronavirus', 'pandemic'])
        
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            stopwords=stopwords,
            max_words=100
        ).generate(titles)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Most Frequent Words in Paper Titles')
        plt.tight_layout()
        plt.show()
    
    def analyze_sources(self):
        """Analyze distribution of papers by source"""
        if self.df_clean is None:
            print("Please clean data first")
            return
            
        source_counts = self.df_clean['source_x'].value_counts().head(10)
        
        plt.figure(figsize=(10, 8))
        plt.pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%')
        plt.title('Distribution of Papers by Source (Top 10)')
        plt.tight_layout()
        plt.show()
        
        return source_counts

# Example usage
if __name__ == "__main__":
    analyzer = CORD19Analyzer('metadata.csv')
    
    if analyzer.load_data():
        analyzer.basic_exploration()
        analyzer.clean_data()
        analyzer.analyze_publications_over_time()
        analyzer.analyze_top_journals()
        analyzer.create_title_wordcloud()
        analyzer.analyze_sources()