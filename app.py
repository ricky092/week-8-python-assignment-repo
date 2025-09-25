
# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from cord19_analysis import CORD19Analyzer
import sys
import os

# Set page configuration
st.set_page_config(
    page_title="CORD-19 Data Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("📊 CORD-19 COVID-19 Research Data Explorer")
    st.write("""
    This application provides an interactive exploration of the CORD-19 dataset, 
    which contains metadata about COVID-19 research papers.
    """)
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = CORD19Analyzer('metadata.csv')
        if not st.session_state.analyzer.load_data():
            st.error("Could not load the dataset. Please make sure 'metadata.csv' is in the correct directory.")
            return
        st.session_state.analyzer.clean_data()
    
    # Sidebar for controls
    st.sidebar.title("Navigation")
    section = st.sidebar.radio(
        "Go to:",
        ["Dataset Overview", "Publication Trends", "Journal Analysis", "Title Analysis", "Source Analysis", "Raw Data"]
    )
    
    if section == "Dataset Overview":
        show_dataset_overview()
    elif section == "Publication Trends":
        show_publication_trends()
    elif section == "Journal Analysis":
        show_journal_analysis()
    elif section == "Title Analysis":
        show_title_analysis()
    elif section == "Source Analysis":
        show_source_analysis()
    elif section == "Raw Data":
        show_raw_data()

def show_dataset_overview():
    st.header("📋 Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Information")
        st.write(f"**Total rows:** {st.session_state.analyzer.df.shape[0]:,}")
        st.write(f"**Total columns:** {st.session_state.analyzer.df.shape[1]}")
        
        # Show missing values
        missing_data = st.session_state.analyzer.df.isnull().sum()
        st.subheader("Missing Values")
        for col, missing_count in missing_data[missing_data > 0].head(10).items():
            st.write(f"- **{col}:** {missing_count:,} missing ({missing_count/st.session_state.analyzer.df.shape[0]*100:.1f}%)")
    
    with col2:
        st.subheader("First 10 Rows")
        st.dataframe(st.session_state.analyzer.df.head(10))
    
    st.subheader("Data Types")
    st.write(st.session_state.analyzer.df.dtypes)

def show_publication_trends():
    st.header("📈 Publication Trends Over Time")
    
    # Year range selector
    min_year = int(st.session_state.analyzer.df_clean['year'].min())
    max_year = int(st.session_state.analyzer.df_clean['year'].max())
    
    year_range = st.slider(
        "Select year range:",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )
    
    # Filter data by year range
    filtered_data = st.session_state.analyzer.df_clean[
        (st.session_state.analyzer.df_clean['year'] >= year_range[0]) & 
        (st.session_state.analyzer.df_clean['year'] <= year_range[1])
    ]
    
    yearly_counts = filtered_data['year'].value_counts().sort_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Publications by Year")
        fig, ax = plt.subplots(figsize=(10, 6))
        yearly_counts.plot(kind='bar', color='skyblue', ax=ax)
        ax.set_title('Number of COVID-19 Publications by Year')
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Publications')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Summary Statistics")
        st.write(f"**Total publications in range:** {len(filtered_data):,}")
        st.write(f"**Average publications per year:** {yearly_counts.mean():.0f}")
        st.write(f"**Year with most publications:** {yearly_counts.idxmax()} ({yearly_counts.max():,} papers)")
        
        st.subheader("Yearly Counts")
        for year, count in yearly_counts.items():
            st.write(f"- **{year}:** {count:,} publications")

def show_journal_analysis():
    st.header("🏥 Journal Analysis")
    
    top_n = st.slider("Number of top journals to show:", 5, 20, 10)
    
    top_journals = st.session_state.analyzer.df_clean['journal_clean'].value_counts().head(top_n)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Top {top_n} Journals")
        fig, ax = plt.subplots(figsize=(10, 8))
        top_journals.plot(kind='barh', color='lightgreen', ax=ax)
        ax.set_title(f'Top {top_n} Journals Publishing COVID-19 Research')
        ax.set_xlabel('Number of Publications')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Journal Statistics")
        st.write(f"**Total unique journals:** {st.session_state.analyzer.df_clean['journal_clean'].nunique():,}")
        st.write(f"**Journal with most publications:** {top_journals.index[0]} ({top_journals.iloc[0]:,} papers)")
        
        st.subheader("Top Journals List")
        for i, (journal, count) in enumerate(top_journals.items(), 1):
            st.write(f"{i}. **{journal.title()}:** {count:,} papers")

def show_title_analysis():
    st.header("📝 Title Analysis")
    
    st.subheader("Word Cloud of Paper Titles")
    
    # Generate word cloud
    titles = ' '.join(st.session_state.analyzer.df_clean['title'].dropna().astype(str))
    
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        max_words=100
    ).generate(titles)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Most Frequent Words in Paper Titles')
    st.pyplot(fig)
    
    # Simple word frequency analysis
    st.subheader("Top Words in Titles")
    words = ' '.join(st.session_state.analyzer.df_clean['title'].dropna().astype(str)).lower().split()
    word_freq = pd.Series(words).value_counts().head(20)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Most common words:")
        for word, freq in word_freq.head(10).items():
            st.write(f"- **{word}:** {freq:,}")
    
    with col2:
        st.write("Word frequency continued:")
        for word, freq in word_freq.tail(10).items():
            st.write(f"- **{word}:** {freq:,}")

def show_source_analysis():
    st.header("🌐 Source Analysis")
    
    source_counts = st.session_state.analyzer.df_clean['source_x'].value_counts().head(10)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribution by Source")
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%')
        plt.title('Distribution of Papers by Source (Top 10)')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Source Statistics")
        st.write(f"**Total unique sources:** {st.session_state.analyzer.df_clean['source_x'].nunique():,}")
        st.write(f"**Largest source:** {source_counts.index[0]} ({source_counts.iloc[0]:,} papers)")
        
        st.subheader("Top Sources")
        for i, (source, count) in enumerate(source_counts.items(), 1):
            st.write(f"{i}. **{source}:** {count:,} papers")

def show_raw_data():
    st.header("📄 Raw Data Explorer")
    
    st.subheader("Sample of the Dataset")
    
    rows_to_show = st.slider("Number of rows to show:", 10, 100, 20)
    
    st.dataframe(st.session_state.analyzer.df_clean.head(rows_to_show))
    
    # Column selection
    selected_columns = st.multiselect(
        "Select columns to display:",
        options=st.session_state.analyzer.df_clean.columns.tolist(),
        default=['title', 'journal', 'year', 'abstract_word_count']
    )
    
    if selected_columns:
        st.dataframe(st.session_state.analyzer.df_clean[selected_columns].head(rows_to_show))
    
    # Data summary
    st.subheader("Data Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Papers", f"{st.session_state.analyzer.df_clean.shape[0]:,}")
    with col2:
        st.metric("Columns", st.session_state.analyzer.df_clean.shape[1])
    with col3:
        st.metric("Years Covered", f"{int(st.session_state.analyzer.df_clean['year'].min())} - {int(st.session_state.analyzer.df_clean['year'].max())}")

if __name__ == "__main__":
    main()