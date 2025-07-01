import streamlit as st
import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import os



import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

def load_and_plot_network(edges_file, nodes_file, 
                         similarity_threshold=0.0, 
                         figsize=(15, 10),
                         layout_type='spring',
                         show_labels=True,
                         label_threshold=2):
    """
    Load network from CSV files and create visualizations
    
    Parameters:
    - edges_file: path to edges CSV file
    - nodes_file: path to nodes CSV file  
    - similarity_threshold: minimum similarity to show edges
    - figsize: figure size tuple
    - layout_type: 'spring', 'circular', 'kamada_kawai', 'spectral'
    - show_labels: whether to show node labels
    - label_threshold: minimum degree to show labels
    """
    
    print("Loading network data...")
    
    # Load data
    try:
        edges_df = pd.read_csv(edges_file)
        nodes_df = pd.read_csv(nodes_file)
        print(f"Loaded {len(edges_df)} edges and {len(nodes_df)} nodes")
    except Exception as e:
        print(f"Error loading files: {e}")
        return None
    
    # Filter edges by similarity threshold
    filtered_edges = edges_df[edges_df['similarity'] >= similarity_threshold]
    print(f"Using {len(filtered_edges)} edges with similarity >= {similarity_threshold}")
    
    # Create network
    G = nx.Graph()
    
    # Add nodes with attributes
    for _, row in nodes_df.iterrows():
        G.add_node(row['username'], 
                  degree=row['degree'],
                  centrality=row['degree_centrality'])
    
    # Add edges with weights
    for _, row in filtered_edges.iterrows():
        if row['user1'] in G.nodes() and row['user2'] in G.nodes():
            G.add_edge(row['user1'], row['user2'], 
                      weight=row['similarity'])
    
    # Remove isolated nodes
    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)
    print(f"Network after filtering: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Removed {len(isolated)} isolated nodes")
    
    if G.number_of_nodes() == 0:
        print("No nodes remaining after filtering. Try lowering the similarity_threshold.")
        return None
    
    # Create the plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: Main network visualization
    plot_main_network(G, ax1, layout_type, show_labels, label_threshold)
    
    # Plot 2: Degree distribution
    plot_degree_distribution(G, nodes_df, ax2)
    
    # Plot 3: Similarity distribution
    plot_similarity_distribution(filtered_edges, ax3)
    
    # Plot 4: Top users bar chart
    plot_top_users(nodes_df, ax4)
    
    plt.tight_layout()
    plt.savefig('network_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print network statistics
    print_network_stats(G, nodes_df, filtered_edges)
    
    return G

def plot_main_network(G, ax, layout_type='spring', show_labels=True, label_threshold=2):
    """Plot the main network visualization"""
    
    # Choose layout
    if layout_type == 'spring':
        pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
    elif layout_type == 'circular':
        pos = nx.circular_layout(G)
    elif layout_type == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout_type == 'spectral':
        pos = nx.spectral_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # Node properties
    degrees = dict(G.degree())
    node_sizes = [degrees[node] * 100 + 100 for node in G.nodes()]
    
    # Color nodes by degree
    node_colors = [degrees[node] for node in G.nodes()]
    
    # Edge properties
    edges = G.edges()
    edge_weights = [G[u][v]['weight'] for u, v in edges]
    edge_widths = [w * 3 for w in edge_weights]
    
    # Draw network
    nodes = nx.draw_networkx_nodes(G, pos, 
                                  node_size=node_sizes,
                                  node_color=node_colors,
                                  cmap=plt.cm.viridis,
                                  alpha=0.8,
                                  ax=ax)
    
    nx.draw_networkx_edges(G, pos,
                          width=edge_widths,
                          alpha=0.6,
                          edge_color='gray',
                          ax=ax)
    
    # Add labels for highly connected nodes
    if show_labels:
        high_degree_nodes = {node: node for node, degree in degrees.items() 
                           if degree >= label_threshold}
        nx.draw_networkx_labels(G, pos, high_degree_nodes, 
                               font_size=8, font_weight='bold', ax=ax)
    
    # Add colorbar
    plt.colorbar(nodes, ax=ax, label='Node Degree')
    
    ax.set_title(f'User Similarity Network\nNodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}', 
                fontsize=14, fontweight='bold')
    ax.axis('off')

def plot_degree_distribution(G, nodes_df, ax):
    """Plot degree distribution"""
    degrees = [d for n, d in G.degree()]
    
    ax.hist(degrees, bins=max(10, len(set(degrees))), alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_xlabel('Degree (Number of Connections)')
    ax.set_ylabel('Number of Users')
    ax.set_title('Degree Distribution')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_degree = np.mean(degrees)
    ax.axvline(mean_degree, color='red', linestyle='--', 
               label=f'Mean: {mean_degree:.1f}')
    ax.legend()

def plot_similarity_distribution(edges_df, ax):
    """Plot similarity score distribution"""
    similarities = edges_df['similarity'].values
    
    ax.hist(similarities, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    ax.set_xlabel('Similarity Score')
    ax.set_ylabel('Number of Connections')
    ax.set_title('Similarity Score Distribution')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_sim = np.mean(similarities)
    ax.axvline(mean_sim, color='blue', linestyle='--', 
               label=f'Mean: {mean_sim:.3f}')
    ax.legend()

def plot_top_users(nodes_df, ax):
    """Plot top users by degree"""
    top_users = nodes_df.nlargest(15, 'degree')
    
    bars = ax.barh(range(len(top_users)), top_users['degree'].values, 
                   color='lightgreen', alpha=0.8)
    ax.set_yticks(range(len(top_users)))
    ax.set_yticklabels([name[:20] + '...' if len(name) > 20 else name 
                       for name in top_users['username'].values])
    ax.set_xlabel('Number of Connections')
    ax.set_title('Top 15 Most Connected Users')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{int(width)}', ha='left', va='center', fontsize=8)

def print_network_stats(G, nodes_df, edges_df):
    """Print detailed network statistics"""
    
    print(f"\n{'='*50}")
    print("DETAILED NETWORK ANALYSIS")
    print(f"{'='*50}")
    
    # Basic stats
    print(f"Network Size:")
    print(f"  - Users (nodes): {G.number_of_nodes():,}")
    print(f"  - Connections (edges): {G.number_of_edges():,}")
    print(f"  - Network density: {nx.density(G):.4f}")
    
    # Degree statistics
    degrees = [d for n, d in G.degree()]
    print(f"\nConnection Statistics:")
    print(f"  - Average connections per user: {np.mean(degrees):.2f}")
    print(f"  - Max connections: {max(degrees)}")
    print(f"  - Min connections: {min(degrees)}")
    print(f"  - Median connections: {np.median(degrees):.1f}")
    
    # Similarity statistics
    similarities = edges_df['similarity'].values
    print(f"\nSimilarity Statistics:")
    print(f"  - Average similarity: {np.mean(similarities):.4f}")
    print(f"  - Max similarity: {max(similarities):.4f}")
    print(f"  - Min similarity: {min(similarities):.4f}")
    print(f"  - Median similarity: {np.median(similarities):.4f}")
    
    # Network structure
    print(f"\nNetwork Structure:")
    print(f"  - Connected components: {nx.number_connected_components(G)}")
    
    if nx.is_connected(G):
        print(f"  - Average shortest path: {nx.average_shortest_path_length(G):.2f}")
        print(f"  - Diameter: {nx.diameter(G)}")
    
    # Top users
    top_users = nodes_df.nlargest(5, 'degree')
    print(f"\nTop 5 Most Connected Users:")
    for _, user in top_users.iterrows():
        print(f"  - {user['username']}: {user['degree']} connections")

def create_community_visualization(edges_file, nodes_file, similarity_threshold=0.2):
    """Create a community-based visualization"""
    
    # Load and create network
    edges_df = pd.read_csv(edges_file)
    nodes_df = pd.read_csv(nodes_file)
    
    filtered_edges = edges_df[edges_df['similarity'] >= similarity_threshold]
    
    G = nx.Graph()
    for _, row in nodes_df.iterrows():
        G.add_node(row['username'])
    
    for _, row in filtered_edges.iterrows():
        if row['user1'] in G.nodes() and row['user2'] in G.nodes():
            G.add_edge(row['user1'], row['user2'], weight=row['similarity'])
    
    # Remove isolated nodes
    G.remove_nodes_from(list(nx.isolates(G)))
    
    if G.number_of_nodes() == 0:
        print("No nodes for community detection")
        return
    
    # Detect communities
    try:
        communities = nx.community.louvain_communities(G)
        print(f"Found {len(communities)} communities")
    except:
        print("Community detection failed")
        return
    
    # Plot communities
    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
    
    # Color nodes by community
    colors = plt.cm.Set3(np.linspace(0, 1, len(communities)))
    
    for i, community in enumerate(communities):
        if len(community) >= 2:  # Only show communities with 2+ members
            nx.draw_networkx_nodes(G, pos, 
                                 nodelist=list(community),
                                 node_color=[colors[i]],
                                 node_size=200,
                                 alpha=0.8,
                                 label=f'Community {i+1} ({len(community)} users)')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='gray', width=0.5)
    
    # Add labels for large communities
    large_communities = [c for c in communities if len(c) >= 3]
    if len(large_communities) > 0:
        labels = {}
        for community in large_communities[:5]:  # Show labels for top 5 communities
            for node in list(community)[:3]:  # Show first 3 nodes of each
                labels[node] = node
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    plt.title('User Communities Based on Content Similarity')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('community_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print community info
    print(f"\nCommunity Analysis:")
    for i, community in enumerate(communities):
        if len(community) >= 2:
            print(f"Community {i+1}: {len(community)} users")
            for user in list(community)[:5]:
                print(f"  - {user}")
            if len(community) > 5:
                print(f"  ... and {len(community) - 5} more")

# Usage examples:
print("Usage:")
print("# Basic network plot:")
print("G = load_and_plot_network('cameroun_tweets_network_edges.csv',")
print("                          'cameroun_tweets_network_nodes.csv',")
print("                          similarity_threshold=0.2)")
print()
print("# Community visualization:")
print("create_community_visualization('cameroun_tweets_network_edges.csv',")
print("                              'cameroun_tweets_network_nodes.csv',")
print("                              similarity_threshold=0.2)")

# Define the snippet structure for easier handling
class Snippet:
    def __init__(self, text, start, duration):
        self.text = text
        self.start = start
        self.duration = duration

def contains_number(text):
    return re.search(r'\d', text) is not None

def extract_with_context(snippets):
    indices_to_keep = set()
    for i, snippet in enumerate(snippets):
        if contains_number(snippet.text):
            indices_to_keep.update([i - 1, i, i + 1])

    final_output = [snippets[i] for i in sorted(indices_to_keep) if 0 <= i < len(snippets)]
    return final_output

def extract_video_id(url):
    import urllib.parse as urlparse
    parsed = urlparse.urlparse(url)
    if parsed.hostname in ['youtu.be']:
        return parsed.path[1:]
    if parsed.hostname in ['www.youtube.com', 'youtube.com']:
        qs = urlparse.parse_qs(parsed.query)
        return qs.get('v', [None])[0]
    return None

def display_snippets_in_groups(fr_snippets, en_snippets, group_size=3):
    for i in range(0, len(fr_snippets), group_size):
        fr_group = fr_snippets[i:i+group_size]
        en_group = en_snippets[i:i+group_size]
        
        # Display group header or separator if you want
        st.markdown("---")
        
        for fr_snip, en_snip in zip(fr_group, en_group):
            st.markdown(f"**[{fr_snip.start:.2f}s] FR:** {fr_snip.text}")
            st.markdown(f"                        **EN:** {en_snip.text}")
            st.write("")


proxies = {
    "http": "http://85.206.93.105:8080",
    "https": "http://85.206.93.105:8080"
}

st.title("📊 Twitter Activity in Cameroon")

# Step 1: File Upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Step 2: Load DataFrame
    df = pd.read_csv(uploaded_file)

    # Preview
    st.write("### Raw Data Preview", df.head())

    # Step 3: Parse 'date' column
    try:
        df['parsed_date'] = pd.to_datetime(df['date'].str.replace('·', ''), format="%b %d, %Y %I:%M %p UTC")

        # Step 4: Aggregate by month
        df['month'] = df['parsed_date'].dt.to_period('M').dt.to_timestamp()
        monthly_counts = df['month'].value_counts().sort_index()

        # Step 5: Plot
        st.write("### Monthly Tweet Volume")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(monthly_counts.index, monthly_counts.values, marker='o', linestyle='-')
        ax.set_title("Monthly Tweet Activity")
        ax.set_xlabel("Month")
        ax.set_ylabel("Tweet Count")
        ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Date parsing failed: {e}")
else:
    st.info("👈 Please upload a `.csv` file with a 'date' column.")


st.title("🕸️ Twitter User Network from Uploaded Tweets")

uploaded_file = st.file_uploader("Upload a CSV file with tweet data", type="csv")

similarity_threshold = st.slider("Select Similarity Threshold", 0.0, 1.0, 0.2, 0.05)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    # Preview
    st.write("### Raw Data Preview", df.head())
    if 'username' not in df.columns or 'text' not in df.columns:
        st.error("CSV must contain 'username' and 'text' columns.")
    else:
        st.success(f"Loaded {len(df)} tweets from {df['username'].nunique()} users.")

        # Step 1: Group tweets per user
        user_texts = df.groupby('username')['text'].apply(lambda x: ' '.join(x)).reset_index()

        # Step 2: TF-IDF vectorization
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(user_texts['text'])

        # Step 3: Cosine similarity between users
        sim_matrix = cosine_similarity(tfidf_matrix)

        # Step 4: Create edges
        edges = []
        for i in range(len(user_texts)):
            for j in range(i + 1, len(user_texts)):
                sim = sim_matrix[i, j]
                if sim >= similarity_threshold:
                    edges.append({
                        'user1': user_texts['username'][i],
                        'user2': user_texts['username'][j],
                        'similarity': sim
                    })

        edges_df = pd.DataFrame(edges)
        nodes_df = pd.DataFrame({
            'username': user_texts['username'],
            'degree': 0,
            'degree_centrality': 0.0
        })

        if len(edges_df) == 0:
            st.warning("No edges found with current threshold. Try lowering it.")
        else:
            # Calculate degree from edges
            from collections import Counter
            all_users = edges_df['user1'].tolist() + edges_df['user2'].tolist()
            degree_count = Counter(all_users)
            nodes_df['degree'] = nodes_df['username'].map(degree_count)
            nodes_df['degree_centrality'] = nodes_df['degree'] / (len(nodes_df) - 1)

            # Save to temporary CSVs
            edges_df.to_csv("edges.csv", index=False)
            nodes_df.to_csv("nodes.csv", index=False)

            st.subheader("📈 Network Graph")
            G = load_and_plot_network("edges.csv", "nodes.csv", similarity_threshold=similarity_threshold)
            st.image("network_analysis.png", caption="User Similarity Network", use_column_width=True)

            st.subheader("👥 Community Detection")
            create_community_visualization("edges.csv", "nodes.csv", similarity_threshold=similarity_threshold)
            st.image("community_analysis.png", caption="User Communities", use_column_width=True)

st.title("YouTube Transcript Number Extractor")

video_url = st.text_input("Enter a YouTube video URL")

if video_url:
    video_id = extract_video_id(video_url)
    if not video_id:
        st.error("Invalid YouTube URL or unable to extract video ID.")
    else:
        st.write(f"Extracted Video ID: {video_id}")
        try:
            transcripts = YouTubeTranscriptApi.list_transcripts(video_id, proxies=proxies)

            try:
                # Try to get English transcript and translate to French
                en_transcript = transcripts.find_generated_transcript(['en']).fetch()
                fr_transcript = transcripts.find_generated_transcript(['en']).translate('fr').fetch()
            except Exception:
                # Fallback: French transcript and translate to English
                fr_transcript = transcripts.find_generated_transcript(['fr']).fetch()
                en_transcript = transcripts.find_generated_transcript(['fr']).translate('en').fetch()

            # Extract relevant snippets with context
            fr_filtered = extract_with_context(fr_transcript)
            en_filtered = extract_with_context(en_transcript)

            if not fr_filtered or not en_filtered:
                st.write("No snippets containing numbers found in transcripts.")
            else:
                st.subheader("Filtered Transcript Snippets with Numbers (French - English)")

                #for fr_snip, en_snip in zip(fr_filtered, en_filtered):
                #   st.markdown(f"**[{fr_snip.start:.2f}s] FR:** {fr_snip.text}")
                #    st.markdown(f"                        **EN:** {en_snip.text}")
                #    st.write("---")
                display_snippets_in_groups(fr_filtered, en_filtered, group_size=3)

        except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
            st.error(f"Transcript unavailable: {e}")
        except Exception as e:
            st.error(f"Failed to fetch or process transcripts: {e}")
