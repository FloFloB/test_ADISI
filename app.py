import streamlit as st
import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable

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
    "http": "http://123.140.146.1:5031",
    "https": "http://123.140.146.1:5031"
}


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
