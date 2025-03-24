# app.py
import streamlit as st
import nltk
from nltk.corpus import wordnet as wn
import random
import re

# Set page config
st.set_page_config(
    page_title="WordNet Poem Transformer",
    page_icon="📝",
    layout="wide"
)

# Download NLTK data (first time only)
@st.cache_resource
def download_nltk_data():
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

download_nltk_data()

# Helper function for POS tagging
def safe_pos_tag(tokens):
    """Try to tag tokens with POS, falling back to None tags if unavailable"""
    try:
        return nltk.pos_tag(tokens)
    except Exception:
        return [(token, None) for token in tokens]

def get_wordnet_pos(treebank_tag):
    """Convert Penn Treebank POS tags to WordNet POS tags"""
    if not treebank_tag:
        return None
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return None

def find_synonyms_or_related(word, pos=None, replacement_strategy='random'):
    """Find synonyms or related words using WordNet"""
    if not word or len(word) <= 2:  # Skip very short words
        return word
    
    # Try to get synsets for the word
    synsets = wn.synsets(word, pos=pos) if pos else wn.synsets(word)
    
    if not synsets:
        return word
    
    # Get all lemmas from all synsets
    all_lemmas = []
    
    if replacement_strategy == 'synonym':
        # Collect all lemmas from all synsets
        for synset in synsets:
            all_lemmas.extend([lemma.name() for lemma in synset.lemmas()])
    elif replacement_strategy == 'hypernym':
        # Get hypernyms (more general terms)
        for synset in synsets:
            hypernyms = synset.hypernyms()
            for hyp in hypernyms:
                all_lemmas.extend([lemma.name() for lemma in hyp.lemmas()])
    elif replacement_strategy == 'hyponym':
        # Get hyponyms (more specific terms)
        for synset in synsets:
            hyponyms = synset.hyponyms()
            for hypo in hyponyms:
                all_lemmas.extend([lemma.name() for lemma in hypo.lemmas()])
    else:  # random mix of all types
        for synset in synsets:
            # Add synonyms
            all_lemmas.extend([lemma.name() for lemma in synset.lemmas()])
            
            # Add some hypernyms and hyponyms if available
            hypernyms = synset.hypernyms()
            hyponyms = synset.hyponyms()
            
            if hypernyms and random.random() > 0.5:
                all_lemmas.extend([lemma.name() for lemma in random.choice(hypernyms).lemmas()])
            
            if hyponyms and random.random() > 0.5:
                all_lemmas.extend([lemma.name() for lemma in random.choice(hyponyms).lemmas()])
    
    # Remove duplicates and the original word
    unique_lemmas = [lemma.replace('_', ' ') for lemma in all_lemmas if lemma != word]
    
    if not unique_lemmas:
        return word
    
    # Return a random lemma from the list
    return random.choice(unique_lemmas)

def transform_poem(poem_text, replacement_rate=0.7, strategy='random', preserve_words=None):
    """Transform a poem by replacing words with related ones from WordNet while preserving formatting"""
    if preserve_words is None:
        preserve_words = []
    
    # Split the poem into lines to preserve formatting
    lines = poem_text.split('\n')
    transformed_lines = []
    
    for line in lines:
        if line.strip() == '':
            # Preserve empty lines
            transformed_lines.append('')
            continue
            
        # Tokenize each line
        tokens = nltk.word_tokenize(line)
        
        # POS tag using safe wrapper (no errors/warnings)
        tagged = safe_pos_tag(tokens)
        
        # Process each word in the line
        transformed_tokens = []
        for word, tag in tagged:
            # Check if the word should be preserved
            if word.lower() in [w.lower() for w in preserve_words]:
                transformed_tokens.append(word)
                continue
                
            # Skip punctuation and very short words
            if not re.match(r'[A-Za-z]', word) or len(word) <= 2:
                transformed_tokens.append(word)
                continue
                
            # Only replace some words based on replacement rate
            if random.random() > replacement_rate:
                transformed_tokens.append(word)
                continue
                
            # Get WordNet POS tag
            wordnet_pos = get_wordnet_pos(tag)
            
            # Find replacement
            replacement = find_synonyms_or_related(word.lower(), wordnet_pos, strategy)
            
            # Preserve capitalization
            if word[0].isupper():
                replacement = replacement.capitalize()
                
            transformed_tokens.append(replacement)
        
        # Reconstruct line with proper spacing for punctuation
        result = ""
        for i, token in enumerate(transformed_tokens):
            if i > 0 and token in ',.:;!?)]}' or (i > 0 and transformed_tokens[i-1] in '([{'):
                result += token
            else:
                result += " " + token
                
        transformed_lines.append(result.strip())
    
    # Join the lines back together with newlines
    return '\n'.join(transformed_lines)

# Streamlit UI
st.title("WordNet Poem Transformer")
st.markdown("""
Transform your poems or lyrics by replacing words with related ones using WordNet.
Enter your text, adjust the settings, and click Transform!
""")

col1, col2 = st.columns([3, 2])

with col1:
    # Sample poems dropdown
    sample_poems = {
        "Custom": "",
        "Roses are Red": "Roses are red,\nViolets are blue,\nSugar is sweet,\nAnd so are you.",
        "Twinkle Twinkle": "Twinkle, twinkle, little star,\nHow I wonder what you are!\nUp above the world so high,\nLike a diamond in the sky.",
        "The Road Not Taken (excerpt)": "Two roads diverged in a yellow wood,\nAnd sorry I could not travel both\nAnd be one traveler, long I stood\nAnd looked down one as far as I could\nTo where it bent in the undergrowth;"
    }
    
    selected_sample = st.selectbox("Choose a sample poem or create your own:", list(sample_poems.keys()))
    
    default_text = sample_poems[selected_sample] if selected_sample != "Custom" else ""
    input_text = st.text_area("Enter your poem or lyrics:", height=200, value=default_text)
    
    col_a, col_b = st.columns(2)
    with col_a:
        strategy = st.selectbox(
            "Replacement strategy:",
            [
                ("Random Mix", "random"),
                ("Synonyms Only", "synonym"),
                ("More General Words", "hypernym"),
                ("More Specific Words", "hyponym")
            ],
            format_func=lambda x: x[0]
        )[1]
    
    with col_b:
        replacement_rate = st.slider("Replacement rate:", 0.1, 1.0, 0.7, 0.1)
    
    preserve_words = st.text_input("Words to preserve (comma separated):")
    preserve_list = [word.strip() for word in preserve_words.split(',')] if preserve_words else []
    
    transform_button = st.button("Transform Poem")

with col2:
    st.subheader("Transformed Result")
    if transform_button and input_text:
        with st.spinner("Transforming poem..."):
            try:
                transformed = transform_poem(
                    input_text,
                    replacement_rate=replacement_rate,
                    strategy=strategy,
                    preserve_words=preserve_list
                )
                st.text_area("", value=transformed, height=350)
                st.download_button(
                    label="Download transformed poem",
                    data=transformed,
                    file_name="transformed_poem.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.info("Enter a poem and click Transform to see the result here.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and NLTK's WordNet")
