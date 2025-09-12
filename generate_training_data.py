import requests
import json
import re
import os
from tqdm import tqdm
import time
import random
import glob

model_name = "gemma3:4b"
base_url = "http://localhost:11434"

# Massively expanded prompt templates with different structures
prompt_templates = [
    # Direct statements
    "Generate a {tone} sentence about {topic} (max 10 words), then classify it.",
    "Write a {sentiment} comment about {topic} (max 10 words), then classify it.",
    "Create a sentence expressing {sentiment} feelings about {topic} (max 10 words), then classify it.",
    "Make a {tone} statement about {topic} (max 10 words), then classify it.",
    "Generate a {tone} opinion on {topic} (max 10 words), then classify it.",
    "Write something {sentiment} about {topic} (max 10 words), then classify it.",

    # Questions and responses
    "Ask a {tone} question about {topic} (max 10 words), then classify it.",
    "Write a {sentiment} response to someone discussing {topic} (max 10 words), then classify it.",
    "Create a {tone} inquiry regarding {topic} (max 10 words), then classify it.",
    "Generate a rhetorical question about {topic} with {sentiment} tone (max 10 words), then classify it.",

    # Conversational styles
    "Write what someone might {tone}ly say about {topic} (max 10 words), then classify it.",
    "Create a {sentiment} remark someone would make about {topic} (max 10 words), then classify it.",
    "Generate a {tone} response to '{topic}' discussion (max 10 words), then classify it.",
    "Write a {sentiment} comeback about {topic} (max 10 words), then classify it.",
    "Make a {tone} observation about {topic} (max 10 words), then classify it.",

    # Specific scenarios
    "Write what a {tone} person would tweet about {topic} (max 10 words), then classify it.",
    "Create a {sentiment} review comment about {topic} (max 10 words), then classify it.",
    "Generate a {tone} text message about {topic} (max 10 words), then classify it.",
    "Write a {sentiment} forum post about {topic} (max 10 words), then classify it.",
    "Make a {tone} social media comment on {topic} (max 10 words), then classify it.",

    # Emotional expressions
    "Express {sentiment} emotions about {topic} in a sentence (max 10 words), then classify it.",
    "Write how someone feeling {tone} would describe {topic} (max 10 words), then classify it.",
    "Create a {sentiment} reaction to {topic} (max 10 words), then classify it.",
    "Generate a {tone} emotional response about {topic} (max 10 words), then classify it.",

    # Action-oriented
    "Write a {tone} suggestion about {topic} (max 10 words), then classify it.",
    "Create a {sentiment} recommendation regarding {topic} (max 10 words), then classify it.",
    "Generate a {tone} complaint about {topic} (max 10 words), then classify it.",
    "Write a {sentiment} compliment about {topic} (max 10 words), then classify it.",
    "Make a {tone} warning about {topic} (max 10 words), then classify it.",

    # Comparative and descriptive
    "Compare {topic} to something else with {sentiment} tone (max 10 words), then classify it.",
    "Describe {topic} in a {tone} way (max 10 words), then classify it.",
    "Write a {sentiment} analogy involving {topic} (max 10 words), then classify it.",
    "Create a {tone} metaphor about {topic} (max 10 words), then classify it.",
]

# Expanded topics with more variety
topics = [
    # Daily life
    "work", "food", "relationships", "money", "health", "weather", "sleep", "commuting",
    "shopping", "cooking", "cleaning", "exercise", "hobbies", "pets", "neighbors",

    # Technology & media
    "technology", "social media", "gaming", "movies", "music", "TV shows", "internet",
    "smartphones", "computers", "streaming", "apps", "websites", "online shopping",

    # Social & cultural
    "politics", "education", "family", "friends", "dating", "marriage", "parenting",
    "community", "culture", "religion", "traditions", "holidays", "celebrations",

    # Entertainment & leisure
    "travel", "sports", "concerts", "festivals", "books", "art", "theater", "museums",
    "parks", "beaches", "hiking", "camping", "photography", "dancing",

    # Issues & challenges
    "crime", "traffic", "pollution", "noise", "crowds", "lines", "waiting", "delays",
    "taxes", "bills", "debt", "unemployment", "stress", "anxiety", "deadlines",

    # Services & institutions
    "healthcare", "insurance", "banks", "government", "schools", "restaurants",
    "stores", "public transport", "airports", "hotels", "customer service"
]

# More nuanced emotional descriptors
tones = [
    # Basic emotions
    "angry", "happy", "sad", "excited", "bored", "worried", "confused", "confident",

    # Complex emotions
    "sarcastic", "cheerful", "frustrated", "hopeful", "anxious", "relieved", "proud",
    "embarrassed", "grateful", "annoyed", "surprised", "disappointed", "curious",

    # Personality traits
    "cynical", "optimistic", "pessimistic", "enthusiastic", "cautious", "bold",
    "skeptical", "trusting", "patient", "impatient", "calm", "energetic",

    # Social tones
    "polite", "rude", "formal", "casual", "friendly", "hostile", "professional",
    "playful", "serious", "humorous", "dramatic", "understated"
]

sentiments = [
    # Basic sentiments
    "negative", "positive", "frustrated", "hopeful", "disappointed", "satisfied",

    # Emotional states
    "angry", "joyful", "melancholic", "euphoric", "bitter", "sweet", "contemptuous",
    "admiring", "resentful", "appreciative", "disgusted", "delighted",

    # Intensity levels
    "mildly positive", "strongly negative", "cautiously optimistic", "deeply frustrated",
    "extremely happy", "somewhat annoyed", "highly critical", "moderately pleased"
]

# Additional prompt variation techniques
prompt_starters = [
    "Write", "Create", "Generate", "Make", "Compose", "Craft", "Produce", "Form",
    "Build", "Construct", "Express", "State", "Declare", "Articulate", "Voice"
]

sentence_types = [
    "sentence", "comment", "statement", "opinion", "remark", "observation", "note",
    "thought", "reflection", "reaction", "response", "reply", "message", "post"
]


def get_random_prompt():
    # Randomly choose template construction method
    if random.random() < 0.7:  # 70% use predefined templates
        template = random.choice(prompt_templates)
        return template.format(
            topic=random.choice(topics),
            tone=random.choice(tones),
            sentiment=random.choice(sentiments)
        ) + "\n\nReturn EXACTLY this format:\n{\n    \"sentence\": \"your sentence here\",\n    \"harmful\": \"harmful\" or \"not_harmful\",\n    \"formality\": \"formal\" or \"informal\",\n    \"sentiment\": \"positive\" or \"negative\"\n}"
    else:  # 30% dynamically construct templates
        starter = random.choice(prompt_starters)
        sentence_type = random.choice(sentence_types)
        topic = random.choice(topics)

        # Randomly add complexity
        modifiers = []
        if random.random() < 0.5:
            modifiers.append(f"with a {random.choice(tones)} tone")
        if random.random() < 0.4:
            modifiers.append(
                f"expressing {random.choice(sentiments)} feelings")
        if random.random() < 0.3:
            contexts = ["in a casual conversation", "as a social media post", "in a review",
                        "as feedback", "as advice", "as a complaint", "as praise"]
            modifiers.append(random.choice(contexts))

        modifier_str = " " + " and ".join(modifiers) if modifiers else ""

        template = f"{starter} a {sentence_type} about {topic}{modifier_str} (max 10 words), then classify it."

        return template + "\n\nReturn EXACTLY this format:\n{\n    \"sentence\": \"your sentence here\",\n    \"harmful\": \"harmful\" or \"not_harmful\",\n    \"formality\": \"formal\" or \"informal\",\n    \"sentiment\": \"positive\" or \"negative\"\n}"

# Enhanced generation parameters for more diversity


def get_generation_params():
    # Vary parameters more dramatically
    return {
        "temperature": random.uniform(0.7, 1.2),  # Higher variation
        "top_p": random.uniform(0.8, 0.95),
        "top_k": random.randint(20, 60),
        "seed": random.randint(1, 2000000),
        # Add some additional randomness
        "repeat_penalty": random.uniform(1.0, 1.2)
    }


os.makedirs("training_data", exist_ok=True)

# Count existing files and lines to determine where to resume
existing_files = glob.glob("training_data/train_*.jsonl")
generated = 0
file_num = 1
examples_in_file = 0

if existing_files:
    # Count total lines across all files
    for file_path in existing_files:
        with open(file_path, 'r') as f:
            generated += sum(1 for line in f if line.strip())

    # Find the highest file number
    file_nums = [int(re.search(r'train_(\d+)\.jsonl', f).group(1))
                 for f in existing_files]
    file_num = max(file_nums)

    # Count lines in the current file
    current_file = f"training_data/train_{file_num:03d}.jsonl"
    if os.path.exists(current_file):
        with open(current_file, 'r') as f:
            examples_in_file = sum(1 for line in f if line.strip())

        # If current file is full, move to next
        if examples_in_file >= 5000:
            file_num += 1
            examples_in_file = 0

print(
    f"Resuming from {generated} examples, file {file_num}, {examples_in_file} examples in current file")

start_time = time.time()

pbar = tqdm(total=50000, initial=generated, desc="Generating")

attempt = 0
failed_attempts = 0
recent_sentences = set()  # Track recent sentences to avoid duplicates

while generated < 50000 and attempt < 100000:
    attempt += 1
    try:
        current_prompt = get_random_prompt()

        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model_name,
                "prompt": current_prompt,
                "stream": False,
                "options": get_generation_params()
            },
            timeout=30
        )

        text = response.json()['response']
        json_match = re.search(r'\{.*?\}', text, re.DOTALL)
        if not json_match:
            failed_attempts += 1
            continue

        data = json.loads(json_match.group())

        if not all(key in data for key in ["sentence", "harmful", "formality", "sentiment"]):
            failed_attempts += 1
            continue

        sentence = data["sentence"].strip()

        # Skip if sentence is too similar to recent ones
        if sentence.lower() in recent_sentences:
            continue

        if len(sentence.split()) <= 10:
            # Add to recent sentences set (keep last 1000)
            recent_sentences.add(sentence.lower())
            if len(recent_sentences) > 1000:
                recent_sentences.pop()

            with open(f"training_data/train_{file_num:03d}.jsonl", 'a') as f:
                f.write(json.dumps(data) + '\n')

            generated += 1
            examples_in_file += 1
            failed_attempts = 0  # Reset on success

            elapsed = time.time() - start_time
            examples_per_min = (generated / elapsed) * 60 if elapsed > 0 else 0
            eta_minutes = ((50000 - generated) /
                           examples_per_min) if examples_per_min > 0 else 0

            pbar.set_description(
                f"{generated} | {examples_per_min:.1f}/min | ETA: {eta_minutes:.0f}min | Fails: {failed_attempts}")
            pbar.update(1)

            if examples_in_file == 5000:
                print(f"File {file_num} complete ({generated} total)")
                file_num += 1
                examples_in_file = 0
                recent_sentences.clear()  # Clear duplicates check for new file

    except Exception as e:
        failed_attempts += 1
        continue

    # Add slight delay with some randomness
    time.sleep(random.uniform(0.03, 0.08))

pbar.close()
print(
    f"Done! Generated {generated} examples in {(time.time() - start_time)/60:.1f} minutes")
