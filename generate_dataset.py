"""Generate deterministic question-answer pairs from varied English templates."""

import csv
import random
from pathlib import Path

OUTPUT_FILE = Path(__file__).with_name("dataset.csv")
SEED = 42


def create_pairs():
    pairs = set()

    def add(questions, answers):
        for question in questions:
            for answer in answers:
                pairs.add((question, answer))

    activities = ["write code", "read books", "play music", "learn English",
                  "drink water", "eat food", "watch movies", "study math",
                  "play games", "make coffee", "draw pictures", "help people"]
    for action in activities:
        add(["What do you want to do?", "What would you like to do?",
             "What do you plan to do?"], [f"I want to {action}.", f"I would like to {action}."])
        add([f"Do you want to {action}?", f"Would you like to {action}?"],
            [f"Yes, I want to {action}.", f"Yes, I would like to {action}.",
             f"No, I do not want to {action}."])
        add(["What can you do?", "What are you able to do?"],
            [f"I can {action}.", f"I can try to {action}."])
        add(["What are you doing?", "What do you do in your free time?"],
            [f"I like to {action}.", f"I want to {action}."])

    things = ["music", "coffee", "tea", "water", "books", "coding",
              "math", "movies", "games", "English", "art", "science"]
    for thing in things:
        add([f"Do you like {thing}?", f"Do you enjoy {thing}?"],
            [f"Yes, I like {thing}.", f"No, I do not like {thing}.",
             f"I really like {thing}."])
        add(["What do you like?", "What do you enjoy?", "What is your favorite thing?"],
            [f"I like {thing}.", f"I enjoy {thing}.", f"My favorite thing is {thing}."])

    weather = ["sunny", "rainy", "cloudy", "windy", "cold", "warm", "hot"]
    for state in weather:
        add(["How is the weather?", "What is the weather like?",
             "How is the weather today?", "Tell me about the weather."],
            [f"It is {state}.", f"The weather is {state} today."])
        add([f"Is it {state} today?", f"Is the weather {state}?"],
            [f"Yes, it is {state}.", f"No, it is not {state}."])

    moods = ["happy", "sad", "tired", "ready", "busy", "excited", "calm"]
    for mood in moods:
        add(["How are you?", "How do you feel?", "How are you feeling today?"],
            [f"I am {mood}.", f"I feel {mood} today."])
        add([f"Are you {mood}?", f"Do you feel {mood}?"],
            [f"Yes, I am {mood}.", f"No, I am not {mood}."])

    places = ["home", "school", "the park", "the library", "the office", "the store"]
    for place in places:
        add(["Where are you?", "Where do you want to go?", "Where will you go?"],
            [f"I am at {place}.", f"I want to go to {place}."])
        add([f"Do you want to go to {place}?"],
            [f"Yes, I want to go to {place}.", f"No, I want to stay home."])

    foods = ["rice", "bread", "fruit", "pizza", "salad", "soup", "eggs"]
    for food in foods:
        add(["What do you want to eat?", "What would you like for lunch?",
             "What food do you like?"],
            [f"I want to eat {food}.", f"I would like {food}.", f"I like {food}."])
        add([f"Do you want {food}?", f"Do you like {food}?"],
            [f"Yes, I like {food}.", f"No, I do not want {food}."])

    times = ["today", "tomorrow", "tonight", "on Monday", "on Friday",
             "in the morning", "in the afternoon", "in the evening"]
    for when in times:
        for action in activities:
            add([f"What will you do {when}?", f"What do you want to do {when}?"],
                [f"I will {action} {when}.", f"I want to {action} {when}."])
            add([f"Will you {action} {when}?"],
                [f"Yes, I will {action} {when}.", f"No, I will not {action} {when}."])

    subjects = ["I", "We", "They", "You"]
    verbs = ["read", "watch", "study", "enjoy", "like", "need", "want"]
    objects = ["books", "movies", "music", "English", "math", "coffee", "water"]
    for subject in subjects:
        for verb in verbs:
            for obj in objects:
                add([f"What do {subject.lower()} {verb}?"],
                    [f"{subject} {verb} {obj}."])

    # Compositional conversations about actions, times, and places.
    for action in activities:
        for when in times:
            for place in places:
                add([f"What will you do {when} at {place}?",
                     f"What do you want to do at {place} {when}?"],
                    [f"I will {action} at {place} {when}.",
                     f"I want to {action} at {place} {when}."])
                add([f"Will you {action} at {place} {when}?"],
                    [f"Yes, I will {action} at {place} {when}.",
                     f"No, I will not {action} at {place} {when}."])

    add(["What is your name?", "Who are you?"],
        ["My name is Tiny AI.", "I am Tiny AI."])
    add(["Can you help me?", "Will you help me?"],
        ["Yes, I can help you.", "I can try to help you."])
    add(["What is your favorite color?"],
        ["My favorite color is blue.", "I like green."])
    add(["Thank you!", "Thanks!"], ["You are welcome.", "I am happy to help."])
    add(["Hello!", "Hi!", "Good morning!"],
        ["Hello!", "Hi!", "How are you?"])
    return sorted(pairs)


def main():
    pairs = create_pairs()
    rng = random.Random(SEED)
    rng.shuffle(pairs)
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["question", "answer"])
        writer.writerows(pairs)
    print(f"Saved {len(pairs):,} question-answer pairs to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
