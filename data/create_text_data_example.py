"""
Example script to create text data file for alignment training
This is a template - you need to provide your own text descriptions
"""
import json
import pickle

# Example: Create text data file
# Format 1: JSON file with list of texts
texts_json = [
    "A person is signing hello",
    "A person is signing thank you",
    "A person is signing goodbye",
    # ... add more text descriptions matching your video sequences
]

# Save as JSON
with open('text_data.json', 'w', encoding='utf-8') as f:
    json.dump({'texts': texts_json}, f, ensure_ascii=False, indent=2)

# Format 2: Pickle file with list
texts_pkl = [
    "A person is signing hello",
    "A person is signing thank you",
    # ... add more
]

# Save as pickle
with open('text_data.pkl', 'wb') as f:
    pickle.dump(texts_pkl, f)

print("Example text data files created!")
print("Note: You need to replace these with your actual text descriptions")
print("The number of texts should match the number of video sequences")

