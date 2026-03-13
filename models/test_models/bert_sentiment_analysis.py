from transformers import pipeline

emotion_pipeline = pipeline(
    "sentiment-analysis", 
    model="bhadresh-savani/distilbert-base-uncased-emotion",
    top_k=None
)

entry = input("Enter a text to analyze: ")
result = emotion_pipeline(entry)
print("Detected Emotions:")

for emotion in result[0]:
    print(f"Emotion: {emotion['label']}, Score: {emotion['score']}")


