import ollama


class EmotionDetector:
    def __init__(self, model, image_path):
        self.model = model
        self.image_path = image_path

    def detect_emotions(self):
        prompt = (
            "Count number of people by each emotions in list (Happy, Angry, Enjoyable, Relaxed, Neutral) of this image?"
        )

        try:
            res = ollama.chat(
                model=self.model,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                        'images': [self.image_path]
                    }
                ]
            )
            response_content = res['message']['content'].lower()
            print(response_content)
            emotions = ["happy", "angry", "enjoyable", "relaxed", "neutral"]
            emotion_counts = {emotion: 0 for emotion in emotions}

            for line in response_content.split('\n'):
                for emotion in emotions:
                    if emotion in line:
                        try:
                            count = int(line.split(':')[1].split()[0].strip())
                            emotion_counts[emotion] = count
                        except (IndexError, ValueError):
                            emotion_counts[emotion] = 0

            return emotion_counts
        except KeyError:
            print("Error: Response does not contain the expected key.")
            return {emotion: 0 for emotion in ["happy", "angry", "enjoyable", "relaxed", "neutral"]}
        except Exception as e:
            print(f"Error: {e}")
            return {emotion: 0 for emotion in ["happy", "angry", "enjoyable", "relaxed", "neutral"]}


# Example usage
# detector = EmotionDetector(model="llava:13b",
#                            image_path="/home/lucy/Documents/ai-ml/code/angelhack24heineiken-master/heineiken_raw_imgs/66501431_1705485008972.jpg")
# emotion_counts = detector.detect_emotions()
# print(emotion_counts)
