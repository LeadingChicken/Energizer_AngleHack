import ollama


class ActivityDetector:
    def __init__(self, model, image_path):
        self.model = model
        self.image_path = image_path

    def detect_activities(self):
        prompt = (
            "Pick words in (Eating, Drinking, Smiling, Talking, Shopping) to describe activities of people in this image?"
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
            activities = [activity for activity in ["eating", "drinking", "smiling", "talking", "shopping"] if
                          activity in response_content]
            return ', '.join(activities)
        except KeyError:
            print("Error: Response does not contain the expected key.")
            return ""

#
# detector = ActivityDetector(model="llava:34b",
#                             image_path="/heineiken_raw_imgs/BZ1A2489.jpg")
# print(detector.detect_activities())
