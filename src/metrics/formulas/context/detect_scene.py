import ollama


class SceneDetector:
    def __init__(self, model, image_path):
        self.model = model
        self.image_path = image_path

    def detect_scene(self):
        prompt = (
            "Pick only one word in (bar, pub, restaurant, grocery, supermarket) to describe scene of this image?"
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
            scenes = ["bar", "pub", "restaurant", "grocery", "supermarket"]
            for scene in scenes:
                if scene in response_content:
                    return scene
            return "unknown"
        except KeyError:
            print("Error: Response does not contain the expected key.")
            return "unknown"

#
# detector = SceneDetector(model="llava:13b",
#                          image_path="/heineiken_raw_imgs/66501431_1705485180833.jpg")
# scene = detector.detect_scene()
# print(scene)
