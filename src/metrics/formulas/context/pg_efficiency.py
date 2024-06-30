import ollama


class EfficiencyAnalyzer:
    def __init__(self, model, image_path):
        self.model = model
        self.image_path = image_path

    def is_efficient(self):
        prompt = (
            "Analyze this image to determine the efficiency of the promotional girl based on the following:\n"
            "Presence Detection: Verify if the promotional girl is present in the images.\n"
            "Activity Detection: Identify activities such as interacting with customers, handing out promotional materials, etc.\n"
            "Interactions Count: Count the number of interactions with customers.\n"
            "Customer Expressions: Analyze customer facial expressions to gauge interest or satisfaction.\n"
            "Based on these criteria, provide a response with the keyword indicating whether the promotional girl is efficient (isEfficient: true) or not (isEfficient: false) only, with no description."
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
            return 'isefficient: true' in response_content
        except KeyError:
            print("Error: Response does not contain the expected key.")
            return False


# analyzer = EfficiencyAnalyzer(model="llava:13b",
#                               image_path="/home/lucy/Documents/ai-ml/code/angelhack24heineiken/heineiken_raw_imgs/BZ1A2489.jpg")
# print(analyzer.is_efficient())
