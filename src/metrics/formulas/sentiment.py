from context.detect_emotions import EmotionDetector


def calculate_sentiment(img_list, model="llava:13b"):
    sentiment_results = {}
    total_sentiment_score = 0
    total_images = len(img_list)

    for image_path in img_list:
        detector = EmotionDetector(model=model, image_path=image_path)
        emotion_counts = detector.detect_emotions()

        positive_sentiments = emotion_counts["happy"] + emotion_counts["enjoyable"] + emotion_counts["relaxed"]
        negative_sentiments = emotion_counts["angry"]
        total_sentiments = positive_sentiments + negative_sentiments + emotion_counts["neutral"]

        sentiment_score = (positive_sentiments - negative_sentiments) / total_sentiments if total_sentiments > 0 else 0
        sentiment_results[image_path] = {
            "sentiment": sentiment_score,
            "count": emotion_counts
        }
        total_sentiment_score += sentiment_score

    average_sentiment_score = total_sentiment_score / total_images if total_images > 0 else 0

    result = {
        "sentiment": sentiment_results,
        "average": average_sentiment_score
    }

    return result


# Example usage
# img_list = ["/home/lucy/Documents/ai-ml/code/angelhack24heineiken-master/heineiken_raw_imgs/BZ1A2489.jpg"]
# sentiment = calculate_sentiment(img_list)
# print(sentiment)
