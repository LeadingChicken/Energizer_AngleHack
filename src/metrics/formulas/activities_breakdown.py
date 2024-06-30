from context.detect_activities import ActivityDetector


def calculate_activities_breakdown(img_list, model="llava:13b"):
    activity_counts = {
        "eating": 0,
        "drinking": 0,
        "smiling": 0,
        "talking": 0,
        "shopping": 0
    }

    total_activities = 0

    for image_path in img_list:
        detector = ActivityDetector(model=model, image_path=image_path)
        activities = detector.detect_activities().split(', ')
        for activity in activities:
            if activity in activity_counts:
                activity_counts[activity] += 1
                total_activities += 1

    activity_breakdown = {activity: (count / total_activities) * 100 if total_activities > 0 else 0 for activity, count
                          in activity_counts.items()}

    return activity_breakdown


# Example usage
# img_list = ["/home/lucy/Documents/ai-ml/code/angelhack24heineiken-master/heineiken_raw_imgs/BZ1A2489.jpg"]
# activity_breakdown = calculate_activities_breakdown(img_list)
# print(activity_breakdown)
