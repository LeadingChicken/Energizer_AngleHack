from context.detect_scene import SceneDetector


def calculate_venue_spread(img_list, model="llava:13b"):
    scene_counts = {
        "bar": 0,
        "pub": 0,
        "restaurant": 0,
        "grocery": 0,
        "supermarket": 0,
        "unknown": 0
    }

    total_images = len(img_list)

    for image_path in img_list:
        detector = SceneDetector(model=model, image_path=image_path)
        scene = detector.detect_scene()
        if scene in scene_counts:
            scene_counts[scene] += 1
        else:
            scene_counts["unknown"] += 1

    venue_spread = {scene: (count / total_images) * 100 for scene, count in scene_counts.items()}

    return venue_spread


# Example usage
# img_list = ["/home/lucy/Documents/ai-ml/code/angelhack24heineiken-master/heineiken_raw_imgs/66501431_1705485180833.jpg"]
# venue_spread = calculate_venue_spread(img_list)
# print(venue_spread)
