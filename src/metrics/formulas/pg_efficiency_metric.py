from context.pg_efficiency import EfficiencyAnalyzer


def calculate_pg_efficiency(img_list, model="llava:13b"):
    efficiency_results = {}
    total_efficiency_score = 0
    total_images = len(img_list)

    for image_path in img_list:
        analyzer = EfficiencyAnalyzer(model=model, image_path=image_path)
        is_efficient = analyzer.is_efficient()
        efficiency_score = 100 if is_efficient else 0
        efficiency_results[image_path] = efficiency_score
        total_efficiency_score += efficiency_score

    average_efficiency_score = total_efficiency_score / total_images if total_images > 0 else 0

    result = average_efficiency_score
    return result


# Example usage
# img_list = ["/home/lucy/Documents/ai-ml/code/angelhack24heineiken-master/heineiken_raw_imgs/BZ1A2489.jpg"]
# efficiency = calculate_pg_efficiency(img_list)
# print(efficiency)
