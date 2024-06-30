from src.metrics.formulas.predict_yolo import handle_predictions

costs = {
    "pg_marketer": 3000000,  # Promotion Girl
    "bucket": 100000,  # Ice bucket
    "fridge": 7000000,  # Fridge
    "billboard": 200000,  # Signage, billboard
    "signage": 100000,  # Signage, poster
    "standee": 800000,  # Standee
    "display_stand": 600000,  # Tent card, display stand
    "tent_card": 200000,  # Tent card
    "parasol": 150000  # Parasol
}


def calculate_cost_estimation(img_list):
    total_sum = 0
    cost_results = {}

    for image_path in img_list:
        result = handle_predictions([image_path])
        class_count = result["total"]
        image_cost = {item: class_count.get(item, 0) * cost for item, cost in costs.items()}
        image_sum = sum(image_cost.values())
        cost_results[image_path] = {
            "cost": image_cost,
            "sum": image_sum
        }
        total_sum += image_sum

    result = {
        "images": cost_results,
        "total_sum": total_sum
    }

    return result


# Example usage
# img_list = ["/home/lucy/Documents/ai-ml/code/angelhack24heineiken-master/heineiken_raw_imgs/BZ1A2489.jpg"]
# cost_estimation = calculate_cost_estimation(img_list)
# print(cost_estimation)
