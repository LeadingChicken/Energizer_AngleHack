import predict_yolo
from brand_reach import brand_reach
from beer_consumption import beer_consumption
from campaign_correct_setup import count_correct_setup
from engagement import handle_engagement
from material_detection import handle_material_detection
from material_usage import count_material_used
from participation_rate import participation_rate
from pg_count import count_pg
from activities_breakdown import calculate_activities_breakdown
from cost_estimation import calculate_cost_estimation
from pg_efficiency_metric import calculate_pg_efficiency
from sentiment import calculate_sentiment
from venue_spread import calculate_venue_spread

def calculate_metrics(img_list):
    # Calculate venue spread for total images
    venue_spread = calculate_venue_spread(img_list)

    metrics = {
        "total": {},
        "images": {},
        "venue_spread": venue_spread,
        "sentiment": [],
        "pg_efficiency": [],
        "cost_estimation": [],
        "activities_breakdown": []
    }

    total_class_count = {}
    total_logo_count = {}
    total_activities = {}
    total_cost = 0
    total_efficiency_score = 0
    total_sentiment_score = 0

    for image_path in img_list:
        result, predictions = predict_yolo.handle_predictions([image_path])
        class_count = result["total"]
        logo_count = result["brands"]
        beer_logos, people, items = predict_yolo.categorize_objects([image_path], predictions)

        # Update total counts
        for key, value in class_count.items():
            total_class_count[key] = total_class_count.get(key, 0) + value
        for key, value in logo_count.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    total_logo_count[sub_key] = total_logo_count.get(sub_key, 0) + sub_value
            else:
                total_logo_count[key] = total_logo_count.get(key, 0) + value

        # Calculate individual metrics
        image_metrics = {
            "brand_reach": {logo: brand_reach(logo, class_count, len(img_list)) for logo in logo_count.keys()},
            "beer_consumption": {logo: beer_consumption(logo, class_count, beer_logos, people, items) for logo in logo_count.keys()},
            "correct_setup": count_correct_setup([class_count]),
            "engagement": {logo: handle_engagement(logo, beer_logos, people) for logo in logo_count.keys()},
            "material_detection": {logo: handle_material_detection(logo, class_count, beer_logos, items) for logo in logo_count.keys()},
            "material_usage": {logo: count_material_used(logo, class_count, logo_count) for logo in logo_count.keys()},
            "participation_rate": participation_rate([class_count]),
            "pg_count": count_pg([class_count])
        }

        metrics["images"][image_path] = image_metrics

        # Calculate additional metrics
        activities_breakdown = calculate_activities_breakdown([image_path])
        cost_estimation = calculate_cost_estimation([image_path])
        pg_efficiency = calculate_pg_efficiency([image_path])
        sentiment = calculate_sentiment([image_path])

        # Add additional metrics to the results
        metrics["activities_breakdown"].append({image_path: activities_breakdown})
        metrics["cost_estimation"].append({image_path: cost_estimation})
        metrics["pg_efficiency"].append({image_path: pg_efficiency})
        metrics["sentiment"].append({image_path: sentiment})

        # Update total metrics
        for activity, count in activities_breakdown.items():
            total_activities[activity] = total_activities.get(activity, 0) + count

        total_cost += cost_estimation["total_sum"]
        total_efficiency_score += pg_efficiency["average"]
        total_sentiment_score += sentiment["average"]

    # Calculate averages for total metrics
    average_efficiency_score = total_efficiency_score / len(img_list) if len(img_list) > 0 else 0
    average_sentiment_score = total_sentiment_score / len(img_list) if len(img_list) > 0 else 0

    metrics["total"] = {
        "class_count": total_class_count,
        "logo_count": total_logo_count,
        "activities_breakdown": total_activities,
        "cost_estimation": total_cost,
        "pg_efficiency": average_efficiency_score,
        "sentiment": average_sentiment_score
    }

    return metrics

# Example usage
if __name__ == "__main__":
    img_list = ["/home/lucy/Documents/ai-ml/code/angelhack24heineiken-master/heineiken_raw_imgs/BZ1A0441.jpg"]
    metrics = calculate_metrics(img_list)
    print(metrics)