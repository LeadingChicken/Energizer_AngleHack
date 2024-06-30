import predict_yolo

def handle_engagement(logo, beer_logos, people):
    num_people = len(people)
    count_engagement = sum(1 for person in people if predict_yolo.find_items_logo(person, beer_logos) == logo)
    return count_engagement, (count_engagement / num_people) * 100 if num_people > 0 else 0

# Example usage
# img_list = ["data/pic1.jpg","data/pic2.jpg","data/pic3.jpg","data/pic4.jpg","data/pic5.jpg","data/pic6.jpg"]
# beer_logos, people, _ = predict_yolo.categorize_objects(img_list)
# print(handle_engagement("heineken_logo", beer_logos, people))
