import predict_yolo


def count_material_used(logo_name, class_count, logo_count):
    logo_items = logo_count[logo_name]
    count = sum(logo_items.values())
    countItems = sum(class_count.values())
    return count, countItems, (count / countItems) * 100 if countItems > 0 else 0

# Example usage
# img_list = ["data/pic1.jpg","data/pic2.jpg","data/pic3.jpg","data/pic4.jpg","data/pic5.jpg","data/pic6.jpg","data/pic8.jpg","data/pic7.jpg"]
# class_count, logo_count = predict_yolo.handle_predictions(img_list)
# print(count_material_used("tiger_logo", class_count, logo_count))
