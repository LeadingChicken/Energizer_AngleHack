import predict_yolo

def check_correct_setup(class_count):
    # >=1 billboard, >=10 beer kegs, >=1 fridge
    return class_count['billboard'] > 1 and class_count['beer_keg'] >= 10 and class_count['fridge'] >= 1

def count_correct_setup(class_count_list):
    count = sum(1 for class_count in class_count_list if check_correct_setup(class_count))
    return count, (count / len(class_count_list)) * 100

# Example usage
# img_list = ["data/pic1.jpg","data/pic2.jpg","data/pic3.jpg","data/pic4.jpg","data/pic5.jpg","data/pic6.jpg"]
# class_count_list = [predict_yolo.handle_predictions([pic_path])[0] for pic_path in img_list]
# print(count_correct_setup(class_count_list))
