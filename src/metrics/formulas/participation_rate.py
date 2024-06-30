import predict_yolo


def participation_rate(class_count_list):
    count = sum(1 for class_count in class_count_list if class_count['consumer'] > 0)
    return count, (count / len(class_count_list)) * 100

# Example usage
# img_list = ["data/pic1.jpg","data/pic2.jpg","data/pic3.jpg","data/pic4.jpg","data/pic5.jpg","data/pic6.jpg","data/pic7.jpg","data/pic8.jpg"]
# class_count_list = [predict_yolo.handle_predictions([pic_path])[0] for pic_path in img_list]
# print(participation_rate(class_count_list))
