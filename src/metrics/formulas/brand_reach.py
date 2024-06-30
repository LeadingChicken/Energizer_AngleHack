import predict_yolo


def brand_count(logo, class_count):
    return class_count[logo]


def brand_reach(logo, class_count, n):
    return (brand_count(logo, class_count) / n) * 100


# Example usage
# img_list = ["/home/lucy/Documents/ai-ml/code/angelhack24heineiken-master/heineiken_raw_imgs/BZ1A0441.jpg"]
# class_count, _ = predict_yolo.handle_predictions(img_list)
# n = len(img_list)
# print(brand_reach("biaviet_logo", class_count, n))
# print(brand_reach("tiger_logo", class_count, n))
