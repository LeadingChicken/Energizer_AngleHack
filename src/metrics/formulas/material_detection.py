import predict_yolo

beer_logo_class = {'bivina_logo', 'heineken_logo', 'larue_logo', 'saigon_logo', 'strongbow_logo', 'tiger_logo',
                   'biaviet_logo'}


def handle_material_detection(logo, class_count, beer_logos, items):
    promotion_materials_name = {"bucket", "standee", "parasol", "fridge", "campain-objects"}
    promotion_materials = [item for item in items if item.name in promotion_materials_name]
    num_materials = len(promotion_materials)
    count_logo_materials = sum(
        1 for item in promotion_materials if predict_yolo.find_items_logo(item, beer_logos) == logo)
    return count_logo_materials, (count_logo_materials / num_materials) * 100 if num_materials > 0 else 0


def material_detection(class_count, beer_logos, items):
    result = {}
    for logo in beer_logo_class:
        count, percentage = handle_material_detection(logo, class_count, beer_logos, items)
        result[logo] = {
            "count": count,
            "percentage": percentage
        }
    return result

# Example usage
# img_list = ["data/pic1.jpg","data/pic2.jpg","data/pic3.jpg","data/pic4.jpg","data/pic8.jpg"]
# beer_logo_class = {'bivina_logo', 'heineken_logo', 'larue_logo', 'saigon_logo', 'strongbow_logo', 'tiger_logo', 'biaviet_logo'}
# class_count, beer_logos, items = predict_yolo.categorize_objects(img_list)
# print(material_detection(class_count, beer_logos, items))
