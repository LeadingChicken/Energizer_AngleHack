import predict_yolo

img_list = ["/home/lucy/Documents/ai-ml/code/angelhack24heineiken-master/heineiken_raw_imgs/BZ1A0441.jpg"]
beer_logo_class = {'bivina_logo', 'heineken_logo', 'larue_logo', 'saigon_logo', 'strongbow_logo', 'tiger_logo', 'biaviet_logo'}

def count_beer_consumer(logo_name, beer_logos, people, items):
    count = 0
    bottles = predict_yolo.categorize_bottle_with_logo(items, logo_name, beer_logos)
    for person in people:
        if predict_yolo.check_drinking(person, bottles):
            count += 1
    return count

def beer_consumption(logo_name, class_count, beer_logos, people, items):
    num_drinkers = count_beer_consumer(logo_name, beer_logos, people, items)
    total_people = class_count['consumer'] + class_count['pg_marketer'] + class_count['staff']
    return num_drinkers, (num_drinkers / total_people) * 100 if total_people > 0 else 0

# Example usage
# class_count, _ = predict_yolo.handle_predictions(img_list)
# beer_logos, people, items = predict_yolo.categorize_objects(img_list)
# print(beer_consumption("tiger_logo", class_count, beer_logos, people, items))
# print(beer_consumption("heineken_logo", class_count, beer_logos, people, items))
