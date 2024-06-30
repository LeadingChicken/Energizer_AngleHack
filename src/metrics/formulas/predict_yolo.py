from roboflow import Roboflow

rf = Roboflow(api_key="07QNF5R45Bwvu1gvNA5o")
project = rf.workspace("anglehackathonenergizer").project("heineken-czc3t")
model = project.version(3).model


class Object():
    def __init__(self, x, y, w, h, name):
        self.open = 100
	self.xul = x - w//2
	self.yul = y - h//2
	self.xlr = x + w//2
	self.ylr = y + h//2
	self.name = name
	self.xCenter = (self.xul+self.xlr)//2
	self.yCenter = (self.yul+self.ylr)//2
	self.xulopen = min(0,self.xul-self.open)
	self.yulopen = min(0,self.yul-self.open)
	self.xlropen = self.xlr+self.open
	self.ylropen = self.ylr+self.open


# Check whether (x,y) inside a rectangle (xul,yul,xlr,ylr)
def check_point_inside_rectangle(xul, yul, xlr, ylr, x, y):
    if x >= xul and x <= xlr and y >= yul and y <= ylr:
        return True
    return False


# Check whether an item is used by a person
def check_used_item(item, people):
    for person in people:
        if check_point_inside_rectangle(person.xulopen, person.yulopen, person.xlropen, person.ylropen, item.xCenter,
                                        item.yCenter):
            return True

    return False


# Check whether a person is drinking or not
def check_drinking(person, bottles):
    for bottle in bottles:
        if check_point_inside_rectangle(person.xulopen, person.yulopen, person.xlropen, person.ylropen, bottle.xCenter,
                                        bottle.yCenter):
            return True

    return False


# output logo of the item or none
def find_items_logo(item, logos):
    for logo in logos:
        # if the logo center is inside that item
        if check_point_inside_rectangle(item.xul, item.yul, item.xlr, item.ylr, logo.xCenter, logo.yCenter):
            return logo.name

    return None


def check_items_in_logo_detection(item, logos, logo_name):
    for logo in logos:
        if logo.name == logo_name and check_point_inside_rectangle(item.xulopen, item.yulopen, item.xlropen,
                                                                   item.ylropen, logo.xCenter, logo.yCenter):
            return True

    return False


# task: count appearence of each class
def count_classes(predictions, class_count):
    data = predictions['predictions']
    for obj in data:
        class_count[obj['class']] += 1


def handle_predictions(img_list):
    class_count = {
        "beer_bottle": 0,
        "beer_keg": 0,
        "billboard": 0,
        "bivina_logo": 0,
        "biaviet_logo": 0,
        "bucket": 0,
        "campain-objects": 0,
        "consumer": 0,
        "display-stand": 0,
        "fridge": 0,
        "heineken_logo": 0,
        "larue_logo": 0,
        "parasol": 0,
        "pg_marketer": 0,
        "signage": 0,
        "staff": 0,
        "standee": 0,
        "strongbow_logo": 0,
        "saigon_logo": 0,
        "tent-card": 0,
        "tiger_logo": 0,
    }

    logo_classes = ['bivina_logo', 'heineken_logo', 'larue_logo', 'saigon_logo', 'strongbow_logo', 'tiger_logo', 'biaviet_logo']
    logo_count = {logo: {cls: 0 for cls in class_count.keys()} for logo in logo_classes}

    # task: categorize objects
    objects = []
    beer_logo_class = set(logo_classes)
    beer_logo = []
    people_class = {'consumer', 'pg_marketer', 'staff'}
    people = []
    items_class = {'beer_bottle', 'beer_keg', 'billboard', 'bucket', 'campain-objects', 'display-stand', 'fridge',
                   'parasol', 'signage', "tent-card", "standee"}
    items = []

    for pic_path in img_list:
        predictions = model.predict(pic_path, confidence=30, overlap=30).json()
        model.predict(pic_path, confidence=40, overlap=30).save("prediction.jpg")
        count_classes(predictions, class_count)
        for value in predictions['predictions']:
            # put objects in different categories
            item = Object(value['x'], value['y'], value['width'], value['height'], value['class'])
            objects.append(item)
            if item.name in beer_logo_class:
                beer_logo.append(item)
            elif item.name in people_class:
                people.append(item)
            else:
                items.append(item)

    for item in items:
        logo = find_items_logo(item, beer_logo)
        if logo:
            logo_count[logo][item.name] += 1

    result = {
        "total": class_count,
        "brands": logo_count
    }
    print(result)
    return result, predictions


def categorize_objects(img_list, predictions):
    objects = []
    beer_logo_class = {'bivina_logo', 'heineken_logo', 'larue_logo', 'saigon_logo', 'strongbow_logo', 'tiger_logo',
                       'biaviet_logo'}
    beer_logos = []
    people_class = {'consumer', 'pg_marketer', 'staff'}
    people = []
    items_class = {'beer_bottle', 'beer_keg', 'billboard', 'bucket', 'campain-objects', 'display-stand', 'fridge',
                   'parasol', 'signage', "tent-card", "standee"}
    items = []

    for pic_path in img_list:
        for value in predictions['predictions']:
            # put objects in different categories
            item = Object(value['x'], value['y'], value['width'], value['height'], value['class'])
            objects.append(item)
            if item.name in beer_logo_class:
                beer_logos.append(item)
            elif item.name in people_class:
                people.append(item)
            else:
                items.append(item)
    print(beer_logos, people, items)
    return beer_logos, people, items


def categorize_items_with_logo(items, logo_name, logos):
    logo_items = []
    for item in items:
        if find_items_logo(item, logos) == logo_name:
            logo_items.append(item)

    return logo_items


def categorize_bottle_with_logo(items, logo_name, logos):
    logo_items = []
    for item in items:
        if item.name == "beer_bottle" and find_items_logo(item, logos) == logo_name:
            logo_items.append(item)

    return logo_items


import os

# img_list = ["/home/lucy/Documents/ai-ml/code/angelhack24heineiken-master/heineiken_raw_imgs/BZ1A0441.jpg"]
# image_dir = "/home/lucy/Documents/ai-ml/code/angelhack24heineiken-master/heineiken_raw_imgs"
# image_files = sorted(os.listdir(image_dir))[:1]
# for image_file in image_files:
#     img_list.append(os.path.join(image_dir, image_file))
# 
# print(img_list)

# class_count, beer_logo_count = handle_predictions(img_list)
# print(handle_predictions(img_list))
# print(categorize_objects(img_list))

