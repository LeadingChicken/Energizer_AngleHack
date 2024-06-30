from roboflow import Roboflow
rf = Roboflow(api_key="vu4c9hnnZyndIDNekNqs")
project = rf.workspace("energizerhackathon").project("hackathon-c7jmg")
version = project.version(6)
dataset = version.download("yolov8")
