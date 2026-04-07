from roboflow import Roboflow
rf = Roboflow(api_key="hba9afcfCPh5nFssLBL4")
project = rf.workspace("supravs-workspace").project("staircase-4ebwt-eolyn")
version = project.version(1)
dataset = version.download("yolov8")
                