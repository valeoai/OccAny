import numpy as np



class ClassMapping:
    def __init__(self):
        
        self.common_classes_2_id, self.id_2_common_classes = self.define_common_classes()
        self.id_2_cityscapes_classes, self.cityscapes_classes_2_common_classes, \
            self.cityscapes_classes_2_common_classes_for_training = self.define_cityscapes_class_mapping()
        self.id_2_kitti_classes, self.kitti_classes_2_common_classes = self.define_kitti_class_mapping()
        
        # self.cityscapes_id_2_common_id = self.create_class_id_mapping(self.id_2_cityscapes_classes, self.cityscapes_classes_2_common_classes, self.common_classes_2_id)
        self.kitti_id_2_common_id = self.create_class_id_mapping(self.id_2_kitti_classes, self.kitti_classes_2_common_classes, self.common_classes_2_id)
        # self.cityscapes_id_2_common_id_for_training = self.create_class_id_mapping(self.id_2_cityscapes_classes, self.cityscapes_classes_2_common_classes_for_training, self.common_classes_2_id)
        
        self.define_colors()
        
        
    def map_cityscapes_id_2_common_id(self, cityscapes_id, unknown_to_empty=False):
        common_id = self.cityscapes_id_2_common_id[cityscapes_id]
        if unknown_to_empty:
            common_id[common_id == 255] = 0
        return common_id

    def map_cityscapes_id_2_common_id_for_training(self, cityscapes_id, unknown_to_empty=False):
        common_id = self.cityscapes_id_2_common_id_for_training[cityscapes_id]
        if unknown_to_empty:
            common_id[common_id == 255] = 0
        return common_id
    
    def map_kitti_id_2_common_id(self, kitti_id):
        valid_mask = kitti_id != 255
        common_id = np.full_like(kitti_id, 255, dtype=np.uint8)
        common_id[valid_mask] = self.kitti_id_2_common_id[kitti_id[valid_mask]]
        return common_id
    
        
    def create_class_id_mapping(self, source_classes, target_from_source_classes, target_classes_2_id):
        source_id_2_target_id = 255 * np.ones(len(source_classes), dtype=np.int32)
        for target_class_name, source_class_list in target_from_source_classes.items():
            for source_class_name in source_class_list:
                source_id_2_target_id[source_classes.index(source_class_name)] = target_classes_2_id[target_class_name]
        
        return source_id_2_target_id
        
    
    
    
    def define_kitti_class_mapping(self):
        # id_2_kitti_classes = [
        #     "empty",
        #     "car",
        #     "bicycle",
        #     "motorcycle",
        #     "truck",
        #     "other-vehicle",
        #     "person",
        #     "bicyclist",
        #     "motorcyclist",
        #     "road",
        #     "parking",
        #     "sidewalk",
        #     "other-ground",
        #     "building",
        #     "fence",
        #     "vegetation",
        #     "trunk",
        #     "terrain",
        #     "pole",
        #     "traffic-sign",
        # ]
        id_2_kitti_classes = [
            "empty", # "empty", we map sky to empty
            "car",
            "bicycle",
            "motorcycle",
            "truck",
            "other-vehicle",
            "person",
            "bicyclist",
            "motorcyclist",
            "road",
            "parking",
            "sidewalk",
            "other-ground",
            "building",
            "fence",
            "vegetation",
            "trunk",
            "terrain",
            "pole",
            "traffic-sign",
            "sky", # added by us
        ]
        
        kitti_classes_2_common_classes = {
            "empty": ["empty", "sky"],
            "car": ["car"],
            "rider": ["bicycle", "motorcycle", "bicyclist", "motorcyclist"],
            "truck": ["truck"],
            "other-vehicle": ["other-vehicle"],
            "person": ["person"],
            "road": ["road"],
            "sidewalk": ["sidewalk"],
            "building": ["building"],
            # "fence": ["fence"],
            "vegetation": ["vegetation", "trunk"],
            "terrain": ["terrain"],
            # "pole": ["pole"],
            # "traffic-sign": ["traffic-sign"],
            "other-objects": ["fence", "pole", "traffic-sign"],
            "other-ground": ["parking", "other-ground"],
        }
        
        return id_2_kitti_classes, kitti_classes_2_common_classes
    
    
    def define_common_classes(self):
        # common_classes_2_id = {
        #     "empty": 0,
        #     "car": 1,
        #     "rider": 2,
        #     "truck": 3,
        #     "other-vehicle": 4,
        #     "person": 5,
        #     "road": 6,
        #     "sidewalk": 7,
        #     "building": 8,
        #     "fence": 9,
        #     "vegetation": 10,
        #     "terrain": 11,
        #     "pole": 12,
        #     "traffic-sign": 13,
        #     "unknown": 255,
        # }
        
        common_classes_2_id = {
            "empty": 0,
            "car": 1,
            "rider": 2,
            "truck": 3,
            "other-vehicle": 4,
            "person": 5,
            "road": 6,
            "sidewalk": 7,
            "building": 8,
            # "fence": 9,
            "vegetation": 9,
            "terrain": 10,
            "other-objects": 11,
            # "sky": 12,
            # "pole": 12,
            # "traffic-sign": 13,
            "other-ground": 12,
            "unknown": 255,
        } # for evaluation
        id_2_common_classes = {v: k for k, v in common_classes_2_id.items()}
        
        return common_classes_2_id, id_2_common_classes
    
    
    def define_cityscapes_class_mapping(self):
        id_2_cityscapes_classes = [
            "Road",
            "Sidewalk",
            "Building",
            "Wall",
            "Fence",
            "Pole",
            "Traffic Light",
            "Traffic Sign",
            "Vegetation",
            "Terrain",
            "Sky",
            "Person",
            "Rider",
            "Car",
            "Truck",
            "Bus",
            "Train",
            "Motorcycle",
            "Bicycle",
        ]
        
        cityscapes_classes_2_common_classes = {
            "car": ["Car"],
            "rider": ["Bicycle", "Motorcycle", "Rider"],
            "truck": ["Truck"],
            "other-vehicle": ["Bus", "Train"],
            "person": ["Person"],
            "road": ["Road"],
            "sidewalk": ["Sidewalk"],
            "building": ["Building"],
            # "fence": ["Fence", "Wall"],
            "vegetation": ["Vegetation"],
            "terrain": ["Terrain"],
            # "pole": ["Pole"],
            # "traffic-sign": ["Traffic Sign"],
            "other-objects": ["Fence", "Wall", "Pole", "Traffic Sign"],
            "unknown": ["Sky", "Traffic Light" ],
        }
        
        cityscapes_classes_2_common_classes_for_training = {
            "car": ["Car"],
            "rider": ["Bicycle", "Motorcycle", "Rider"],
            "truck": ["Truck"],
            "other-vehicle": ["Bus", "Train"],
            "person": ["Person"],
            "road": ["Road"],
            "sidewalk": ["Sidewalk"],
            "building": ["Building"],
            # "fence": ["Fence", "Wall"],
            "vegetation": ["Vegetation"],
            "terrain": ["Terrain"],
            # "pole": ["Pole"],
            # "traffic-sign": ["Traffic Sign"],
            "other-objects": ["Fence", "Wall", "Pole", "Traffic Sign"],
            "sky": ["Sky"],
            "unknown": ["Traffic Light" ],
        }
        
        return id_2_cityscapes_classes, cityscapes_classes_2_common_classes, cityscapes_classes_2_common_classes_for_training
    
    def get_color(self, class_id):
        class_id[class_id == 255] = 13
        colors = self.common_class_id_2_color[class_id]
        return colors # [H, W, 4]
    
 
    def define_colors(self):
        self.common_class_id_2_color = np.array([
            [0, 0, 0, 255], #"empty": 0,
            [100, 150, 245, 255], # "car": 1,
            [30, 60, 150, 255], # "rider": 2,
            [80, 30, 180, 255], #"truck": 3,
            [100, 80, 250, 255], #"other-vehicle": 4,
            [255, 30, 30, 255], #"person": 5,
            [255, 0, 255, 255], # "road": 6,
            [75, 0, 75, 255], # "sidewalk": 7,
            [255, 200, 0, 255], # "building": 8,
            # [255, 120, 50, 255], # "fence": ,
            [0, 175, 0, 255], # "vegetation": 9,
            [150, 240, 80, 255], #"terrain": 10,
            [255, 240, 150, 255], #"pole": 11,
            # [255, 0, 0, 255], #"traffic-sign": ,
            [135, 206, 235, 255], # "sky": 12,
            [255, 255, 255, 255], # "unknown": 255,
        ])[:,:3].astype(np.uint8)
        
        self.kitti_class_id_2_color = np.array([
            [0, 0, 0, 255], # "empty"
            [100, 150, 245, 255], # "car"
            [100, 230, 245, 255], # "bicycle"
            [30, 60, 150, 255], # "motorcycle"
            [80, 30, 180, 255], # "truck"
            [100, 80, 250, 255], # "other-vehicle"
            [255, 30, 30, 255], # "person"
            [255, 40, 200, 255], # "rider"
            [150, 30, 90, 255], # "motorcyclist"
            [255, 0, 255, 255], # "road"
            [255, 150, 255, 255], # "parking"
            [75, 0, 75, 255], # "sidewalk"
            [175, 0, 75, 255], # "other-ground"
            [255, 200, 0, 255], # "building"
            [255, 120, 50, 255], # "fence"
            [0, 175, 0, 255], # "vegetation"
            [135, 60, 0, 255], # "trunk"
            [150, 240, 80, 255], # "terrain"
            [255, 240, 150, 255], # "pole"
            [255, 0, 0, 255], # "traffic-sign"
            [255, 255, 255, 255], # "unknown"
        
        ])
  
        
