# Copyright (C) 2018  Artsiom Sanakoyeu and Dmytro Kotovenko
#
# This file is part of Adaptive Style Transfer
#
# Adaptive Style Transfer is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Adaptive Style Transfer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import print_function
# import pandas as pd
import numpy as np
import os
import time
from tqdm import tqdm
import scipy.misc
import utils
import random


class ArtDataset():
    def __init__(self, path_to_art_dataset):

        self.dataset = [os.path.join(path_to_art_dataset, x) for x in os.listdir(path_to_art_dataset)]
        print("Art dataset contains %d images." % len(self.dataset))

    def get_batch(self, augmentor, batch_size=1):
        """
        Reads data from dataframe data containing path to images in column 'path' and, in case of dataframe,
         also containing artist name, technique name, and period of creation for given artist.
         In case of content images we have only the 'path' column.
        Args:
            augmentor: Augmentor object responsible for augmentation pipeline
            batch_size: size of batch
        Returns:
            dictionary with fields: image
        """

        batch_image = []

        for _ in range(batch_size):
            image = scipy.misc.imread(name=random.choice(self.dataset), mode='RGB')

            if max(image.shape) > 1800.:
                image = scipy.misc.imresize(image, size=1800. / max(image.shape))
            if max(image.shape) < 800:
                # Resize the smallest side of the image to 800px
                alpha = 800. / float(min(image.shape))
                if alpha < 4.:
                    image = scipy.misc.imresize(image, size=alpha)
                    image = np.expand_dims(image, axis=0)
                else:
                    image = scipy.misc.imresize(image, size=[800, 800])

            if augmentor:
                batch_image.append(augmentor(image).astype(np.float32))
            else:
                batch_image.append((image).astype(np.float32))
        # Now return a batch in correct form
        batch_image = np.asarray(batch_image)

        return {"image": batch_image}

    def initialize_batch_worker(self, queue, augmentor, batch_size=1, seed=228):
        np.random.seed(seed)
        while True:
            batch = self.get_batch(augmentor=augmentor, batch_size=batch_size)
            queue.put(batch)


class PlacesDataset():
    # categories_names = \
    #     ['/a/airfield', '/a/airplane_cabin', '/a/airport_terminal', '/a/alcove', '/a/alley', '/a/amphitheater',
    #      '/a/amusement_arcade', '/a/amusement_park', '/a/apartment_building', '/a/apartment_building/outdoor',
    #      '/a/aquarium', '/a/aqueduct', '/a/arcade', '/a/arch', '/a/archaelogical_excavation', '/a/archive',
    #      '/a/arena/hockey', '/a/arena/performance', '/a/arena/rodeo', '/a/army_base', '/a/art_gallery', '/a/art_school',
    #      '/a/art_studio', '/a/artists_loft', '/a/assembly_line', '/a/athletic_field/outdoor',
    #      '/a/atrium/public', '/a/attic', '/a/auditorium', '/a/auto_factory', '/a/auto_showroom',
    #      '/b/badlands', '/b/bakery/shop', '/b/balcony/exterior', '/b/balcony/interior',
    #      '/b/ball_pit', '/b/ballroom', '/b/bamboo_forest', '/b/bank_vault', '/b/banquet_hall', '/b/bar', '/b/barn',
    #      '/b/barndoor', '/b/baseball_field', '/b/basement', '/b/basketball_court/indoor',
    #      '/b/bathroom', '/b/bazaar/indoor', '/b/bazaar/outdoor', '/b/beach', '/b/beach_house',
    #      '/b/beauty_salon', '/b/bedchamber', '/b/bedroom', '/b/beer_garden', '/b/beer_hall', '/b/berth',
    #      '/b/biology_laboratory', '/b/boardwalk', '/b/boat_deck', '/b/boathouse', '/b/bookstore', '/b/booth',
    #      '/b/booth/indoor', '/b/botanical_garden', '/b/bow_window/indoor', '/b/bowling_alley',
    #      '/b/boxing_ring', '/b/bridge', '/b/building_facade', '/b/bullring', '/b/burial_chamber', '/b/bus_interior',
    #      '/b/bus_station/indoor', '/b/butchers_shop', '/b/butte', '/c/cabin', '/c/cabin/outdoor',
    #      '/c/cafeteria', '/c/campsite', '/c/campus', '/c/canal/natural', '/c/canal/urban', '/c/candy_store',
    #      '/c/canyon', '/c/car_interior', '/c/carrousel', '/c/castle', '/c/catacomb', '/c/cemetery', '/c/chalet',
    #      '/c/chemistry_lab', '/c/childs_room', '/c/church/indoor', '/c/church/outdoor', '/c/classroom',
    #      '/c/clean_room', '/c/cliff', '/c/closet', '/c/clothing_store', '/c/coast', '/c/cockpit', '/c/coffee_shop',
    #      '/c/computer_room', '/c/conference_center', '/c/conference_room', '/c/construction_site', '/c/corn_field',
    #      '/c/corral', '/c/corridor', '/c/cottage', '/c/courthouse', '/c/courtyard', '/c/creek', '/c/crevasse',
    #      '/c/crosswalk', '/d/dam', '/d/delicatessen', '/d/department_store', '/d/desert/sand',
    #      '/d/desert/vegetation', '/d/desert_road', '/d/diner/outdoor', '/d/dining_hall', '/d/dining_room',
    #      '/d/discotheque', '/d/doorway/outdoor', '/d/dorm_room', '/d/downtown', '/d/dressing_room',
    #      '/d/driveway', '/d/drugstore', '/e/elevator', '/e/elevator/door', '/e/elevator_lobby', '/e/elevator_shaft',
    #      '/e/embassy', '/e/engine_room', '/e/entrance_hall', '/e/escalator/indoor', '/e/excavation',
    #      '/f/fabric_store', '/f/farm', '/f/fastfood_restaurant', '/f/field/cultivated', '/f/field/wild',
    #      '/f/field_road', '/f/fire_escape', '/f/fire_station', '/f/fishpond', '/f/flea_market/indoor',
    #      '/f/florist_shop/indoor', '/f/food_court', '/f/football_field', '/f/forest/broadleaf', '/f/forest_path',
    #      '/f/forest_road', '/f/formal_garden', '/f/fountain', '/g/galley',
    #      '/g/garage/indoor', '/g/garage/outdoor', '/g/gas_station', '/g/gazebo/exterior',
    #      '/g/general_store', '/g/general_store/indoor', '/g/general_store/outdoor', '/g/gift_shop', '/g/glacier',
    #      '/g/golf_course', '/g/greenhouse', '/g/greenhouse/indoor', '/g/greenhouse/outdoor', '/g/grotto',
    #      '/g/gymnasium/indoor', '/h/hangar/indoor', '/h/hangar/outdoor', '/h/harbor',
    #      '/h/hardware_store', '/h/hayfield', '/h/heliport', '/h/highway', '/h/home_office', '/h/home_theater',
    #      '/h/hospital', '/h/hospital_room', '/h/hot_spring', '/h/hotel/outdoor', '/h/hotel_room',
    #      '/h/house', '/h/hunting_lodge/outdoor', '/i/ice_cream_parlor', '/i/ice_floe',
    #      '/i/ice_shelf', '/i/ice_skating_rink/indoor', '/i/ice_skating_rink/outdoor',
    #      '/i/iceberg', '/i/igloo', '/i/industrial_area', '/i/inn/outdoor', '/i/islet', '/j/jacuzzi',
    #      '/j/jacuzzi/indoor', '/j/jail_cell', '/j/japanese_garden', '/j/jewelry_shop', '/j/junkyard', '/k/kasbah',
    #      '/k/kennel/outdoor', '/k/kindergarden_classroom', '/k/kitchen', '/l/lagoon', '/l/lake/natural', '/l/landfill',
    #      '/l/landing_deck', '/l/laundromat', '/l/lawn', '/l/lecture_room',
    #      '/l/legislative_chamber', '/l/library/indoor', '/l/library/outdoor', '/l/lighthouse',
    #      '/l/living_room', '/l/loading_dock', '/l/lobby', '/l/lock_chamber', '/l/locker_room', '/m/mansion',
    #      '/m/manufactured_home', '/m/market/indoor', '/m/market/outdoor', '/m/marsh',
    #      '/m/martial_arts_gym', '/m/mausoleum', '/m/medina', '/m/mezzanine', '/m/moat/water', '/m/mosque/outdoor',
    #      '/m/motel', '/m/mountain', '/m/mountain_path', '/m/mountain_snowy', '/m/movie_theater/indoor',
    #      '/m/museum', '/m/museum/indoor', '/m/museum/outdoor', '/m/music_studio',
    #      '/n/natural_history_museum', '/n/nursery', '/n/nursing_home', '/o/oast_house', '/o/ocean', '/o/office',
    #      '/o/office_building', '/o/office_cubicles', '/o/oilrig', '/o/operating_room', '/o/orchard', '/o/orchestra_pit',
    #      '/p/pagoda', '/p/palace', '/p/pantry', '/p/park', '/p/parking_garage/indoor',
    #      '/p/parking_garage/outdoor', '/p/parking_lot', '/p/pasture', '/p/patio', '/p/pavilion', '/p/pet_shop',
    #      '/p/pharmacy', '/p/phone_booth', '/p/physics_laboratory', '/p/picnic_area', '/p/pier', '/p/pizzeria',
    #      '/p/playground', '/p/playroom', '/p/plaza', '/p/pond', '/p/porch', '/p/promenade', '/p/pub', '/p/pub/indoor',
    #      '/r/racecourse', '/r/raceway', '/r/raft', '/r/railroad_track', '/r/rainforest', '/r/reception',
    #      '/r/recreation_room', '/r/repair_shop', '/r/residential_neighborhood', '/r/restaurant',
    #      '/r/restaurant_kitchen', '/r/restaurant_patio', '/r/rice_paddy', '/r/river', '/r/rock_arch', '/r/roof_garden',
    #      '/r/rope_bridge', '/r/ruin', '/r/runway', '/s/sandbox', '/s/sauna', '/s/schoolhouse', '/s/science_museum',
    #      '/s/server_room', '/s/shed', '/s/shoe_shop', '/s/shopfront', '/s/shopping_mall/indoor',
    #      '/s/shower', '/s/ski_resort', '/s/ski_slope', '/s/sky', '/s/skyscraper', '/s/slum', '/s/snowfield',
    #      '/s/soccer_field', '/s/stable', '/s/stadium/baseball', '/s/stadium/football',
    #      '/s/stadium/soccer', '/s/stage/indoor', '/s/stage/outdoor', '/s/staircase', '/s/storage_room',
    #      '/s/street', '/s/subway_station/platform', '/s/supermarket', '/s/sushi_bar', '/s/swamp',
    #      '/s/swimming_hole', '/s/swimming_pool/indoor', '/s/swimming_pool/outdoor', '/s/synagogue',
    #      '/s/synagogue/outdoor', '/t/television_room', '/t/television_studio', '/t/temple', '/t/temple/asia',
    #      '/t/throne_room', '/t/ticket_booth', '/t/topiary_garden', '/t/tower', '/t/toyshop', '/t/train_interior',
    #      '/t/train_station/platform', '/t/tree_farm', '/t/tree_house', '/t/trench', '/t/tundra',
    #      '/u/underwater', '/u/underwater/ocean_deep', '/u/utility_room', '/v/valley', '/v/vegetable_garden',
    #      '/v/veterinarians_office', '/v/viaduct', '/v/village', '/v/vineyard', '/v/volcano', '/v/volleyball_court',
    #      '/v/volleyball_court/outdoor', '/w/waiting_room', '/w/water_park', '/w/water_tower', '/w/waterfall',
    #      '/w/watering_hole', '/w/wave', '/w/wet_bar', '/w/wheat_field', '/w/wind_farm', '/w/windmill', '/y/yard',
    #      '/y/youth_hostel', '/z/zen_garden']

    categories_names = \
        ['/a/abbey', '/a/arch', '/a/amphitheater', '/a/aqueduct', '/a/arena/rodeo', '/a/athletic_field/outdoor',
         '/b/badlands', '/b/balcony/exterior', '/b/bamboo_forest', '/b/barn', '/b/barndoor', '/b/baseball_field',
         '/b/basilica', '/b/bayou', '/b/beach', '/b/beach_house', '/b/beer_garden', '/b/boardwalk', '/b/boathouse',
         '/b/botanical_garden', '/b/bullring', '/b/butte', '/c/cabin/outdoor', '/c/campsite', '/c/campus',
         '/c/canal/natural', '/c/canal/urban', '/c/canyon', '/c/castle', '/c/church/outdoor', '/c/chalet',
         '/c/cliff', '/c/coast', '/c/corn_field', '/c/corral', '/c/cottage', '/c/courtyard', '/c/crevasse',
         '/d/dam', '/d/desert/vegetation', '/d/desert_road', '/d/doorway/outdoor', '/f/farm', '/f/fairway',
         '/f/field/cultivated', '/f/field/wild', '/f/field_road', '/f/fishpond', '/f/florist_shop/indoor',
         '/f/forest/broadleaf', '/f/forest_path', '/f/forest_road', '/f/formal_garden', '/g/gazebo/exterior',
         '/g/glacier', '/g/golf_course', '/g/greenhouse/indoor', '/g/greenhouse/outdoor', '/g/grotto', '/g/gorge',
         '/h/hayfield', '/h/herb_garden', '/h/hot_spring', '/h/house', '/h/hunting_lodge/outdoor', '/i/ice_floe',
         '/i/ice_shelf', '/i/iceberg', '/i/inn/outdoor', '/i/islet', '/j/japanese_garden', '/k/kasbah',
         '/k/kennel/outdoor', '/l/lagoon', '/l/lake/natural', '/l/lawn', '/l/library/outdoor', '/l/lighthouse',
         '/m/mansion', '/m/marsh', '/m/mausoleum', '/m/moat/water', '/m/mosque/outdoor', '/m/mountain',
         '/m/mountain_path', '/m/mountain_snowy', '/o/oast_house', '/o/ocean', '/o/orchard', '/p/park',
         '/p/pasture', '/p/pavilion', '/p/picnic_area', '/p/pier', '/p/pond', '/r/raft', '/r/railroad_track',
         '/r/rainforest', '/r/rice_paddy', '/r/river', '/r/rock_arch', '/r/roof_garden', '/r/rope_bridge',
         '/r/ruin', '/s/schoolhouse', '/s/sky', '/s/snowfield', '/s/swamp', '/s/swimming_hole',
         '/s/synagogue/outdoor', '/t/temple/asia', '/t/topiary_garden', '/t/tree_farm', '/t/tree_house',
         '/u/underwater/ocean_deep', '/u/utility_room', '/v/valley', '/v/vegetable_garden', '/v/viaduct',
         '/v/village', '/v/vineyard', '/v/volcano', '/w/waterfall', '/w/watering_hole', '/w/wave',
         '/w/wheat_field', '/z/zen_garden', '/a/alcove', '/a/apartment-building/outdoor', '/a/artists_loft',
         '/b/building_facade', '/c/cemetery']
    categories_names = [x[1:] for x in categories_names]

    def __init__(self, path_to_dataset):
        self.dataset = []
        for category_idx, category_name in enumerate(tqdm(self.categories_names)):
            print(category_name, category_idx)
            if os.path.exists(os.path.join(path_to_dataset, category_name)):
                for file_name in tqdm(os.listdir(os.path.join(path_to_dataset, category_name))):
                    self.dataset.append(os.path.join(path_to_dataset, category_name, file_name))
            else:
                print("Category %s can't be found in path %s. Skip it." %
                      (category_name, os.path.join(path_to_dataset, category_name)))

        print("Finished. Constructed Places2 dataset of %d images." % len(self.dataset))

    def get_batch(self, augmentor, batch_size=1):
        """
        Generate bathes of images with attached labels(place category) in two different formats:
        textual and one-hot-encoded.
        Args:
            augmentor: Augmentor object responsible for augmentation pipeline
            batch_size: size of batch we return
        Returns:
            dictionary with fields: image
        """

        batch_image = []
        for _ in range(batch_size):
            image = scipy.misc.imread(name=random.choice(self.dataset), mode='RGB')
            image = scipy.misc.imresize(image, size=2.)
            image_shape = image.shape

            if max(image_shape) > 1800.:
                image = scipy.misc.imresize(image, size=1800. / max(image_shape))
            if max(image_shape) < 800:
                # Resize the smallest side of the image to 800px
                alpha = 800. / float(min(image_shape))
                if alpha < 4.:
                    image = scipy.misc.imresize(image, size=alpha)
                    image = np.expand_dims(image, axis=0)
                else:
                    image = scipy.misc.imresize(image, size=[800, 800])

            batch_image.append(augmentor(image).astype(np.float32))

        return {"image": np.asarray(batch_image)}

    def initialize_batch_worker(self, queue, augmentor, batch_size=1, seed=228):
        np.random.seed(seed)
        while True:
            batch = self.get_batch(augmentor=augmentor, batch_size=batch_size)
            queue.put(batch)




