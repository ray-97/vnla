from __future__ import division
import os
import sys
import csv
import numpy as np
import math
import base64
import json
import random
import networkx as nx
from collections import defaultdict
import scipy.stats
from pprint import pprint

sys.path.append('../../build')
import MatterSim

from oracle import make_oracle
from utils import load_datasets, load_nav_graphs
import utils

csv.field_size_limit(sys.maxsize)


class EnvBatch():

    def __init__(self, from_train_env=None, img_features=None, batch_size=100):
        if from_train_env is not None:
            self.features = from_train_env.features
            self.image_h  = from_train_env.image_h
            self.image_w  = from_train_env.image_w
            self.vfov     = from_train_env.vfov
        elif img_features is not None:
            self.image_h, self.image_w, self.vfov, self.features = \
                utils.load_img_features(img_features)
        else:
            print('Image features not provided')
            self.features = None
            self.image_w = 640
            self.image_h = 480
            self.vfov = 60
        self.sims = []
        for i in range(batch_size):
            sim = MatterSim.Simulator()
            sim.setRenderingEnabled(False)
            sim.setDiscretizedViewingAngles(True)
            sim.setCameraResolution(self.image_w, self.image_h)
            sim.setCameraVFOV(math.radians(self.vfov))
            sim.setNavGraphPath(
                os.path.join(os.getenv('PT_DATA_DIR', '../../../data'), 'connectivity'))
            sim.init()
            self.sims.append(sim)

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId

    def newEpisodes(self, scanIds, viewpointIds, headings):
        for i, (scanId, viewpointId, heading) in enumerate(zip(scanIds, viewpointIds, headings)):
            self.sims[i].newEpisode(scanId, viewpointId, heading, 0) # simulator.newEp object for each item in batch

    def getStates(self):
        feature_states = []
        for sim in self.sims:
            state = sim.getState()
            long_id = self._make_id(state.scanId, state.location.viewpointId)
            if self.features:
                feature = self.features[long_id][state.viewIndex,:]
                feature_states.append((feature, state))
            else:
                feature_states.append((None, state))
        return feature_states

    def makeActions(self, actions):
        for i, (index, heading, elevation) in enumerate(actions):
            self.sims[i].makeAction(index, heading, elevation)


class VNLABatch():

    def __init__(self, hparams, split=None, tokenizer=None, from_train_env=None,
                 traj_len_estimates=None):
        self.env = EnvBatch(
            from_train_env=from_train_env.env if from_train_env is not None else None,
            img_features=hparams.img_features, batch_size=hparams.batch_size)

        self.random = random
        self.random.seed(hparams.seed)

        self.tokenizer = tokenizer
        self.split = split
        self.batch_size = hparams.batch_size
        self.max_episode_length = hparams.max_episode_length
        self.n_subgoal_steps = hparams.n_subgoal_steps

        self.traj_len_estimates = defaultdict(list) # initialization for a dict which has lists for value

        self.query_ratio = hparams.query_ratio

        self.no_room = hasattr(hparams, 'no_room') and hparams.no_room

        if self.split is not None:
            self.load_data(load_datasets([split], hparams.data_path, # load_datasets from util. See asknav_train.json for instance.
                prefix='noroom' if self.no_room else 'asknav'))

        # Estimate time budget using the upper 95% confidence bound
        if traj_len_estimates is None:
            for k in self.traj_len_estimates:
                self.traj_len_estimates[k] = min(self.max_episode_length,
                    float(np.average(self.traj_len_estimates[k]) +
                    1.95 * scipy.stats.sem(self.traj_len_estimates[k])))
                assert not math.isnan(self.traj_len_estimates[k])
        else:
            for k in self.traj_len_estimates:
                if k in traj_len_estimates:
                    self.traj_len_estimates[k] = traj_len_estimates[k]
                else:
                    self.traj_len_estimates[k] = self.max_episode_length

    def make_traj_estimate_key(self, item):
        if self.no_room:
            key = (item['start_region_name'], item['object_name'])
        else:
            key = (item['start_region_name'], item['end_region_name'])
        return key

    def encode(self, instr):
        if self.tokenizer is None:
            sys.exit('No tokenizer!')
        return self.tokenizer.encode_sentence(instr)

    def load_data(self, data): # called when VNLABatch was initialized. Data are json objects.
        self.data = []
        # training data item objects have keys: distances, end_region_name, end_regions, heading, instructions,
            # object_indices, object_name, path_id, paths, scan, start_region, start_region_name, trajectories
        
        self.scans = set()
        for item in data:
            self.scans.add(item['scan'])
            key = self.make_traj_estimate_key(item) # [start_region_name, end_region_name]
            self.traj_len_estimates[key].extend( # updates traj_len_estimates of VNLABatch object
                len(t) for t in item['trajectories']) # len(t) = 3

            for j,instr in enumerate(item['instructions']):
                new_item = dict(item)
                del new_item['instructions']
                new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                new_item['instruction'] = instr
                self.data.append(new_item)

        self.reset_epoch()

        if self.split is not None:
            print('VNLABatch loaded with %d instructions, using split: %s' % (
                len(self.data), self.split))

    def _next_minibatch(self): # for generating data and keeping track of what to add into batch after iters
        if self.ix == 0:
            self.random.shuffle(self.data)
        batch = self.data[self.ix:self.ix+self.batch_size]
        # self.data from load_data, each element of batch corresponds to an object in the json file
        if len(batch) < self.batch_size:
            self.random.shuffle(self.data)
            self.ix = self.batch_size - len(batch)
            batch += self.data[:self.ix]
        else:
            self.ix += self.batch_size
        self.batch = batch

    def set_data_and_scans(self, data, scans):
        self.data = data
        self.scans = scans

    def reset_epoch(self):
        self.ix = 0

    def _get_obs(self):
        obs = []
        for i, (feature, state) in enumerate(self.env.getStates()):
            item = self.batch[i] # batch is made from data loaded from train.json
            obs.append({
                'instr_id' : item['instr_id'],
                'scan' : state.scanId,
                'point': state.location.point,
                'viewpoint' : state.location.viewpointId,
                'viewIndex' : state.viewIndex,
                'heading' : state.heading,
                'elevation' : state.elevation,
                'feature' : feature,
                'step' : state.step,
                'navigableLocations' : state.navigableLocations,
                'instruction' : self.instructions[i],
                'goal_viewpoints' : [path[-1] for path in item['paths']],
                'init_viewpoint' : item['paths'][0][0]
            })
            # from reset
            obs[-1]['max_queries'] = self.max_queries_constraints[i] # sets attributes for last appended ob
            obs[-1]['traj_len'] = self.traj_lens[i]
            if 'instr_encoding' in item:
                obs[-1]['instr_encoding'] = item['instr_encoding']
                print("item encoding")
                print(item['instr_encoding'])

            # print("last obs / batch instance:")
            # print(obs[-1]['instr_id']) 38024_0
            # print(obs[-1]['scan'])    82sE5b5pLXE
            # print(obs[-1]['point'])   [-23.087, 14.866, 1.4126]
            # print(obs[-1]['viewpoint'])   626241dbd7334a45868a1092a3e195af
            # print(obs[-1]['viewIndex'])   17
            # print(obs[-1]['heading'])      2.61799
            # print(obs[-1]['elevation'])      0.0
            # print(obs[-1]['feature'])     [floats]
            # print(obs[-1]['step'])        0
            # print(obs[-1]['navigableLocations'])  [<MatterSim.ViewPoint object>]
            # print(obs[-1]['instruction'])     "find a picture in one of the offices ."
            # print(obs[-1]['goal_viewpoints'])     ['302e8ac8ecf84...','89eeae92b2f...']
            # print(obs[-1]['init_viewpoint'])  626241dbd7334a45868a1092a3e195af
            # print(obs[-1]['max_queries'])     9
            # print(obs[-1]['traj_len'])    17
            # input()

        return obs

    # This is where we need to work on?
    def _calculate_max_queries(self, traj_len):
        ''' Sample a help-requesting budget given a time budget. '''

        max_queries = self.query_ratio * traj_len / self.n_subgoal_steps
        int_max_queries = int(max_queries)
        frac_max_queries = max_queries - int_max_queries
        return int_max_queries + (self.random.random() < frac_max_queries)

    def reset(self, is_eval):
        ''' Load a new minibatch / episodes. '''

        self._next_minibatch()

        scanIds = [item['scan'] for item in self.batch] # refer to asknav_train.json for item in batch
        viewpointIds = [item['paths'][0][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        self.instructions = [item['instruction'] for item in self.batch] # eg. find an armchair in the living rm
        self.env.newEpisodes(scanIds, viewpointIds, headings)

        self.max_queries_constraints = [None] * self.batch_size # batch_size given in verbal_hard.json
        self.traj_lens = [None] * self.batch_size

        for i, item in enumerate(self.batch):
            # Assign time budget
            if is_eval:
                # If eval use expected trajectory length between start_region and end_region
                key = self.make_traj_estimate_key(item)
                traj_len_estimate = self.traj_len_estimates[key]
            else:
                # If train use average oracle trajectory length
                traj_len_estimate = sum(len(t)
                    for t in item['trajectories']) / len(item['trajectories'])
            self.traj_lens[i] = min(self.max_episode_length, int(round(traj_len_estimate)))

            # Assign help-requesting budget
            self.max_queries_constraints[i] = self._calculate_max_queries(self.traj_lens[i])
            assert not math.isnan(self.max_queries_constraints[i])

        return self._get_obs()

    def step(self, actions):
        self.env.makeActions(actions)
        return self._get_obs()

    def modify_instruction(self, idx, instr, type): # i, verbal_subgoal, edit type
        ''' Modify end-goal. '''
        if type == 'prepend':
            self.instructions[idx] = instr + self.batch[idx]['instruction'] # self.instructions init during reset()
        elif type == 'append':
            self.instructions[idx] = self.batch[idx]['instruction'] + instr
        elif type == 'replace':
            self.instructions[idx] = instr

    def get_obs(self):
        return self._get_obs()



