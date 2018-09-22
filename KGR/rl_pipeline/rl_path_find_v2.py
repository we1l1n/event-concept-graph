import os
import sys
import time
import json
import codecs
import random
from copy import copy
from queue import Queue
from collections import namedtuple, Counter, defaultdict
from itertools import count
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
from gensim.similarities.index import AnnoyIndexer
####################################################################################################################
# config
state_dim = 200
# action_space = 70 # 35*2
eps_start = 1
eps_end = 0.1
epe_decay = 1000
replay_memory_size = 10000
batch_size = 128
embedding_dim = 100
gamma = 0.99
target_update_freq = 1000

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))


def l2_distance(e1, e2):
    return np.sqrt(np.sum(np.square(e1 - e2)))


def compare(v1, v2):
    return sum(v1 == v2)


def prob_norm(probs):
    return probs/sum(probs)

####################################################################################################################
# KB


class KB(object):
    def __init__(self):
        self.entities = {}  # {'实体id':[Path.{relation,entity2},]}

    def addRelation(self, entity1, relation, entity2, weight):
        # add direct connections
        if entity1 in self.entities.keys():
            self.entities[entity1].append(Path(relation, entity2, weight))
        else:
            # entities{entity1：Path{.relation.connected_entity}}
            self.entities[entity1] = [Path(relation, entity2, weight)]

    def getPathsFrom(self, entity):
        return self.entities[entity]

    def removePath(self, entity1, entity2):
        # remove direct connection between e1 and e2
        for idx, path in enumerate(self.entities[entity1]):
            if(path.connected_entity == entity2):
                del self.entities[entity1][idx]
                break
        for idx, path in enumerate(self.entities[entity2]):
            if(path.connected_entity == entity1):
                del self.entities[entity2][idx]
                break

    # by = symbol/embedding
    def pickRandomIntermediatesBetween(self, entity1, entity2, num, by='symbol'):
        # TO DO: COULD BE IMPROVED BY NARROWING THE RANGE OF
        # RANDOM EACH TIME ITERATIVELY CHOOSE AN INTERMEDIATE
        if num > len(self.entities) - 2:
            raise ValueError('Number of Intermediates picked is larger than possible',
                             'num_entities: {}'.format(len(self.entities)),
                             'num_itermediates: {}'.format(num))
        # non-return samples
        if by == 'symbol':
            return random.sample(set(self.entities.keys())-set([entity1, entity2]), num)
        else:
            similar_words = [w for w,p in wv_ent.most_similar(positive=[entity1,entity2], topn=100, indexer=annoy_index_ent)]
            return random.sample(similar_words,num)

    def __str__(self):
        return ''.join([entity+','.join([str(path) for path in self.entities[entity]])
                        for entity in self.entities])


class Path(object):
    def __init__(self, relation, connected_entity, weight):
        self.relation = relation
        self.connected_entity = connected_entity
        self.weight = weight

    def __str__(self):
        return "rel:{},next_entity:{}".format(self.relation, self.connected_entity)

    __repr__ = __str__
####################################################################################################################
# BFS


class foundPaths(object):
    def __init__(self, kb_status):
        # {entity:status}
        # status:(isFound,prevNode,relation)
        self.entities = kb_status

    def isFound(self, entity):
        return self.entities[entity][0]

    def markFound(self, entity, prevNode, relation, weight):
        self.entities[entity] = (True, prevNode, relation, weight)

    def reconstructPath(self, entity1, entity2):  # after BFS
        curNode, entity_list, path_list,weight_list = entity2, [entity2], [], []  # from tail
        while(curNode != entity1):       # status:(isFound,prevNode, relation)
            path_list.append(self.entities[curNode][2])  # relation
            weight_list.append(self.entities[curNode][3])
            curNode = self.entities[curNode][1]         # prevNode
            entity_list.append(curNode)
            if len(entity_list)!=len(set(entity_list)):return [],[]
        return entity_list[::-1], path_list[::-1], weight_list[::-1]

    def __str__(self):
        return ''.join([entity + "[{},{},{}]".format(status[0], status[1], status[2])
                        for entity, status in self.entities.iteritems()])


def BFS(kb, entity1, entity2, num_paths=1,max_steps=5):
    '''
    input: kb=KB(),head,tail,num_paths:bfs-non-fitst-path
    output: (True, entity_list, path_list)
    '''
    path_finder = foundPaths(
        copy(kb_status)); path_finder.markFound(entity1, None, None, None)
    q = Queue(); q.put(entity1)
    entity_lists, path_lists = [], []
    step = 0
    print('BFS:{}->{}'.format(entity1,entity2))
    while(not q.empty() and step<=max_steps):
        step += 1
        # q.get():single-mode
        # level-mode for level control 
        curNodes = [q.get() for i in range(q.qsize())]
        # forbiden large fan-out
        if len(curNodes)>100000:break
        # leval-random-mode  
        random.shuffle(curNodes)
        print('step:{},len(curNodes):{}'.format(step,len(curNodes)))
        for curNode in curNodes:
            for path in kb.getPathsFrom(curNode):  # get connections
                connectRelation, nextEntity, weight = path.relation, path.connected_entity, path.weight
                if(not path_finder.isFound(nextEntity)):  # put for continue search
                    q.put(nextEntity)
                    path_finder.markFound(nextEntity, curNode, connectRelation, weight)
                if(nextEntity == entity2):  # arrive tail
                    path_finder.markFound(nextEntity, curNode, connectRelation, weight)
                    entity_list, path_list, weight_list = path_finder.reconstructPath(
                        entity1, entity2)
                    if num_paths == 1: return (1, entity_list, path_list, weight_list)
                    else:
                        if len(entity_lists) == num_paths:
                            return (num_paths, entity_lists, path_lists, weight_list)
                        else:
                            entity_lists.append(entity_list)
                            path_lists.append(path_list)
                            weight_lists.append(weight_list)

    return (len(entity_lists), entity_lists, path_lists, weight_lists)
####################################################################################################################
# ENV


class Env(object):
    """knowledge graph environment definition"""

    def __init__(self, data_path, relation='/r/Causes'):
        self.entity2id_ = dict([(line.split()[0], int(line.split()[1])) for line in
                  codecs.open(data_path+'entity2id.txt', 'r', encoding='utf-8') if len(line.split()) == 2])
        print('len(self.entity2id_):',len(self.entity2id_))
        self.relation2id_ = dict([(line.split()[0], int(line.split()[1])) for line in
                    codecs.open(data_path+'relation2id.txt', 'r', encoding='utf-8') if len(line.split()) == 2])
        print('len(self.relation2id_):',len(self.relation2id_))
        print('self.relation2id_:',self.relation2id_)
        self.relations = list(self.relation2id_.keys())
        # global action_space;action_space = len(self.relations);print('global action_space:',action_space)
        embeddings = json.load(open(data_path+'E.vec.json', 'r'))
        self.entity2vec = np.array(embeddings['ent_embeddings'])
        self.relation2vec = np.array(embeddings['rel_embeddings'])

        # check
        assert self.entity2vec.shape[0] == len(self.entity2id_);assert self.entity2vec.shape[1] == 100
        assert self.relation2vec.shape[0] == len(self.relation2id_);assert self.relation2vec.shape[1] == 100

        self.weight = []
        self.path = []
        self.path_relations = []

        # kb_env_rl filter:rel
        # self.kb = [line.rsplit() for line in codecs.open(data_path+'kb_env_rl.txt','r',encoding='utf-8')\
        # if line.split()[2] != relation and line.split()[2] != relation+'_inv']
        # print('len(kb):',len(self.kb))

        #import pdb;pdb.set_trace()

        # reconstruct rel2id
        '''
        relation2id_ = []
        relation2vec = []
        for rel in zip(self.relation2id_.items(),self.relation2vec):
            if rel[0][0] in evidence_rel:
                relation2id_.append(rel[0][0])
                relation2vec.append(rel[1])
        self.relation2id_ = dict([(relation2id_[i],i) for i in range(len(relation2id_))])
        print('self.relation2id_:',self.relation2id_)
        self.relations = list(self.relation2id_.keys())
        self.relation2vec = np.array(relation2vec)
        '''


        # valid actions
        self.valid_actions_ = defaultdict(dict)
        [self.valid_actions_[self.entity2id_[line.split()[0]]].update(\
                            {line.split()[1]:(self.relation2id_[line.split()[2]],float(line.split()[3]))})
        for line in codecs.open(data_path+'kb_env_rl.txt', 'r', encoding='utf-8')
        #if len(line.split()) == 3 and line.split()[2] != relation and line.split()[2] != relation+'_inv']  # and line.split()[0] in self.entity2id_.keys()]
        if len(line.split()) == 3 and line.split()[2] in evidence_rel]

        self.die = 0  # record how many times does the agent choose an invalid path

    def interact(self, state, action):
        '''
        This function process the interact from the agent
        state: is [current_position, target_position]
        action: an integer
        return: (reward, [new_postion, target_position], done)
        '''
        done = 0  # Whether the episode has finished
        curr_pos, target_pos = state[:-1]
        chosed_relation = self.relations[action]
        print(f'action:{chosed_relation}')
        print(f'chosed_action_id:{action}')
        # self.get_valid_actions(curr_pos)
        valid_actions = self.valid_actions_[curr_pos]
        print(f'valid_actions:{valid_actions}')
        valid_actions_id = set(valid_actions.values())
        print(f'valid_actions_id:{valid_actions_id}')
        choices = [(entity,rel[1]) for entity, rel in valid_actions.items()
                                                                    if action == rel_id]
        if len(choices) == 0:
            reward = -1
            self.die += 1
            next_state = state  # stay in the initial state
            next_state[-1] = self.die
            return (reward, next_state, done)
        else:  # find a valid step
            next_pos,weight = random.choice(choices)
            self.weight.append(weight)
            self.path.append(chosed_relation + ' -> ' + next_pos)
            self.path_relations.append(chosed_relation)
            print('Find a valid step:', next_pos, 'Action index:', action)
            self.die = 0
            reward = 0
            next_pos = self.entity2id_[next_pos]
            next_state = [next_pos, target_pos, self.die]

            if next_pos == target_pos:
                print('Find a path:', self.path)
                done = 1
                reward = 0
                next_state = None
            return (reward, next_state, done)

    def idx_state(self, idx_list):
        if idx_list != None:
            curr = self.entity2vec[idx_list[0], :]
            targ = self.entity2vec[idx_list[1], :]
            return np.expand_dims(np.concatenate((curr, targ - curr)), axis=0)
        else:
            return None

    # def get_valid_actions(self, entityID): # valid action space <= action space
        # actions = dict([(triple[1],self.relation2id_[triple[2]]) for triple in self.kb if entityID == self.entity2id_[triple[0]]])
        # return actions

    def path_embedding(self, path):
        embeddings = [self.relation2vec[self.relation2id_[relation], :]
            for relation in path]
        embeddings = np.reshape(embeddings, (-1, embedding_dim))
        path_encoding = np.sum(embeddings, axis=0)
        return np.reshape(path_encoding, (-1, embedding_dim))
####################################################################################################################
# TEACHER

def quality_reward(weight):
    if isinstance(weight,list):
        return sum([1/(-np.log2(p)) for p in weight])/len(weight)
    else:
        return 1/(-np.log2(weight))

def bfs_teacher(e1, e2, num_paths, env, kb):
    # BFS path collect
    # suc1, entity_lists1, path_lists1 = BFS(kb, e1, e2,num_paths=num_paths//2);print(f'{suc1}:BFS forward done')
    # suc2, entity_lists2, path_lists2 = BFS(kb, e2, e1,num_paths=num_paths//2);print(f'{suc2}:BFS backward done')
    # entity_lists = entity_lists1+entity_lists2
    # path_lists = path_lists1+path_lists2
    suc, entity_lists, path_lists, weight_lists = BFS(
        kb, e1, e2, num_paths=num_paths); print(f'{suc}:RAW BFS done')
    mean_step_bfs = (count_bfs*mean_step_bfs + len(path_lists))/(count_bfs+1)
    count_bfs += 1

    #print('len(entities):', len(entity_lists)); print(entity_lists)
    #print('len(paths):', len(path_lists)); print(path_lists)
    path_strs = [''.join([e+'->'+p+'->' for e, p in zip(entity_list[:-1], path_list)]) +
                         entity_list[-1] for entity_list, path_list in zip(entity_lists, path_lists)]
    print('\n'.join(sorted(path_strs)))
    # episodes
    print('collect episodes')
    good_episodes=[]
    targetID=env.entity2id_[e2]
    for path in zip(entity_lists, path_lists, weight_lists):
        good_episode=[]
        for i in range(len(path[0]) - 1):
            currID=env.entity2id_[
                path[0][i]]; nextID=env.entity2id_[path[0][i+1]]
            state_curr=[currID, targetID, 0]; state_next=[nextID, targetID, 0]
            actionID=env.relation2id_[path[1][i]]
            good_episode.append(Transition(state=env.idx_state(state_curr),\
                                           action=actionID, \
                                           next_state=env.idx_state(state_next), \
                                           reward=quality_reward(path[2][i])))
                                           #reward=1))  # each time step reward==1
        good_episodes.append(good_episode)
    return good_episodes

def bibfs_teacher(e1, e2, num_paths, env, kb, by='embed'):
    # Bi-BFS path collect
    intermediates=kb.pickRandomIntermediatesBetween(
        e1, e2, num_paths, by=by) # by='symbol'
    print('intermediates:', intermediates)
    entity_lists=[]; path_lists=[]; weight_lists = []
    for i in range(num_paths):
        suc1, entity_list1, path_list1, weight_list1=BFS(
            kb, e1, intermediates[i])#; print(f'{i}:BFS left done')
        suc2, entity_list2, path_list2, weight_list2=BFS(
            kb, intermediates[i], e2)#; print(f'{i}:BFS right done')
        if suc1 and suc2:
            entity_lists.append(entity_list1 + entity_list2[1:])
            path_lists.append(path_list1 + path_list2)
            weight_lists.append(weight_list1 + weight_list2)
    print('BIBFS found paths:', len(path_lists))
    # clean the path
    # duplicate
    # drop [min:max]
    print('path clean')
    entity_lists_new=[]
    path_lists_new=[]
    weight_lists_new = []
    for entities, relations, weights in zip(entity_lists, path_lists, weight_lists):
        path=[entities[int(i/2)] if i % 2 == 0 else (relations[int(i/2)],weights[int(i/2)])\
                    for i in range(len(entities)+len(relations))]
        entity_stats=Counter(entities).items()
        duplicate_ents=[item for item in entity_stats if item[1] != 1]
        duplicate_ents.sort(key=lambda x: x[1], reverse=True)
        for item in duplicate_ents:
            ent=item[0]
            ent_idx=[i for i, x in enumerate(path) if x == ent]
            if len(ent_idx) != 0:
                min_idx=min(ent_idx)
                max_idx=max(ent_idx)
                if min_idx != max_idx:
                    path=path[:min_idx] + path[max_idx:]
        entities_new=[]
        relations_new=[]
        weights_new = []
        for idx, item in enumerate(path):
            if idx % 2 == 0:
                entities_new.append(item)
            else:
                relations_new.append(item[0])
                weights_new.append(item[1])
        entity_lists_new.append(entities_new)
        path_lists_new.append(relations_new)
        weight_lists_new.append(weights_new)
    #print('len(entities):', len(entity_lists_new),
    #      'len(paths):', len(path_lists_new))
    if by == 'embed':
        mean_step_bibfs_vec = (count_bibfs_vec*mean_step_bibfs_vec + len(path_lists_new))/(count_bibfs_vec+1)
        count_bibfs_vec += 1
    else:
        mean_step_bibfs_random = (count_bibfs_random*mean_step_bibfs_random + len(path_lists_new))/(count_bibfs_random+1)
        count_bibfs_random += 1
    path_strs = [''.join([e+'->'+p+'->' for e, p in zip(entity_list[:-1], path_list)]) +
                         entity_list[-1] for entity_list, path_list in zip(entity_lists_new, path_lists_new)]
    print('\n'.join(sorted(path_strs)))
    # episodes
    print('collect episodes')
    good_episodes=[]
    targetID=env.entity2id_[e2]
    for path in zip(entity_lists_new, path_lists_new, weight_lists_new):
        good_episode=[]
        for i in range(len(path[0]) - 1):
            currID=env.entity2id_[
                path[0][i]]; nextID=env.entity2id_[path[0][i+1]]
            state_curr=[currID, targetID, 0]; state_next=[nextID, targetID, 0]
            actionID=env.relation2id_[path[1][i]]
            good_episode.append(Transition(state=env.idx_state(state_curr),\
                                           action=actionID, \
                                           next_state=env.idx_state(state_next), \
                                           reward=quality_reward(path[2][i])))
                                           #reward=1))  # each time step reward==1
        good_episodes.append(good_episode)
    return good_episodes

def path_clean(path):
    rel_ents=path.split(' -> ')
    relations=[]
    entities=[]
    for idx, item in enumerate(rel_ents):
        if idx % 2 == 0:
            relations.append(item)
        else:
            entities.append(item)
    entity_stats=Counter(entities).items()
    duplicate_ents=[item for item in entity_stats if item[1] != 1]
    duplicate_ents.sort(key=lambda x: x[1], reverse=True)
    for item in duplicate_ents:
        ent=item[0]
        ent_idx=[i for i, x in enumerate(rel_ents) if x == ent]
        if len(ent_idx) != 0:
            min_idx=min(ent_idx)
            max_idx=max(ent_idx)
            if min_idx != max_idx:
                rel_ents=rel_ents[:min_idx] + rel_ents[max_idx:]
    return ' -> '.join(rel_ents)
####################################################################################################################
# NETWORK
def policy_nn(state, state_dim, action_dim, initializer):
    """
    策略网络(P(a_t|s_t;theta))
    state_dim -relu-> 512 -relu-> 1024 -softmax-> action_dim
    state -> action_prob [action_dim]
    action_dim == 关系数量
    """
    w1=tf.get_variable('W1', [state_dim, 512], initializer=initializer,
                       regularizer=tf.contrib.layers.l2_regularizer(0.01))
    b1=tf.get_variable('b1', [512], initializer=tf.constant_initializer(0.0))
    h1=tf.nn.relu(tf.matmul(state, w1) + b1)
    w2=tf.get_variable('w2', [512, 1024], initializer=initializer,
                       regularizer=tf.contrib.layers.l2_regularizer(0.01))
    b2=tf.get_variable('b2', [1024], initializer=tf.constant_initializer(0.0))
    h2=tf.nn.relu(tf.matmul(h1, w2) + b2)
    w3=tf.get_variable('w3', [1024, action_dim], initializer=initializer,
                       regularizer=tf.contrib.layers.l2_regularizer(0.01))
    b3=tf.get_variable('b3', [action_dim],
                       initializer=tf.constant_initializer(0.0))
    action_prob=tf.nn.softmax(tf.matmul(h2, w3) + b3)
    return action_prob

def value_nn(state, state_dim, initializer):
    """
    state_dim -relu-> 64 -> 1
    state -> value_estimated
    """
    w1=tf.get_variable('w1', [state_dim, 64], initializer=initializer)
    b1=tf.get_variable('b1', [64], initializer=tf.constant_initializer(0.0))
    h1=tf.nn.relu(tf.matmul(state, w1) + b1)
    w2=tf.get_variable('w2', [64, 1], initializer=initializer)
    b2=tf.get_variable('b2', [1], initializer=tf.constant_initializer(0.0))
    value_estimated=tf.matmul(h1, w2) + b2
    return tf.squeeze(value_estimated)

def q_network(state, state_dim, action_space, initializer):
    """
    state_dim -relu-> 128 -relu-> 64 -> action_space
    state -> [w1,b1,w2,b2,w3,b3,action_values]
    """
    w1=tf.get_variable('w1', [state_dim, 128], initializer=initializer)
    b1=tf.get_variable('b1', [128], initializer=tf.constant_initializer(0))
    h1=tf.nn.relu(tf.matmul(state, w1) + b1)
    w2=tf.get_variable('w2', [128, 64], initializer=initializer)
    b2=tf.get_variable('b2', [64], initializer=tf.constant_initializer(0))
    h2=tf.nn.relu(tf.matmul(h1, w2) + b2)
    w3=tf.get_variable('w3', [64, action_space], initializer=initializer)
    b3=tf.get_variable('b3', [action_space],
                       initializer=tf.constant_initializer(0))
    action_values=tf.matmul(h2, w3) + b3
    return [w1, b1, w2, b2, w3, b3, action_values]
####################################################################################################################
# SL
class SupervisedPolicy(object):
    def __init__(self, learning_rate=0.001):
        self.initializer=tf.contrib.layers.xavier_initializer()
        with tf.variable_scope('supervised_policy'):
            self.state=tf.placeholder(
                tf.float32, [None, state_dim], name='state')
            self.action=tf.placeholder(tf.int32, [None], name='action')
            self.action_prob=policy_nn(
                self.state, state_dim, action_space, self.initializer)
            action_mask=tf.cast(tf.one_hot(
                self.action, depth=action_space), tf.bool)
            self.picked_action_prob=tf.boolean_mask(
                self.action_prob, action_mask)

            self.loss=tf.reduce_sum(-tf.log(self.picked_action_prob)) + \
                                    sum(tf.get_collection(
                                        tf.GraphKeys.REGULARIZATION_LOSSES, scope='supervised_policy'))
            self.optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op=self.optimizer.minimize(self.loss)

    def predict(self, state, sess=None):
        sess=sess or tf.get_default_session()
        return sess.run(self.action_prob, {self.state: state})

    def update(self, state, action, sess=None):
        sess=sess or tf.get_default_session()
        _, loss=sess.run([self.train_op, self.loss], {
                         self.state: state, self.action: action})
        print('loss:', loss)  # /state.shape[0])
        return loss

def sl_train(episodes=500):
    tf.reset_default_graph()
    policy_nn=SupervisedPolicy()
    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for episode in range(len(train_pairs) if len(train_pairs) < episodes else episodes):
            print("Episode %d" % episode); print(
                'Training Sample:', train_pairs[episode % episodes][:-1])
            sample=train_pairs[episode % episodes].split()
            try:
                # good_episodes from teacher
                if sample[0] != sample[1]:
                    bfs_episodes=bfs_teacher(
                    sample[0], sample[1], num_paths, env, kb)
                    print('len(bfs_episodes):',len(bfs_episodes))
                    bibfs_episodes=bibfs_teacher(
                    sample[0], sample[1], num_paths, env, kb)
                    print('len(bibfs_episodes):',len(bibfs_episodes))
                    bibfs_random_episodes=bibfs_teacher(
                    sample[0], sample[1], num_paths, env, kb, by='symbol')
                    print('len(bibfs_episodes):',len(bibfs_random_episodes))
                    good_episodes = bfs_episodes+bibfs_episodes+bibfs_random_episodes
            except Exception as e:
                print('Cannot find a path');good_episodes=[] ;continue
            for item in good_episodes:  # one episode one supervised batch*<state,action> to update theta
                state_batch, action_batch=[], []
                for t, transition in enumerate(item):
                    state_batch.append(transition.state)
                    action_batch.append(transition.action)
                state_batch=np.squeeze(state_batch)
                state_batch=np.reshape(state_batch, [-1, state_dim])
                policy_nn.update(state_batch, action_batch)
        saver.save(sess, sl_model_path)
        print('Model saved')


def sl_test(episodes=300):
    tf.reset_default_graph()
    policy_nn=SupervisedPolicy()
    print('len(test_pairs):', len(test_pairs), 'test_episodes:', episodes)
    success=0
    saver=tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, sl_model_path); print('Model reloaded')
        for episode in range(episodes):
            try: print('Test sample %d: %s' % (episode, test_pairs[episode][:-1]))
            except: continue
            sample=test_pairs[episode].split()
            # reset env path
            env.weight, env.path, env.path_relations = [], [], []
            state_idx=[env.entity2id_[sample[0]], env.entity2id_[sample[1]], 0]
            for t in count():
                state_vec=env.idx_state(state_idx)
                action_probs=policy_nn.predict(state_vec)
                action_chosen=np.random.choice(
                    np.arange(action_space), p=np.squeeze(action_probs))
                reward, next_state, done=env.interact(state_idx, action_chosen)
                if done or t == max_steps_test:
                    if done:
                        print('Success')
                        mean_step_sl = (count_sl*mean_step_sl + len(env.path_relations))/(count_sl+1)
                        count_sl += 1
                        success += 1
                    print(f'success:{success},Episode ends')
                    break
                state_idx=next_state
            print('Success persentage:', success/episodes)

    print('Success persentage:', success/episodes); f_logs.write(
        'sl_test-Success persentage:'+str(success/episodes)+'\n')
####################################################################################################################
# RL
class PolicyNetwork(object):
    def __init__(self, scope='policy_network', learning_rate=0.001):
        self.initializer=tf.contrib.layers.xavier_initializer()
        with tf.variable_scope(scope):
            self.state=tf.placeholder(
                tf.float32, [None, state_dim], name='state')
            self.action=tf.placeholder(tf.int32, [None], name='action')
            # +target
            self.target=tf.placeholder(tf.float32, name='target')
            self.action_prob=policy_nn(
                self.state, state_dim, action_space, self.initializer)

            action_mask=tf.cast(tf.one_hot(
                self.action, depth=action_space), tf.bool)
            self.picked_action_prob=tf.boolean_mask(
                self.action_prob, action_mask)
            # +target
            self.loss=tf.reduce_sum(-tf.log(self.picked_action_prob)*self.target) + \
                                    sum(tf.get_collection(
                                        tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope))
            self.optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op=self.optimizer.minimize(self.loss)

    def predict(self, state, sess=None):
        sess=sess or tf.get_default_session()
        return sess.run(self.action_prob, {self.state: state})

    def update(self, state, target, action, sess=None):
        sess=sess or tf.get_default_session()
        # +target
        feed_dict={self.state: state, self.target: target, self.action: action}
        _, loss=sess.run([self.train_op, self.loss], feed_dict)
        print('loss:', loss)  # /state.shape[0])
        return loss


def REINFORCE(train_pairs, policy_nn, num_episodes):
    success=0
    path_found=[]
    for i_episode in range(num_episodes):
        start=time.time()
        print('Episode %d' % i_episode); print(
            'Training sample: ', train_pairs[i_episode][:-1])
        sample=train_pairs[i_episode].split()
        # reset env path
        env.weight, env.path, env.path_relations = [], [], []
        state_idx=[env.entity2id_[sample[0]], env.entity2id_[sample[1]], 0]
        episode, state_batch_negative, action_batch_negative=[], [], []
        for t in count():
            state_vec=env.idx_state(state_idx)
            action_probs=policy_nn.predict(state_vec)
            action_chosen=np.random.choice(
                np.arange(action_space), p=np.squeeze(action_probs))
            reward, new_state_idx, done=env.interact(state_idx, action_chosen)
            # the action fails for this step
            if reward == -1:
                state_batch_negative.append(state_vec)
                action_batch_negative.append(action_chosen)
            new_state_vec=env.idx_state(new_state_idx)
            episode.append(Transition(state=state_vec,\
                                      action=action_chosen,\
                                      next_state=new_state_vec,\
                                      reward=reward))
            if done or t == max_steps: break
            state_idx=new_state_idx
        # Discourage the agent when it choose an invalid step
        if len(state_batch_negative) != 0:
            print('Penalty to invalid steps:', len(state_batch_negative))
            policy_nn.update(np.reshape(state_batch_negative,
                             (-1, state_dim)), -0.05, action_batch_negative)

        # If the agent success, do one optimization
        def update_episode(policy_nn, episode, total_reward):
            state_batch=[]
            action_batch=[]
            for t, transition in enumerate(episode):
                if transition.reward == 0:
                    state_batch.append(transition.state)
                    action_batch.append(transition.action)
            policy_nn.update(np.reshape(
                state_batch, (-1, state_dim)), total_reward, action_batch)

        if done == 1:
            print('Success')
            path_found.append(path_clean(' -> '.join(env.path)))
            success += 1
            length_reward, global_reward = 1/len(env.path), 1
            #total_reward=0.1*global_reward + 0.9*length_reward
            total_reward= 0.3*global_reward + 0.3*length_reward + 0.4*quality_reward(env.weight)
            update_episode(policy_nn, episode, total_reward)
            print('total_reward success')
        else:
            global_reward=-0.5 #0.05->0.5
            update_episode(policy_nn, episode, global_reward)
            print('Failed, Do one teacher guideline')
            try:
                bfs_episodes=bfs_teacher(
                    sample[0], sample[1], teacher_num_paths, env, kb)
                print('len(bfs_episodes):',len(bfs_episodes))
                bibfs_episodes=bibfs_teacher(
                    sample[0], sample[1], teacher_num_paths, env, kb)
                print('len(bibfs_episodes):',len(bibfs_episodes))
                good_episodes = bfs_episodes+bibfs_episodes
                [update_episode(policy_nn, episode, 1)
                                for episode in good_episodes]
                print('Teacher guideline success')
            except Exception as e:
                print('Teacher guideline failed')
        print('Episode time: ', time.time() - start)
        print('Success:', success)
        print('Success percentage:', success/num_episodes)
    print('Success percentage:', success/num_episodes); f_logs.write(
        'rl_train-Success persentage:'+str(success/num_episodes)+'\n')
    # store path stats
    path_found_relation=[' -> '.join([rel for ix, rel in enumerate(path.split(' -> ')) if ix % 2 == 0]) \
                                                                                 for path in path_found]
    relation_path_stats=sorted(
        Counter(path_found_relation).items(), key=lambda x: x[1], reverse=True)
    f_logs.write('path_stats:\n')
    [f_logs.write(item[0]+'\t'+str(item[1])+'\n')
                  for item in relation_path_stats]
    print('Path stats saved')

def rl_retrain(episodes=300):
    print('Start retraining'); tf.reset_default_graph()
    # restore form parameters of supervised_policy
    policy_network=PolicyNetwork(scope='supervised_policy')
    saver=tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, sl_model_path); print("sl_policy restored")
        REINFORCE(train_pairs, policy_network, len(train_pairs)
                  if len(train_pairs) < episodes else episodes)
        saver.save(sess, rl_model_path)
    print('Retrained model saved')

def rl_test(episodes=500):
    tf.reset_default_graph()
    # restore form parameters of supervised_policy
    policy_network=PolicyNetwork(scope='supervised_policy')
    success=0
    saver=tf.train.Saver()
    path_found=[]
    path_set=set()

    with tf.Session() as sess:
        saver.restore(sess, rl_model_path); print('Model reloaded')
        for episode in range(len(test_pairs) if len(test_pairs) < episodes else episodes):
            print('Test sample %d: %s' % (episode, test_pairs[episode][:-1]))
            sample=test_pairs[episode].split()
            # reset env path
            env.weight, env.path, env.path_relations = [], [], []
            state_idx=[env.entity2id_[sample[0]], env.entity2id_[sample[1]], 0]
            transitions=[]
            for t in count():
                state_vec=env.idx_state(state_idx)
                action_probs=np.squeeze(policy_network.predict(state_vec))
                action_chosen=np.random.choice(
                    np.arange(action_space), p=action_probs)
                reward, new_state_idx, done=env.interact(
                    state_idx, action_chosen)
                new_state_vec=env.idx_state(new_state_idx)
                transitions.append(Transition(state=state_vec,\
                                              action=action_chosen,\
                                              next_state=new_state_vec,\
                                              reward=reward))
                if done or t == max_steps_test:
                    if done:
                        success += 1; print("Success")
                        mean_step_rl = (count_rl*mean_step_rl + len(env.path_relations))/(count_rl+1)
                        count_rl += 1
                        path_found.append(path_clean(' -> '.join(env.path)))
                    else:
                        print('Episode ends due to step limit')
                    break
                state_idx=new_state_idx
            if done:
                if len(path_set) != 0:
                    path_found_embedding=[env.path_embedding(
                        path.split(' -> ')) for path in path_set]
                    curr_path_embedding=env.path_embedding(env.path_relations)
                    path_found_embedding=np.reshape(
                        path_found_embedding, (-1, embedding_dim))
                    cos_sim=cosine_similarity(
                        path_found_embedding, curr_path_embedding)
                    diverse_reward=-np.mean(cos_sim)
                    print('diverse_reward', diverse_reward)
                    # total_reward = 0.1*global_reward + 0.8*length_reward + 0.1*diverse_reward
                    state_batch=[]
                    action_batch=[]
                    for t, transition in enumerate(transitions):
                        if transition.reward == 0:
                            state_batch.append(transition.state)
                            action_batch.append(transition.action)
                    policy_network.update(np.reshape(
                        state_batch, (-1, state_dim)), 0.1*diverse_reward, action_batch)
                path_set.add(' -> '.join(env.path_relations))
            print('Success:', success)
            print('Success persentage:', success/episodes)
    print('Success persentage:', success/episodes); f_logs.write(
        'rl_test-Success persentage:'+str(success/episodes)+'\n')
    # env path reset
    env.path, env.path_relations=[], []
    # store path to use
    path_found_relation=[' -> '.join([rel for ix, rel in enumerate(path.split(' -> ')) if ix % 2 == 0]) \
                                                                                 for path in path_found]
    relation_path_stats=sorted(
        Counter(path_found_relation).items(), key=lambda x: x[1], reverse=True)
    ranking_path=sorted([(path_stat[0], len(path_stat[0].split(' -> '))) \
                           for path_stat in relation_path_stats],\
                          key=lambda x: x[1])
    f_logs.write('path_to_use:\n')
    [f_logs.write(item[0]+'\n') for item in ranking_path]
    print('path to use saved')
    f_logs.close()
####################################################################################################################
# MAIN
# relation = '/r/tweet/open/cause'
relation='/r/tweet/open/kill'
rel_path=relation.replace('/', '_')
data_path='./'
task_path=data_path + 'tasks/' + rel_path + '/'
print('task_path:{},relation:{}'.format(task_path,relation))
if not os.path.exists(data_path + 'tasks/'): os.mkdir(data_path + 'tasks/')
if not os.path.exists(task_path): os.mkdir(task_path)

sl_model_path=task_path+'policy_supervised_' + rel_path
rl_model_path=task_path+'policy_retrained_' + rel_path
logs_path=task_path+'logs.txt'
f_logs=codecs.open(logs_path, 'a+', encoding='utf-8')

train_pairs=[line for line in codecs.open(
    data_path+rel_path+'_train_pos', 'r', encoding='utf-8')]
test_pairs=train_pairs

# vec similar query
wv_ent = KeyedVectors.load_word2vec_format('entity2vec.bin', binary=True)
annoy_index_ent = AnnoyIndexer()
annoy_index_ent.load('entity2vec.index')
annoy_index_ent.model = wv_ent

wv_rel = KeyedVectors.load_word2vec_format('relation2vec.bin', binary=True)
annoy_index_rel = AnnoyIndexer()
annoy_index_rel.load('relation2vec.index')
annoy_index_rel.model = wv_rel

evidence_num = 1000
evidence_rel = [r for r,p in wv_rel.most_similar(positive=[relation],topn=evidence_num,indexer=annoy_index_rel)]

# evidence control
evidence_rel = [ev_rel for ev_rel in evidence_rel if ev_rel.startswith('/r/tweet/open/')]

evidence_rel.remove(relation);evidence_rel.remove(relation+'_inv')
print('evidence_rel:{0}'.format(evidence_rel))
print('len(evidence_rel):{0}'.format(len(evidence_rel)))

# env kb
env=Env(data_path, relation)
kb=KB()
#[kb.addRelation(line.rsplit()[0], line.rsplit()[2], line.rsplit()[1]) \
#    for line in codecs.open(data_path+'kb_env_rl.txt', 'r', encoding='utf-8')\
#    if len(line.split()) == 3 and line.split()[2] != relation and line.split()[2] != relation+'_inv']
[kb.addRelation(line.rsplit()[0], line.rsplit()[2], line.rsplit()[1], float(line.rsplit()[2])) \
    for line in codecs.open(data_path+'kb_env_rl.txt', 'r', encoding='utf-8')\
    if len(line.split()) == 3 and line.split()[2] in evidence_rel]
kb_status=dict([(entity, (False, '', '', '')) for entity in kb.entities.keys()])

# key parameter
action_space=len(evidence_rel)
num_paths=10
teacher_num_paths=3
max_steps = 50
max_steps_test = 50

# experiments log
mean_step_bfs = 0
count_bfs = 0
mean_step_bibfs_vec = 0
count_bibfs_vec = 0
mean_step_bibfs_random = 0
count_bibfs = 0
mean_step_sl = 0
count_sl = 0
mean_step_rl = 0
count_rl = 0



def training_pipeline(pipeline=4,
                    sl_train_episodes=1000,
                    sl_test_episodes=1000,
                    rl_retrain_episodes=1000,
                    rl_test_episodes=1000):
    '''
    python rl_path_find.py 4 1000 1000 1000 1000
    '''
    # experiment settings
    f_logs.write('experiment settings:\n')
    f_logs.write('evidence_num:'+str(evidence_num)+'\n')
    f_logs.write('action_space:'+str(action_space)+'\n')
    f_logs.write('num_paths:'+str(num_paths)+'\n')
    f_logs.write('teacher_num_paths:'+str(teacher_num_paths)+'\n')
    f_logs.write('max_steps:'+str(max_steps)+'\n')
    f_logs.write('max_steps_test:'+str(max_steps_test)+'\n')
    # train pipeline
    sl_train(sl_train_episodes) if pipeline > 3 else ''
    sl_test(sl_test_episodes) if pipeline > 2 else ''
    rl_retrain(rl_retrain_episodes) if pipeline > 1 else ''
    rl_test(rl_test_episodes) if pipeline > 0 else ''
    # len(path) distribution
    f_logs('len(path) distribution:\n')
    f_logs.write(str(mean_step_bfs)+'\n')
    f_logs.write(str(mean_step_bibfs_vec)+'\n')
    f_logs.write(str(mean_step_bibfs_random)+'\n')
    f_logs.write(str(mean_step_sl)+'\n')
    f_logs.write(str(mean_step_rl)+'\n')

if __name__ == '__main__':
    import fire
    fire.Fire(training_pipeline)

## 0->1:
# control the evidence