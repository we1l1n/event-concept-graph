import os,sys,time,codecs,random
from copy import copy
from queue import Queue
from collections import namedtuple, Counter,defaultdict
from itertools import count
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
####################################################################################################################
# config
state_dim = 200
action_space = 400
eps_start = 1
eps_end = 0.1
epe_decay = 1000
replay_memory_size = 10000
batch_size = 128
embedding_dim = 100
gamma = 0.99
target_update_freq = 1000
max_steps = 50
max_steps_test = 50
num_paths = 10

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

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
        self.entities = {} # {'实体id':[Path.{relation,entity2},]}

    def addRelation(self, entity1, relation, entity2):
        # add direct connections
        if entity1 in self.entities.keys():
            self.entities[entity1].append(Path(relation, entity2))
        else:
            # entities{entity1：Path{.relation.connected_entity}}
            self.entities[entity1] = [Path(relation, entity2)] 
           

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

    def pickRandomIntermediatesBetween(self, entity1, entity2, num):
        # TO DO: COULD BE IMPROVED BY NARROWING THE RANGE OF
        # RANDOM EACH TIME ITERATIVELY CHOOSE AN INTERMEDIATE  
        if num > len(self.entities) - 2:
            raise ValueError('Number of Intermediates picked is larger than possible',\
                             'num_entities: {}'.format(len(self.entities)), \
                             'num_itermediates: {}'.format(num))
        return random.sample(set(self.entities.keys())-set([entity1,entity2]),num) # non-return samples

    def __str__(self):
        return ''.join([entity+','.join([str(path) for path in self.entities[entity]]) \
                        for entity in self.entities])

class Path(object):
    def __init__(self, relation, connected_entity):
        self.relation = relation
        self.connected_entity = connected_entity

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

    def markFound(self, entity, prevNode, relation):
        self.entities[entity] = (True, prevNode, relation)

    def reconstructPath(self, entity1, entity2): # after BFS
        curNode,entity_list,path_list = entity2,[entity2],[] # from tail
        while(curNode != entity1):       # status:(isFound,prevNode, relation)
            path_list.append(self.entities[curNode][2]) # relation
            curNode = self.entities[curNode][1]         # prevNode
            entity_list.append(curNode)
        return entity_list[::-1],path_list[::-1]

    def __str__(self):
        return ''.join([entity + "[{},{},{}]".format(status[0],status[1],status[2]) \
                        for entity, status in self.entities.iteritems()])

def BFS(kb, entity1, entity2):
    '''
    input: kb=KB(),head,tail
    output: (True, entity_list, path_list)
    '''
    path_finder = foundPaths(copy(kb_status));path_finder.markFound(entity1, None, None)
    q = Queue();q.put(entity1)
    while(not q.empty()):
        curNode = q.get()
        for path in kb.getPathsFrom(curNode): # get connections
            connectRelation,nextEntity = path.relation,path.connected_entity
            if(not path_finder.isFound(nextEntity)): # put for continue search
                q.put(nextEntity)
                path_finder.markFound(nextEntity, curNode, connectRelation)
            if(nextEntity == entity2): # arrive tail
                entity_list, path_list = path_finder.reconstructPath(entity1, entity2)
                return (True, entity_list, path_list)
    return (False, None, None)
####################################################################################################################
# ENV
class Env(object):
    """knowledge graph environment definition"""
    def __init__(self, data_path, relation='concept:worksfor'):
        self.entity2id_ = entity2id = dict([(line.split()[0],int(line.split()[1])) for line in \
                  codecs.open(data_path+'entity2id.txt','r',encoding='utf-8') if len(line.split()) == 2])
        self.relation2id_ = relation2id = dict([(line.split()[0],int(line.split()[1])) for line in \
                    codecs.open(data_path+'relation2id.txt','r',encoding='utf-8') if len(line.split()) == 2])
        self.relations = list(self.relation2id_.keys())
        self.entity2vec = np.loadtxt(data_path + 'entity2vec.bern')
        self.relation2vec = np.loadtxt(data_path + 'relation2vec.bern')

        self.path = []
        self.path_relations = []

        # kb_env_rl filter:rel
        #self.kb = [line.rsplit() for line in codecs.open(data_path+'kb_env_rl.txt','r',encoding='utf-8')\
        #if line.split()[2] != relation and line.split()[2] != relation+'_inv']
        #print('len(kb):',len(self.kb))

        # valid actions
        self.valid_actions_ = defaultdict(dict)
        [self.valid_actions_[self.entity2id_[line.split()[0]]].update({line.split()[1]:self.relation2id_[line.split()[2]]}) for line in codecs.open(data_path+'kb_env_rl.txt','r',encoding='utf-8')\
        if line.split()[2] != relation and line.split()[2] != relation+'_inv']

        self.die = 0 # record how many times does the agent choose an invalid path

    def interact(self, state, action):
        '''
        This function process the interact from the agent
        state: is [current_position, target_position] 
        action: an integer
        return: (reward, [new_postion, target_position], done)
        '''
        done = 0 # Whether the episode has finished
        curr_pos,target_pos = state[:-1]
        chosed_relation = self.relations[action]
        #print(f'action:{chosed_relation}')
        print(f'chosed_action_id:{action}')
        valid_actions = self.valid_actions_[curr_pos]#self.get_valid_actions(curr_pos)
        valid_actions_id = set(valid_actions.values())
        print(f'valid_actions_id:{valid_actions_id}')
        choices = [entity for entity,rel_id in valid_actions.items() if action == rel_id]
        if len(choices) == 0:
            reward = -1
            self.die += 1
            next_state = state # stay in the initial state
            next_state[-1] = self.die
            return (reward, next_state, done)
        else: # find a valid step
            next_pos = random.choice(choices)
            self.path.append(chosed_relation + ' -> ' + next_pos)
            self.path_relations.append(chosed_relation)
            print('Find a valid step:',next_pos,'Action index:',action)
            self.die = 0
            reward = 0
            next_pos = self.entity2id_[next_pos]
            next_state = [next_pos, target_pos, self.die]

            if next_pos == target_pos:
                print('Find a path:',self.path)
                done = 1
                reward = 0
                next_state = None
            return (reward, next_state, done)

    def idx_state(self, idx_list):
        if idx_list != None:
            curr = self.entity2vec[idx_list[0],:]
            targ = self.entity2vec[idx_list[1],:]
            return np.expand_dims(np.concatenate((curr, targ - curr)),axis=0)
        else:
            return None

    #def get_valid_actions(self, entityID): # valid action space <= action space
        #actions = dict([(triple[1],self.relation2id_[triple[2]]) for triple in self.kb if entityID == self.entity2id_[triple[0]]])
        #return actions

    def path_embedding(self, path):
        embeddings = [self.relation2vec[self.relation2id_[relation],:] for relation in path]
        embeddings = np.reshape(embeddings, (-1,embedding_dim))
        path_encoding = np.sum(embeddings, axis=0)
        return np.reshape(path_encoding,(-1, embedding_dim))
####################################################################################################################
# TEACHER
def teacher(e1, e2, num_paths, env, kb):
    # Bi-BFS path collect
    intermediates = kb.pickRandomIntermediatesBetween(e1, e2, num_paths)
    print('intermediates:',intermediates)      
    entity_lists = [];path_lists = []
    for i in range(num_paths):
        suc1, entity_list1, path_list1 = BFS(kb, e1, intermediates[i]);print(f'{i}:BFS left done')
        suc2, entity_list2, path_list2 = BFS(kb, intermediates[i], e2);print(f'{i}:BFS right done')
        if suc1 and suc2:
            entity_lists.append(entity_list1 + entity_list2[1:])
            path_lists.append(path_list1 + path_list2)
    print('BFS found paths:', len(path_lists))
    # clean the path 
    # duplicate
    # drop [min:max]
    print('path clean')
    entity_lists_new = []
    path_lists_new = []
    for entities, relations in zip(entity_lists, path_lists):
        path = [entities[int(i/2)] if i%2 == 0 else relations[int(i/2)]\
                    for i in range(len(entities)+len(relations))]
        entity_stats = Counter(entities).items()
        duplicate_ents = [item for item in entity_stats if item[1]!=1]
        duplicate_ents.sort(key = lambda x:x[1], reverse=True)
        for item in duplicate_ents:
            ent = item[0]
            ent_idx = [i for i,x in enumerate(path) if x == ent]
            if len(ent_idx)!=0:
                min_idx = min(ent_idx)
                max_idx = max(ent_idx)
                if min_idx!=max_idx:
                    path = path[:min_idx] + path[max_idx:]
        entities_new = []
        relations_new = []
        for idx, item in enumerate(path):
            if idx%2 == 0:
                entities_new.append(item)
            else:
                relations_new.append(item)
        entity_lists_new.append(entities_new);path_lists_new.append(relations_new)
    print('len(entities):',len(entity_lists_new),'len(paths):',len(path_lists_new))
    # episodes
    print('collect episodes')
    good_episodes = []
    targetID = env.entity2id_[e2]
    for path in zip(entity_lists_new,path_lists_new):
        good_episode = []
        for i in range(len(path[0]) -1):
            currID = env.entity2id_[path[0][i]];nextID = env.entity2id_[path[0][i+1]]
            state_curr = [currID, targetID, 0];state_next = [nextID, targetID, 0]
            actionID = env.relation2id_[path[1][i]]
            good_episode.append(Transition(state = env.idx_state(state_curr),\
                                           action = actionID, \
                                           next_state = env.idx_state(state_next), \
                                           reward = 1)) # each time step reward==1
        good_episodes.append(good_episode)
    return good_episodes

def path_clean(path):
    rel_ents = path.split(' -> ')
    relations = []
    entities = []
    for idx, item in enumerate(rel_ents):
        if idx%2 == 0:
            relations.append(item)
        else:
            entities.append(item)
    entity_stats = Counter(entities).items()
    duplicate_ents = [item for item in entity_stats if item[1]!=1]
    duplicate_ents.sort(key = lambda x:x[1], reverse=True)
    for item in duplicate_ents:
        ent = item[0]
        ent_idx = [i for i, x in enumerate(rel_ents) if x == ent]
        if len(ent_idx)!=0:
            min_idx = min(ent_idx)
            max_idx = max(ent_idx)
            if min_idx!=max_idx:
                rel_ents = rel_ents[:min_idx] + rel_ents[max_idx:]
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
    w1 = tf.get_variable('W1', [state_dim, 512], initializer = initializer, regularizer=tf.contrib.layers.l2_regularizer(0.01))
    b1 = tf.get_variable('b1', [512], initializer = tf.constant_initializer(0.0))
    h1 = tf.nn.relu(tf.matmul(state, w1) + b1)
    w2 = tf.get_variable('w2', [512, 1024], initializer = initializer, regularizer=tf.contrib.layers.l2_regularizer(0.01))
    b2 = tf.get_variable('b2', [1024], initializer = tf.constant_initializer(0.0))
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
    w3 = tf.get_variable('w3', [1024, action_dim], initializer = initializer, regularizer=tf.contrib.layers.l2_regularizer(0.01))
    b3 = tf.get_variable('b3', [action_dim], initializer = tf.constant_initializer(0.0))
    action_prob = tf.nn.softmax(tf.matmul(h2,w3) + b3)
    return action_prob

def value_nn(state, state_dim, initializer):
    """
    state_dim -relu-> 64 -> 1
    state -> value_estimated
    """
    w1 = tf.get_variable('w1', [state_dim, 64], initializer = initializer)
    b1 = tf.get_variable('b1', [64], initializer = tf.constant_initializer(0.0))
    h1 = tf.nn.relu(tf.matmul(state,w1) + b1)
    w2 = tf.get_variable('w2', [64,1], initializer = initializer)
    b2 = tf.get_variable('b2', [1], initializer = tf.constant_initializer(0.0))
    value_estimated = tf.matmul(h1, w2) + b2
    return tf.squeeze(value_estimated)

def q_network(state, state_dim, action_space, initializer):
    """
    state_dim -relu-> 128 -relu-> 64 -> action_space
    state -> [w1,b1,w2,b2,w3,b3,action_values]
    """
    w1 = tf.get_variable('w1', [state_dim, 128], initializer=initializer)
    b1 = tf.get_variable('b1', [128], initializer = tf.constant_initializer(0))
    h1 = tf.nn.relu(tf.matmul(state, w1) + b1)
    w2 = tf.get_variable('w2', [128, 64], initializer = initializer)
    b2 = tf.get_variable('b2', [64], initializer = tf.constant_initializer(0))
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
    w3 = tf.get_variable('w3', [64, action_space], initializer = initializer)
    b3 = tf.get_variable('b3', [action_space], initializer = tf.constant_initializer(0))
    action_values = tf.matmul(h2, w3) + b3
    return [w1,b1,w2,b2,w3,b3,action_values]
####################################################################################################################
# SL
class SupervisedPolicy(object):
    def __init__(self, learning_rate = 0.001):
        self.initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope('supervised_policy'):
            self.state = tf.placeholder(tf.float32, [None, state_dim], name = 'state')
            self.action = tf.placeholder(tf.int32, [None], name = 'action')
            self.action_prob = policy_nn(self.state, state_dim, action_space, self.initializer)
            action_mask = tf.cast(tf.one_hot(self.action, depth = action_space), tf.bool)
            self.picked_action_prob = tf.boolean_mask(self.action_prob, action_mask)

            self.loss = tf.reduce_sum(-tf.log(self.picked_action_prob)) + \
                                    sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope = 'supervised_policy'))
            self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, state, sess = None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action_prob, {self.state: state})

    def update(self, state, action, sess = None):
        sess = sess or tf.get_default_session()
        _, loss = sess.run([self.train_op, self.loss], {self.state: state, self.action: action})
        print('loss:',loss)#/state.shape[0])
        return loss

def sl_train(episodes=500):
    tf.reset_default_graph()
    policy_nn = SupervisedPolicy()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for episode in range(len(train_pairs) if len(train_pairs)<episodes else episodes):
            print("Episode %d" % episode);print('Training Sample:', train_pairs[episode%episodes][:-1])
            sample = train_pairs[episode%episodes].split()
            try:
                good_episodes = teacher(sample[0], sample[1], num_paths, env, kb) # good_episodes from teacher
            except Exception as e:
                print('Cannot find a path');continue
            for item in good_episodes: # one episode one supervised batch*<state,action> to update theta
                state_batch,action_batch = [],[]
                for t, transition in enumerate(item):
                    state_batch.append(transition.state)
                    action_batch.append(transition.action)
                state_batch = np.squeeze(state_batch)
                state_batch = np.reshape(state_batch, [-1, state_dim])
                policy_nn.update(state_batch, action_batch)
        saver.save(sess, 'models/policy_supervised_' + relation)
        print('Model saved')


def sl_test(episodes=300):
    tf.reset_default_graph()
    policy_nn = SupervisedPolicy()
    print('len(test_pairs):',len(test_pairs),'test_episodes:',episodes)
    success = 0
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, 'models/policy_supervised_'+ relation);print('Model reloaded')
        for episode in range(episodes):
            try:print('Test sample %d: %s' % (episode,test_pairs[episode][:-1]))
            except:continue
            sample = test_pairs[episode].split()
            # reset env path
            env.path,env.path_relations = [],[]
            state_idx = [env.entity2id_[sample[0]], env.entity2id_[sample[1]], 0]
            for t in count():
                state_vec = env.idx_state(state_idx)
                action_probs = policy_nn.predict(state_vec)
                action_chosen = np.random.choice(np.arange(action_space), p = np.squeeze(action_probs))
                reward, next_state, done = env.interact(state_idx, action_chosen)
                if done or t == max_steps_test:
                    if done:
                        print('Success')
                        success += 1
                    print(f'success:{success},Episode ends')
                    break
                state_idx = next_state
            print('Success persentage:', success/episodes)

    print('Success persentage:', success/episodes)
####################################################################################################################
# RL
class PolicyNetwork(object):
    def __init__(self, scope = 'policy_network', learning_rate = 0.001):
        self.initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [None, state_dim], name = 'state')
            self.action = tf.placeholder(tf.int32, [None], name = 'action')
            # +target
            self.target = tf.placeholder(tf.float32, name = 'target')
            self.action_prob = policy_nn(self.state, state_dim, action_space, self.initializer)

            action_mask = tf.cast(tf.one_hot(self.action, depth = action_space), tf.bool)
            self.picked_action_prob = tf.boolean_mask(self.action_prob, action_mask)
            # +target
            self.loss = tf.reduce_sum(-tf.log(self.picked_action_prob)*self.target) + \
                                    sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope))
            self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, state, sess = None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action_prob, {self.state:state})

    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()
        # +target
        feed_dict = { self.state: state, self.target: target, self.action: action  }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        print('loss:',loss)#/state.shape[0])
        return loss


def REINFORCE(train_pairs, policy_nn, num_episodes):
    success = 0
    path_found = []
    for i_episode in range(num_episodes):
        start = time.time()
        print('Episode %d' % i_episode);print('Training sample: ', train_pairs[i_episode][:-1])
        sample = train_pairs[i_episode].split()
        # reset env path
        env.path,env.path_relations = [],[]
        state_idx = [env.entity2id_[sample[0]], env.entity2id_[sample[1]], 0]
        episode,state_batch_negative,action_batch_negative = [],[],[]
        for t in count():
            state_vec = env.idx_state(state_idx)
            action_probs = policy_nn.predict(state_vec)
            action_chosen = np.random.choice(np.arange(action_space), p = np.squeeze(action_probs))
            reward, new_state_idx, done = env.interact(state_idx, action_chosen)
            # the action fails for this step
            if reward == -1: 
                state_batch_negative.append(state_vec)
                action_batch_negative.append(action_chosen)
            new_state_vec = env.idx_state(new_state_idx)
            episode.append(Transition(state = state_vec,\
                                      action = action_chosen,\
                                      next_state = new_state_vec,\
                                      reward = reward))
            if done or t == max_steps:break
            state_idx = new_state_idx
        # Discourage the agent when it choose an invalid step
        if len(state_batch_negative) != 0:
            print('Penalty to invalid steps:',len(state_batch_negative))
            policy_nn.update(np.reshape(state_batch_negative, (-1, state_dim)), -0.05, action_batch_negative)

        # If the agent success, do one optimization
        def update_episode(policy_nn,episode,total_reward):      
            state_batch = []
            action_batch = []
            for t, transition in enumerate(episode):
                if transition.reward == 0:
                    state_batch.append(transition.state)
                    action_batch.append(transition.action)
            policy_nn.update(np.reshape(state_batch,(-1,state_dim)), total_reward, action_batch)
            
        if done == 1:
            print('Success')
            path_found.append(path_clean(' -> '.join(env.path)))
            success += 1
            length_reward,global_reward = 1/len(env.path),1
            total_reward = 0.1*global_reward + 0.9*length_reward
            update_episode(policy_nn,episode,total_reward)
            print('total_reward success')
        else:
            global_reward = -0.05
            update_episode(policy_nn,episode,global_reward)
            print('Failed, Do one teacher guideline')
            try:
                good_episodes = teacher(sample[0], sample[1], 3, env, kb)
                [update_episode(policy_nn,episode,1) for episode in good_episodes]
                print('Teacher guideline success')
            except Exception as e:
                print('Teacher guideline failed')
        print('Episode time: ',time.time() - start)
        print('Success:',success)
        print('Success percentage:',success/num_episodes)
    print('Success percentage:',success/num_episodes)
    # store path stats
    path_found_relation = [' -> '.join([rel for ix,rel in enumerate(path.split(' -> ')) if ix%2 == 0]) \
                                                                                 for path in path_found]
    relation_path_stats = sorted(Counter(path_found_relation).items(),key = lambda x:x[1],reverse=True)
    with codecs.open('./tasks/'+relation+'/path_stats.txt','w',encoding='utf-8') as f:
        [f.write(item[0]+'\t'+str(item[1])+'\n') for item in relation_path_stats]
        print('Path stats saved')

def rl_retrain(episodes=300):
    print('Start retraining');tf.reset_default_graph()
    policy_network = PolicyNetwork(scope = 'supervised_policy') # restore form parameters of supervised_policy
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, 'models/policy_supervised_' + relation);print("sl_policy restored")
        REINFORCE(train_pairs, policy_network, len(train_pairs) if len(train_pairs)<episodes else episodes)
        saver.save(sess, 'models/policy_retrained_' + relation)
    print('Retrained model saved')

def rl_test(episodes=500):
    tf.reset_default_graph()
    policy_network = PolicyNetwork(scope = 'supervised_policy') # restore form parameters of supervised_policy
    success = 0
    saver = tf.train.Saver()
    path_found = []
    path_set = set()

    with tf.Session() as sess:
        saver.restore(sess, 'models/policy_retrained_' + relation);print('Model reloaded')
        for episode in range(len(test_pairs) if len(test_pairs)<episodes else episodes):
            print('Test sample %d: %s' % (episode,test_pairs[episode][:-1]))
            sample = test_pairs[episode].split()
            # reset env path
            env.path,env.path_relations = [],[]
            state_idx = [env.entity2id_[sample[0]], env.entity2id_[sample[1]], 0]
            transitions = []
            for t in count():
                state_vec = env.idx_state(state_idx)
                action_probs = np.squeeze(policy_network.predict(state_vec))
                action_chosen = np.random.choice(np.arange(action_space), p = action_probs)
                reward, new_state_idx, done = env.interact(state_idx, action_chosen)
                new_state_vec = env.idx_state(new_state_idx)
                transitions.append(Transition(state = state_vec,\
                                              action = action_chosen,\
                                              next_state = new_state_vec,\
                                              reward = reward))
                if done or t == max_steps_test:
                    if done:
                        success += 1;print("Success")
                        path_found.append(path_clean(' -> '.join(env.path)))
                    else:
                        print('Episode ends due to step limit')
                    break
                state_idx = new_state_idx
            if done:
                if len(path_set) != 0:
                    path_found_embedding = [env.path_embedding(path.split(' -> ')) for path in path_set]
                    curr_path_embedding = env.path_embedding(env.path_relations)
                    path_found_embedding = np.reshape(path_found_embedding, (-1,embedding_dim))
                    cos_sim = cosine_similarity(path_found_embedding, curr_path_embedding)
                    diverse_reward = -np.mean(cos_sim)
                    print('diverse_reward', diverse_reward)
                    #total_reward = 0.1*global_reward + 0.8*length_reward + 0.1*diverse_reward 
                    state_batch = []
                    action_batch = []
                    for t, transition in enumerate(transitions):
                        if transition.reward == 0:
                            state_batch.append(transition.state)
                            action_batch.append(transition.action)
                    policy_network.update(np.reshape(state_batch,(-1,state_dim)), 0.1*diverse_reward, action_batch)
                path_set.add(' -> '.join(env.path_relations))
            print('Success:',success)
            print('Success persentage:', success/episodes)
    print('Success persentage:', success/episodes)
    # env path reset
    env.path,env.path_relations = [],[]
    # store path to use 
    path_found_relation = [' -> '.join([rel for ix,rel in enumerate(path.split(' -> ')) if ix%2 == 0]) \
                                                                                 for path in path_found]
    relation_path_stats = sorted(Counter(path_found_relation).items(), key = lambda x:x[1], reverse=True)
    ranking_path = sorted([(path_stat[0],len(path_stat[0].split(' -> '))) \
                           for path_stat in relation_path_stats],\
                          key = lambda x:x[1])
    with codecs.open('./tasks/'+relation+'/path_to_use.txt','w',encoding='utf-8') as f:
        [f.write(item[0]+'\n') for item in ranking_path]
        print('path to use saved')
####################################################################################################################
# MAIN
relation = 'concept_worksfor'
data_path =  '../../NELL-995/'
task_path = data_path + 'tasks/' + relation +'/'
train_pairs = [line for line in codecs.open(task_path+'train_pos','r',encoding='utf-8')]
test_pairs = train_pairs  #= [line for line in codecs.open(task_path+'sort_test.pairs','r',encoding='utf-8')]
env = Env(data_path, relation.replace('_',':'))
kb = KB()
[kb.addRelation(line.rsplit()[0],line.rsplit()[1],line.rsplit()[2]) \
    for line in codecs.open(task_path+'graph.txt','r',encoding='utf-8')]
kb_status = dict([(entity,(False,'','')) for entity in kb.entities.keys()])


def training_pipeline(pipeline=4,
                    sl_train_episodes=500,
                    sl_test_episodes=300,
                    rl_retrain_episodes=300,
                    rl_test_episodes=500):
    '''
    python rl_path_find.py 4 500 300 300 500
    '''
    sl_train(sl_train_episodes) if pipeline > 3 else ''
    sl_test(sl_test_episodes) if pipeline > 2 else ''
    rl_retrain(rl_retrain_episodes) if pipeline > 1 else ''
    rl_test(rl_test_episodes) if pipeline > 0 else ''

if __name__ == '__main__':
    import fire
    fire.Fire(training_pipeline)
