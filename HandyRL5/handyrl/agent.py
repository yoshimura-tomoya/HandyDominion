# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# agent classes

import random

import numpy as np

from .util import softmax

class ManualControlAgent:
    def __init__(self,key=None):
        self.key=key              

    def reset(self, env, show=False):
        pass

    def action(self, env, player, show=False):
        env.show_state() #状況表示
        actions = env.legal_actions(player)
        s = ''
        for i in range(len(actions)):
            if actions[i]<33:
                s += ' ' + str(actions[i]) + ':' + env.cardpool[actions[i]].name+':'+str(env.card_counts[actions[i]])
            else:
                s += ' ' + str(actions[i]) + ':' + 'None'
        print(s)
        print('input your choice')
        x = input()
        x = int(x)
        return x
    
    def play_card(self, env):
        env.act_card_p()


    def observe(self, env, player, show=False):
        return [0.0]

class RandomAgent:
    def reset(self, env, show=False):
        pass

    def action(self, env, player, show=False):
        actions = env.legal_actions(player)
        return random.choice(actions)

    def observe(self, env, player, show=False):
        return [0.0]
    
    def play_card(self, env):
        env.act_card()


class RuleBasedAgent(RandomAgent):
    def __init__(self, key=None):
        self.key = key

    def action(self, env, player, show=False):
        if hasattr(env, 'rule_based_action'):
            return env.rule_based_action(player, key=self.key)
        else:
            return random.choice(env.legal_actions(player))
    
    def play_card(self, env):
        env.act_card()

    def print_outputs(env, prob, v):
        if hasattr(env, 'print_outputs'):
            env.print_outputs(prob, v)
        else:
            if v is not None:
                print('v = %f' % v)
            if prob is not None:
                print('p = %s' % (prob * 1000).astype(int))


class Agent:
    def __init__(self, model, temperature=0.0, observation=True):
        # model might be a neural net, or some planning algorithm such as game tree search
        self.model = model
        self.hidden = None
        self.temperature = temperature
        self.observation = observation

    def reset(self, env, show=False):
        self.hidden = self.model.init_hidden()

    def plan(self, obs):
        outputs = self.model.inference(obs, self.hidden)
        self.hidden = outputs.pop('hidden', None)
        return outputs

    def action(self, env, player, show=False):
        obs = env.observation(player)
        outputs = self.plan(obs)
        actions = env.legal_actions(player)
        p = outputs['policy']
        v = outputs.get('value', None)
        mask = np.ones_like(p)
        mask[actions] = 0
        p = p - mask * 1e32

        #if show:
            #if env.turn()==0:
                #print_outputs(env, softmax(p), v)

        if self.temperature == 0:
            ap_list = sorted([(a, p[a]) for a in actions], key=lambda x: -x[1])
            return ap_list[0][0]
        else:
            return random.choices(np.arange(len(p)), weights=softmax(p / self.temperature))[0]

    def observe(self, env, player, show=False):
        v = None
        if self.observation:
            obs = env.observation(player)
            outputs = self.plan(obs)
            v = outputs.get('value', None)
            if show:
                print_outputs(env, None, v)
        return v
    
    def play_card(self, env):
        env.act_card()
    
class StrAgent(RandomAgent):
    def __init__(self, key=None):
        self.key = key

    def action(self, env, player, show=False):
        actions = env.legal_actions(player)
        s_base1 = [-1e32,1,10,-1e32,-1e32,100,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1]
        mask = np.ones_like(s_base1)
        mask[actions] = 0
        if env.card_counts[5]<=6:
            s_base1[4]=10
        if env.card_counts[5]<=2:
            s_base1[3]=1
        p=s_base1 - mask * 1e32

        return random.choices(np.arange(len(p)), weights=softmax(p))[0]
    
    def play_card(self, env):
        env.act_card()

class StrAgent_p(RandomAgent):
    "ステロ+poacher(最大4枚)"
    def __init__(self, key=None):
        self.key = key

    def action(self, env, player, show=False):
        actions = env.legal_actions(player)
        s_base_s = [-1e32,1,10,-1e32,-1e32,100,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,1,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1]
        mask = np.ones_like(s_base_s)
        mask[actions] = 0
        p=s_base_s
        if env.card_counts[5]<=6:
            p[4]=10
        if env.card_counts[5]<=2:
            p[3]=1
        if env.cards[player][19]>=5:
            p[19] = -1e32
        p=p - mask * 1e32

        return random.choices(np.arange(len(p)), weights=softmax(p))[0]
    
    def play_card(self, env):
        env.act_card()
    
class StrAgent_m(RandomAgent):
    "ステロ+militia(最大4枚)"
    def __init__(self, key=None):
        self.key = key

    def action(self, env, player, show=False):
        actions = env.legal_actions(player)
        s_base_s = [-1e32,1,10,-1e32,-1e32,100,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,1,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1]
        mask = np.ones_like(s_base_s)
        mask[actions] = 0
        p=s_base_s
        if env.card_counts[5]<=6:
            p[4]=10
        if env.card_counts[5]<=2:
            p[3]=1
        if env.cards[player][20]>=5:
            p[20] = -1e32
        p=p - mask * 1e32

        return random.choices(np.arange(len(p)), weights=softmax(p))[0]
    
    def play_card(self, env):
        env.act_card()
    
class StrAgent_w(RandomAgent):
    "ステロ+witch(最大4枚)"
    def __init__(self, key=None):
        self.key = key

    def action(self, env, player, show=False):
        actions = env.legal_actions(player)
        s_base_s = [-1e32,1,10,-1e32,-1e32,100,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,8,-1e32,-1]
        mask = np.ones_like(s_base_s)
        mask[actions] = 0
        p=s_base_s
        if env.card_counts[5]<=6:
            p[4]=10
        if env.card_counts[5]<=2:
            p[3]=1
        if env.cards[player][31]>=5:
            p[31] = -1e32
        p=p - mask * 1e32

        return random.choices(np.arange(len(p)), weights=softmax(p))[0]
    
    def play_card(self, env):
        env.act_card()
    
class StrAgent_b(RandomAgent):
    "ステロ+bandit(最大4枚)"
    def __init__(self, key=None):
        self.key = key

    def action(self, env, player, show=False):
        actions = env.legal_actions(player)
        s_base_s = [-1e32,1,10,-1e32,-1e32,100,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,8,-1e32,-1e32,-1]
        mask = np.ones_like(s_base_s)
        mask[actions] = 0
        p=s_base_s
        if env.card_counts[5]<=6:
            p[4]=10
        if env.card_counts[5]<=2:
            p[3]=1
        if env.cards[player][30]>=5:
            p[30] = -1e32
        p=p - mask * 1e32

        return random.choices(np.arange(len(p)), weights=softmax(p))[0]
    
    def play_card(self, env):
        env.act_card()

class StrAgent_s(RandomAgent):
    "ステロ+smithy(最大4枚)"
    def __init__(self, key=None):
        self.key = key

    def action(self, env, player, show=False):
        actions = env.legal_actions(player)
        s_base_s = [-1e32,1,10,-1e32,-1e32,100,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,5,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1e32,-1]
        mask = np.ones_like(s_base_s)
        mask[actions] = 0
        p=s_base_s
        if env.card_counts[5]<=6:
            p[4]=10
        if env.card_counts[5]<=2:
            p[3]=1
        if env.cards[player][16]>=4:
            p[16] = -1e32
        p=p - mask * 1e32

        return random.choices(np.arange(len(p)), weights=softmax(p))[0]

    def play_card(self, env):
            env.act_card()
    
class ComboAgent(RandomAgent):
    "コンボ（cardpool=comboの時限定）"
    def __init__(self, key=None):
        self.key = key

    def action(self, env, player, show=False):
        actions = env.legal_actions(player)
        c_base = [-1e32,-1e32,-1e32,-3,10,0,-2,1,-1,0,-1,1,-1e32,1,1,2,-1e32,-1]
        mask = np.ones_like(c_base)
        mask[actions] = 0
        p=c_base
        if env.card_counts[16]<=6:
            p[12] = 3
        if env.cards[player][4]>=1:
            p[4] = -1e32
        if env.cards[player][11]>=4:
            p[11] = -1e32
        if env.cards[player][13]>=3:
            p[13] = -1e32
        if env.card_counts[16]<=8:
            p[16] = 15
        p = p - mask*1e32

        return random.choices(np.arange(len(p)), weights=softmax(p))[0]
    
    def play_card(self, env):
        env.act_card()

class EnsembleAgent(Agent):
    def reset(self, env, show=False):
        self.hidden = [model.init_hidden() for model in self.model]

    def plan(self, obs):
        outputs = {}
        for i, model in enumerate(self.model):
            o = model.inference(obs, self.hidden[i])
            for k, v in o.items():
                if k == 'hidden':
                    self.hidden[i] = v
                else:
                    outputs[k] = outputs.get(k, []) + [v]
        for k, vl in outputs.items():
            outputs[k] = np.mean(vl, axis=0)
        return outputs
    
    def play_card(self, env):
        env.act_card()


class SoftAgent(Agent):
    def __init__(self, model):
        super().__init__(model, temperature=1.0)
