# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# implementation of Geister

import random
import itertools

import numpy as np
from operator import attrgetter
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..environment import BaseEnvironment

INF = ()

class DominionNet(nn.Module):
    def __init__(self):
        super(DominionNet, self).__init__()

        self.inputDim = 21*14
        self.hiddenDim = 256
        self.outputDim=19
        self.num_layers   = 1
        self.lstm = nn.LSTM(input_size = self.inputDim,
                            hidden_size = self.hiddenDim,
                            num_layers = self.num_layers,
                            batch_first=True)
        self.output_layer = nn.Linear(self.hiddenDim, self.outputDim)

    def init_hidden(self, batch_size=[]):
        hidden = tuple([
            torch.zeros(*batch_size, self.hiddenDim),
            torch.zeros(*batch_size, self.hiddenDim)
        ])
        return hidden

    def forward(self, x, hidden):
        b, s = x['action'], x['state']
        h = torch.cat([b, s], dim=1)
        flatten = nn.Flatten()
        hf = flatten(h)
        if hf.shape[0] == 128:
            hf = hf.unsqueeze(1)

        output, (hidden, cell) = self.lstm(hf, hidden) #LSTM層
        output = self.output_layer(hidden[-1]) #全結合層
        if output.dim()==1:
            pol = output[:18].unsqueeze(0)
            val = output[18].unsqueeze(0).unsqueeze(0) #unbatched
        elif output.dim()==2:
            pol = output[:, :18]
            val = output[:, 18].unsqueeze(1) #batched

        print(hidden[0].shape)

        return {'policy': pol, 'value': val, 'hidden': hidden}
    
"""
カードと番号の対応表
0:curse
1:copper
2:estate
3:cellar
4:moat
5:silver
6:merchant
7:remodel
8:workshop
9:village
10:militia
11:smithy
12:duchy
13:market
14:mine
15:gold
16:province

"""
class Card(object):
    """
    Represents a class of card.

    To save computation, only one of each card should be constructed. Decks can
    contain many references to the same Card object.
    """
    def __init__(self, name, cost, treasure=0, vp=0, coins=0, cards=0,
                 actions=0, buys=0, potionCost=0, effect=(), isAttack=False,
                 isDefense=False, reaction=(), duration=()):
        self.name = name
        self.cost = cost
        self.potionCost = potionCost
        if isinstance(treasure, int):
            self.treasure = treasure
        else:
            self.treasure = property(treasure)
        if isinstance(vp, int):
            self.vp = vp
        else:
            self.vp = property(vp)
        self.coins = coins
        self.cards = cards
        self.actions = actions
        self.buys = buys
        self._isAttack = isAttack
        self._isDefense = isDefense

        if not isinstance(effect, (tuple, list)):
            self.effect = (effect,)
        else:
            self.effect = effect
        self.reaction = reaction
        self.duration = duration

    def isVictory(self):
        if self.name=='Gardens':
            return True
        return self.vp > 0

    def isCurse(self):
        return self.vp < 0

    def isTreasure(self):
        return self.treasure > 0

    def isAction(self):
        return (self.coins or self.cards or self.actions or self.buys or
                self.effect)

    def isAttack(self):
        return self._isAttack

    def isDefense(self):
        return self._isDefense

    def perform_action(self, Env):
        assert self.isAction()
        player = Env.turn()
        if self.cards:
            Env.draw(player, self.cards)
        if (self.coins or self.actions or self.buys):
            game = Env.change_state(
                player,
                a=self.actions,
                b=self.buys,
                c=self.coins
            )
        for action in self.effect:
            action(Env)

    def __str__(self): return self.name
    def __cmp__(self, other):
        if other is None: return -1
        return ((self.cost, self.name)-(other.cost, other.name)>0) - ((self.cost, self.name)-(other.cost, other.name)<0)
    def __hash__(self):
        return hash(self.name)
    def __repr__(self): return self.name

curse    = Card('Curse', 0, vp=-1)
estate   = Card('Estate', 2, vp=1)
duchy    = Card('Duchy', 5, vp=3)
province = Card('Province', 8, vp=6)

copper = Card('Copper', 0, treasure=1)
silver = Card('Silver', 3, treasure=2)
gold   = Card('Gold', 6, treasure=3)
# simple actions
village = Card('Village', 3, actions=2, cards=1)
woodcutter = Card('Woodcutter', 3, coins=2, buys=1)
smithy = Card('Smithy', 4, cards=3)
festival = Card('Festival', 5, coins=2, actions=2, buys=1)
market = Card('Market', 5, coins=1, cards=1, actions=1, buys=1)
laboratory = Card('Laboratory', 5, cards=2, actions=1)

def chapel_action(Env):
    Env.make_decision(
        TrashDecision(Env, Env.turn(), 0, 4)
    )

def cellar_action(Env):
    a=Env.hand_size(Env.turn())
    Env.make_decision(
        DiscardDecision(Env, Env.turn(), 0, 100)
    )
    card_diff = a - Env.hand_size(Env.turn())
    Env.draw(Env.turn(), card_diff)

def warehouse_action(Env):
    Env.make_decision(
        DiscardDecision(Env, Env.turn(), 3, 3)
    )

def council_room_action(Env):
    Env.evreone_else_draw_card()

def militia_attack(Env, attack=True):
    player = Env.turn()
    player = (player+1) % Env.num_players()
    while player != Env.turn():
        if attack:
            if Env.is_defended(player):
                player = (player+1) % Env.num_players()
                continue
        x=Env.hand_size(player) - 3
        Env.make_decision(
            DiscardDecision(Env, player, x, x)
        )
        player = (player+1) % Env.num_players()

def witch_attack(Env):
    Env.attack_with_gain(curse)

def workshop_action(Env):
    Env.make_decision(
        GainDecision(Env, Env.turn(), 4)
    )

def sentry_action(Env):
    Env.show_cards(Env.turn(), 2)
    Env.make_decision(
        TrashDecision_fromopen(Env, Env.turn(), 0, 2)
    )
    Env.make_decision(
        DiscardDecision_fromopen(Env, Env.turn(), 0, 2)
    )
    while len(Env.opens)>0:
        Env.make_decision(
            BackDecision_Open(Env, Env.turn())
        )

def bandit_action(Env):
    Env.gain(Env.turn(), gold)
    Env.remove_card(Env.turn(), gold)
    Env.bandit_attack(attack=True)

def throneroom_action(Env):
    Env.make_decision(
        PlaytwiceDecision(Env, Env.turn())
    )

def vassal_action(Env):
    Env.show_cards(Env.turn(), 1)
    Env.make_decision(
        PlayDecisionO(Env, Env.turn())
    )
    Env.discard_from_open(Env.turn())

def harbinger_action(Env):
    Env.make_decision(
        BackDecision_discard(Env, Env.turn())
    )

def artisan_action(Env):
    Env.make_decision(
        GainDecision_hand(Env, Env.turn(), 5)
    )
    Env.make_decision(
        BackDecision_hand(Env, Env.turn())
    )

def bureaucrat_action(Env):
    Env.gain_ondeck(Env.turn(), silver)
    Env.remove_card(silver)
    Env.bureaucrat_attack(attack=True)

def moneylender_action(Env):
    Env.moneylender_act()

def mine_action(Env):
    Env.make_decision(
        OpenDecision_Tr(Env, Env.turn())
    )
    if len(Env.opens) != 0: 
        cost = Env.open_cost() + 3
        Env.make_decision(
            GainDecision_Tr(Env, Env.turn(), cost)
        )
        Env.trash_from_open()
    
def remodel_action(Env):
    Env.make_decision(
        OpenDecision(Env, Env.turn())
    )
    if len(Env.opens) != 0:
        cost = Env.open_cost() + 2
        Env.make_decision(
            GainDecision(Env, Env.turn(), cost)
        )
        Env.trash_from_open()
    
def poacher_action(Env):
    x = Env.poacher_count()
    if x > 0:
        newturn = Env.make_decision(
            DiscardDecision(Env, Env.turn(), x, x)
        )

def library_action(Env):
    player = Env.turn()
    while Env.hand_size(player) < 7:
        if Env.deck_size(player) + Env.discard_size(player) == 0:
            break
        Env.draw_cards(1)
        if Env.hands[player][Env.hand_size(player)-1].isAction():
            Env.make_decision(
                LibraryDecision(Env, player)
            )
    Env.discard_from_open()

gardens = Card('Gardens', 4)
merchant = Card('Merchant', 3, cards=1, actions=1)
chapel = Card('Chapel', 2, effect=chapel_action)
cellar = Card('Cellar', 2, actions=1, effect=cellar_action)
warehouse = Card('Warehouse', 3, cards=3, actions=1, effect=warehouse_action)
council_room = Card('Council Room', 5, cards=4, buys=1,
                    effect=council_room_action)
militia = Card('Militia', 4, coins=2, isAttack=True, effect=militia_attack)
moat = Card('Moat', 2, cards=2, isDefense=True)
workshop = Card('Workshop', 3, effect=workshop_action)
witch = Card('Witch', 5, cards=2, isAttack=True, effect=witch_attack)
sentry = Card('Sentry', 5, cards=1, actions=1, effect=sentry_action)
bandit = Card('Bandit', 5, isAttack=True, effect=bandit_action)
throneroom = Card('TroneRoom', 4, effect=throneroom_action)
vassal = Card('Vassal', 3, coins=2, effect=vassal_action)
harbinger = Card('Harbinger', 3, cards=1, actions=1, effect=harbinger_action)
artisan = Card('Artisan', 6, effect=artisan_action)
bureaucrat = Card('Bureaucrat', 4, isAttack=True, effect=bureaucrat_action)
moneylender = Card('Moneylender', 4, effect=moneylender_action)
mine = Card('Mine', 5, effect=mine_action)
remodel = Card('Remodel', 4, effect=remodel_action)
poacher = Card('Poacher', 4, cards=1, coins=1, actions=1, effect=poacher_action)
library = Card('Library', 5, effect=library_action)

variable_cards = [village, cellar, smithy, festival, market, laboratory,
chapel, council_room, moat, workshop, witch, bandit,
throneroom, vassal, harbinger, artisan, bureaucrat, gardens, moneylender, mine,
remodel, merchant, poacher, library, militia, sentry]

class Environment(BaseEnvironment):
    cardpool = [curse, copper, estate, cellar, moat, silver, merchant, remodel, workshop, village,
                 militia, smithy, duchy, market, mine, gold, province]

    def __init__(self, args=None):
        super().__init__()
        self.args = args if args is not None else {}
        self.reset()

    def reset(self, args=None):
        self.game_args = args if args is not None else {}
        self.card_counts = [30,32,8,10,10,40,10,10,10,10,10,10,12,10,10,30,12]
        self.turn_count = -4
        self.hands = [[] for _ in range(4)]
        self.decks = [[1,1,1,1,1,1,1,2,2,2] for _ in range(4)]
        self.discards = [[] for _ in range(4)]
        self.table = []
        self.opens = []
        self.states = [[1,1,0] for _ in range(4)] #action,buy,+coin
        self.round = self.turn_count // 4
        self.winners = [0,0,0,0]
        self.record = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
        self.cards = [[0,7,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0] for _ in range(4)]

    def draw(self, player, n=1):
        hand=self.hands[player]
        deck=self.decks[player]
        discard=self.discards[player]
        if len(deck) >= n:
            self.hands[player]=hand+deck[:n]
            self.decks[player]=deck[n:]
        elif len(discard)+len(deck) >= n:
            x=n-len(deck)
            newdeck = discard
            random.shuffle(newdeck)
            self.hands[player]=hand+deck+newdeck[:x]
            self.decks[player]=newdeck[x:]
            self.discards[player]=[]
        else:
            self.hands[player]=hand+deck+discard
            self.decks[player]=[]
            self.discards[player]=[]
    
    def change_state(self, player, a=0, b=0, c=0):
        self.states[player][0] += a
        self.states[player][1] += b
        self.states[player][2] += c
    
    def deck_size(self, player):
        return len(self.all_cards(player))
    __len__ = deck_size

    def all_cards(self, player):
        return self.hands[player] + self.decks[player] + self.discards[player]

    def hand_value(self, player):
        """How many coins can the player spend?"""
        x=0
        for n in range(self.hand_size(player)):
            card = self.cardpool[self.hands[player][n]]
            x+=card.treasure
        return self.states[player][2] + x

    def hand_size(self, player):
        return len(self.hands[player])
    
    def discard_size(self, player):
        return len(self.discards[player])

    def is_defended(self, player):
        for n in range(self.hand_size(player)):
            card = self.cardpool[self.hands[player][n]]
            if card.isDefense():
                return True
        return False
    
    def get_reactions(self):
        """
        TODO: implement complex reactions like Secret Chamber
        """
        return []

    def show_cards(self, player, n=1):
        """
        Show n cards from top of decks
        """
        hand=self.hands[player]
        deck=self.decks[player]
        discard=self.discards[player]
        if len(deck) >= n:
            self.opens=deck[:n]
            self.decks[player]=deck[n:],
        elif len(discard)+len(deck) >= n:
            x=n-len(deck)
            newdeck = discard
            random.shuffle(newdeck)
            self.opens=deck+newdeck[:x]
            self.deck[player]=newdeck[x:]
        else:
            self.opens=deck+discard
            self.decks[player]=[]
            self.discards[player]=[]
    
    def show_card(self, player, card):
        """
        Show card from hand
        """
        hand=self.hands[player]
        index = list(hand).index(card)
        self.hands[player] = hand[:index] + hand[index+1:]
        self.opens=[hand[index]]
        
    def open_size(self):
        return len(self.opens)

    def gain(self, player, card):
        "Gain a single card."
        self.discards[player]=self.discards[player]+[card]
        self.cards[player][card] += 1
    
    def gain_ondeck(self, player, card):
        "Gain a single card on deck."
        self.discards[player]=[card]+self.discards[player]
        self.cards[player][card] += 1
    
    def gain_hand(self, player, card):
        "Gain a single card in hand."

        self.hands[player]=[card]+self.hands[player]
        self.cards[player][card] += 1
    
    def gain_cards(self, player, cards):
        "Gain multiple cards."
        self.discards[player]=self.discards[player]+cards
        for i in cards:
            self.cards[player][i] += 1

    def play_card(self, player, card):
        """
        Play a card from the hand into the tableau.

        Decreasing the number of actions available is handled in
        play_action(card).
        """
        hand=self.hands[player]
        index = list(hand).index(card)
        self.hands[player] = hand[:index] + hand[index+1:]
        self.table=self.table+[hand[index]]
    
    def play_opentop(self, card):
        """
        for vassal
        """
        self.table=self.table+[card]
        self.opens=[]
    
    def play_action(self, player, card):
        """
        Play an action card, putting it in the tableau and decreasing the
        number of actions remaining.

        This does not actually put the Action into effect; the Action card
        does that when it is chosen in an ActDecision.
        """
        self.play_card(self.turn(), card)
        self.states[player][0]=self.states[player][0]-1

    def discard_card(self, player, card):
        """
        Discard a single card from the hand.
        """
        hand=self.hands[player]
        index = list(hand).index(card)
        self.hands[player] = hand[:index] + hand[index+1:]
        self.discards[player]=self.discards[player]+[hand[index]]

    def trash_card(self, player, card):
        """
        Remove a card from the game.
        """
        hand=self.hands[player]
        index = list(hand).index(card)
        self.hands[player] = hand[:index] + hand[index+1:]
    
    def discard_card_from_open(self, player, card):
        """
        Discard a single card from the show.
        """
        if card != None:
            index = list(self.opens).index(card)
            self.opens = self.opens[:index] + self.opens[index+1:]
            self.discards[player]=self.discards[player]+[card]

    def trash_card_from_open(self, card):
        """
        Remove a card from opens.
        """
        if card != None:
            index = list(self.opens).index(card)
            self.opens = self.opens[:index] + self.opens[index+1:]
    
    def back_from_open(self, player):
        """
        back cards from the open to decks .
        """
        self.decks[player]=self.opens+self.decks[player]
        self.opens=[]

    def back_open_card(self, player, card):
        """
        back a card from the open to decks .
        """
        index = list(self.opens).index(card)
        self.opens = self.opens[:index] + self.opens[index+1:]
        self.decks[player]=[card]+self.decks[player]
    
    def discard_from_open(self, player):
        """
        discard cards from the open.
        """
        self.discards[player]=self.discards[player]+self.opens
        self.opens=[]
    
    def trash_from_open(self):
        """
        trash cards from the open.
        """
        self.opens=[]
    
    def back_from_discard(self, player, card):
        """
        Back a card from the discard on the drowpile.
        """

        index = list(self.discards[player]).index(card)
        self.discards[player] = self.discards[player][:index] + self.discards[player][index+1:]
        self.decks[player]=[card]+self.decks[player]
    
    def back_from_hand(self, player, card):
        """
        Back a card from the hand on the drowpile.
        """
        hand =self.hands[player]
        index = list(self.hands).index(card)
        self.hands[player] = hand[:index] + hand[index+1:]
        self.decks[player]=[card]+self.decks[player]

    def back_open_hand(self, player):
        "バグ回避用"
        self.hands[player] = self.hands[player] + self.opens
        self.opens = []
    
    def open_cost(self):
        """for check the cost"""
        card = self.cardpool[self.opens[0]]
        return card.cost
    
    def actionable(self):
        """Are there actions left to take with this hand?"""
        X=False
        player = self.turn()
        for n in range(self.hand_size(player)):
            card = self.cardpool[self.hands[player][n]]
            if card.isAction():
                X=True
                break
        return (self.states[player][0] > 0 
                and X)

    def buyable(self, player):
        """Can this hand still buy a card?"""
        return self.states[player][1] > 0

    def score(self, player):
        """How many points is this deck worth?"""
        score = 0
        deck_size = self.deck_size(player)
        x= deck_size//10
        for n in range(len(self.all_cards(player))):
            card = self.cardpool[self.all_cards(player)[n]]
            score += card.vp
            if card == gardens:
                score += x
        return score

    def num_players(self):
        return len(self.players())

    def remove_card(self, card):
        """
        Remove a single card from the table.
        """
        self.card_counts[card] -= 1

    def everyone_else_makes_a_decision(self, decision_template, attack=False):
        player = self.turn()
        player = (player+1) % self.num_players()
        while player != self.turn():
            if attack:
                if self.is_defended(player):
                    player = (player+1) % self.num_players()
                    continue
            decision = decision_template
            self.make_decision(decision)
            player = (player+1)%self.num_players()
        

    def attack_with_decision(self, decision):
        return self.everyone_else_makes_a_decision(decision, attack=True)
    
    def everyone_else_gains_a_card(self, card, attack=False):
        player = self.turn()
        player = (player+1) % self.num_players()
        while player != self.turn():
            if attack:
                if self.is_defended(player):
                    player = (player+1) % self.num_players()
                    continue
            self.gain(player, card)
            player = (player+1) % self.num_players()
    
    def everyone_else_draw_card(self):
        player = self.turn()
        player = (player+1) % self.num_players()
        while player != self.turn():
            self.draw(player, 1)
            player = (player+1) % self.num_players()

    def attack_with_gain(self, card):
        return self.everyone_else_gains_a_card(card, attack=True)
    
    def bandit_attack(self, attack=False):
        player = self.turn()
        player = (player+1) % self.num_players()
        while player != self.turn():
            if attack:
                if self.is_defended(player):
                    player = (player+1) % self.num_players()
                    continue
            self.show_cards(player, 2)
            treasures_sorted = [ca for ca in self.opens if self.cardpool[ca].isTreasure()]
            treasures_sorted.sort(key=lambda a: self.cardpool[a].treasure)
            if len(treasures_sorted) == 0:
                self.discard_from_open(player)
            else:
                self.trash_card_from_open(treasures_sorted[0])
                self.discard_from_open(player)
            player = (player+1) % self.num_players()
            
    
    def bureaucrat_attack(self, attack=False):
        player = self.turn()
        player = (player+1) % self.num_players()
        while player != self.turn():
            if attack:
                if self.is_defended(player):
                    player = (player+1) % self.num_players()
                    continue
            victory_cards = [ca for ca in self.hands[player] if self.cardpool[ca].isVictory()]
            if len(victory_cards) == 0:
                print (self.hands[player])
            else:
                print ("%s back %s" % (player, victory_cards[0]))
                self.back_from_hand(player, victory_cards[0])
            player = (player+1) % self.num_players()
    
    def moneylender_act(self):
        player=self.turn()
        coppers = [ca for ca in self.hands[player] if self.cardpool[ca].name == 'Copper']
        if len(coppers) > 0:
            self.trash_card(player, coppers[0])
            self.states[player][1] += 3

    def poacher_count(self):
        zeros = 0
        for count in self.card_counts.values():
            if count == 0: zeros += 1
        return zeros

    def __str__(self):
        # output state

        s = 'turn = ' + str(self.turn_count) + '\n'
        for i in range(len(self.cardpool)):
            s += ' ' + self.cardpool[i].name+':'+str(self.card_counts[i])+ '\n'
        return s
    
    def act2one_hot(self, action):
        player = self.turn()
        one_hot = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        if action == 17:
            return one_hot
        else:
            one_hot[player] = 1
            one_hot[action+4] = 1
            return one_hot

    def play(self, action, _=None):
        # state transition
        player = self.turn()
        if(self.turn_count<0):
            random.shuffle(self.decks[player])
        if action==17:
            self.states[player][1]=0
        else:
            self.gain(player, action)
            self.remove_card(action)
            self.states[player][1] -= 1
            self.states[player][2] -= self.cardpool[action].cost
        self.record.append(self.act2one_hot(action))
        if self.buyable(player)==False:
            self.end_turn()

    def turn(self):
        return self.players()[self.turn_count % 4]

    def end_turn(self):
        player = self.turn()
        self.discards[player] = self.discards[player] + self.hands[player] + self.table
        self.hands[player] = []
        self.table = []
        self.draw(player, 5)
        self.states[player]= [1,1,0]
        self.turn_count += 1

    def terminal(self):
        # check whether terminal state or not
        if self.turn_count > 400: return True #ターンがかかりすぎた場合の処理
        if self.card_counts[16] == 0: return True
        zeros = 0
        i=0
        for i in range(17):
            if self.card_counts[i] == 0: zeros += 1
        return (zeros >= 3)
    
    def players(self):
        return [0,1,2,3]

    def reward(self):
        # return immediate rewards
        return {p: -0.01 for p in self.players()}
    
    def win(self):
        scores=[0,0,0,0]
        m=0
        for i in self.players():
            scores[i]=self.score(i)
        m=max(scores)
        print(scores,m)
        for j in self.players():
            if scores[j]==m:
                self.winners[j]=1

    def outcome(self):
        # return terminal outcomes
        self.win()
        outcomes = self.winners
        return {p: outcomes[idx] for idx, p in enumerate(self.players())}
    
    def coins(self):
        player=self.turn()
        add = 0
        merchants = self.table.count(6)
        for i in range(len(self.hands[player])):
            if self.hands[player][i]==5:
                add = merchants
                break
        return self.hand_value(player) + add

    def legal(self):
        actions = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
        cost = self.coins()
        print('table '+str(self.table)+' hand '+str(self.hands[self.turn()])+' cost '+ str(cost))
        actions = [i for i in actions if self.cardpool[i].cost <= cost]
        return actions

    def legal_actions(self, _=None):
        # return legal action list
        actions = self.legal()
        i=0
        while i < len(actions):
            if self.card_counts[i]==0:
                actions=actions[:i]+actions[i+1:]
            else:
                i+=1
        actions = actions + [17]
        return actions
    
    def card_choices(self):
        """
        List all the cards that can currently be bought.
        """
        i=0
        choices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
        while i < len(choices):
            if self.card_counts[i]==0:
                choices=choices[:i]+choices[i+1:]
            else:
                i+=1
        return choices
    
    def action2str(self, action):
        if action == 17:
            return None
        return self.cardpool[action].name

    def str2action(self, s):
        if s == None:
            return 17
        for i in range(self.cardpool):
            if s == self.cardpool[i].name:
                return i
            
    def count_discard(self, player):
        count = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        for i in range(self.discard_size(player)):
            count[self.discards[player][i]] += 1
        return count
    
    def count_hand(self, player):
        count = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        for i in range(self.hand_size(player)):
            count[self.hands[player][i]] += 1
        return count
    
    def act_card(self):
        player = self.turn()
        while self.actionable():
            self.make_decision(ActDecision(self, player))
    
    def hand_card(self, player):
        player == self.turn()
        if player == 0:
            x = self.count_hand(0)
        else:
            x = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        return x


    def observation(self, player=None):
        # state representation to be fed into neural networks
        turn_view = player is None or player == self.turn()

        a = np.array([
            self.record[-1]
        ]).astype(np.float32) # 1*1*21

        x = [self.score(0)]+[self.score(1)]+[self.score(2)]+[self.score(3)]+self.card_counts
        s = np.stack([
            # score + remain
            x,
            # my/opponent's number of cards
            [1,0,0,0] + self.count_discard(0),
            [1,0,0,0] + self.hand_card(0),
            [1,0,0,0] + self.cards[0],
            [0,1,0,0] + self.count_discard(1),
            [0,1,0,0] + self.hand_card(1),
            [0,1,0,0] + self.cards[1],
            [0,0,1,0] + self.count_discard(2),
            [0,0,1,0] + self.hand_card(2),
            [0,0,1,0] + self.cards[2],
            [0,0,0,1] + self.count_discard(3),
            [0,0,0,1] + self.hand_card(3),
            [0,0,0,1] + self.cards[3],
        ]).astype(np.float32) # 1*13*21

        return {'action': a, 'state': s}

    def net(self):
        return DominionNet()
    
    def make_decision(self, decision):
        if isinstance(decision, ActDecision):
            choice = self.make_act_decision(decision)
        elif isinstance(decision, PlayDecision):
            choice = self.make_act_decision(decision)
        elif isinstance(decision, PlayDecisionO):
            choice = self.make_act_decision(decision)
        elif isinstance(decision, PlaytwiceDecision):
            choice = self.make_act_decision(decision)
        elif isinstance(decision, DiscardDecision):
            choice = self.make_discard_decision(decision)
        elif isinstance(decision, TrashDecision):
            choice = self.make_trash_decision(decision)
        elif isinstance(decision, DiscardDecision_fromopen):
            choice = self.make_discard_decision_fromopen(decision)
        elif isinstance(decision, TrashDecision_fromopen):
            choice = self.make_trash_decision_fromopen(decision)
        elif isinstance(decision, BackDecision_discard):
            choice = self.make_back_discard(decision)
        elif isinstance(decision, BackDecision_hand):
            choice = self.make_back_hand(decision)
        elif isinstance(decision, BackDecision_Open):
            choice = self.make_random_decision(decision)
        elif isinstance(decision, OpenDecision):
            choice = self.make_open_decision(decision)
        elif isinstance(decision, OpenDecision_Tr):
            choice = self.make_open_decision_Tr(decision)
        elif isinstance(decision, GainDecision):
            choice = self.make_gain_decision(decision)
        elif isinstance(decision, GainDecision_hand):
            choice = self.make_gain_decision(decision)
        elif isinstance(decision, GainDecision_Tr):
            choice = self.make_gain_decision(decision)
        elif isinstance(decision, LibraryDecision):
            choice = self.make_library_decision(decision)
        else:
            raise NotImplementedError
        return decision.choose(choice)
    
    def buy_priority_order(self):
        """
        Provide a buy_priority by ordering the cards from least to most
        important.
        """
        provinces_left = self.card_counts[16]
        if provinces_left <= self.cutoff1:
            return [None, 2, 5, 12, 16]
        elif provinces_left <= self.cutoff2:
            return [None, 5, 12, 15, 16]
        else:
            return [None, 5, 15, 16]
    
    def buy_priority(self, card):
        """
        Assign a numerical priority to each card that can be bought.
        """
        try:
            return self.buy_priority_order().index(card)
        except ValueError:
            return -1
    
    def make_buy_decision(self, decision):
        """
        Choose a card to buy.

        By default, this chooses the card with the highest positive
        buy_priority.
        """
        choices = decision.choices()
        choices.sort(key=lambda x: self.buy_priority(x))
        return choices[-1]
    
    def make_gain_decision(self, decision):
        """
        Choose a card to gain.

        Todo
        """
        choices = decision.choices()
        if choices==[]:
            return None
        return choices[-1]
    
    def act_priority(self, choice):
        """
        Assign a numerical priority to each action. Higher priority actions
        will be chosen first.
        """
        if choice is None: return 0
        return (100*self.cardpool[choice].actions + 10*(self.cardpool[choice].coins + self.cardpool[choice].cards) +
                    self.cardpool[choice].buys) + 1
    
    def make_act_decision(self, decision):
        """
        Choose an Action to play.

        By default, this chooses the action with the highest positive
        act_priority.
        """
        choices = decision.choices()
        choices.sort(key=lambda x: self.act_priority(x))
        return choices[-1]
    
    def make_trash_decision_incremental(self, decision, choices, chosen, allow_none=True):
        "Choose a single card to trash."
        deck = self.all_cards(decision.player)
        money = sum([self.cardpool[card].treasure + self.cardpool[card].coins for card in deck]) - sum([self.cardpool[card].treasure + self.cardpool[card].coins for card in chosen])
        if 0 in choices:
            return 0
        elif 1 in choices and money > 3:
            return 1
        elif self.round < 10 and 2 in choices:
            # TODO: judge how many turns are left in the game and whether
            # an estate is worth it
            return 2
        elif allow_none:
            return None
        else:
            # oh shit, we don't know what to trash
            # get rid of whatever looks like it's worth the least
            choices.sort(key=lambda x: (self.cardpool[x].vp, self.cardpool[x].cost))
            return choices[0]

    def make_trash_decision(self, decision):
        """
        The default way to decide which cards to trash is to repeatedly
        choose one card to trash until None is chosen.

        TrashDecision is a MultiDecision, so return a list.
        """
        latest = False
        chosen = []
        choices = decision.choices()
        while choices and latest is not None and len(chosen) < decision.max:
            latest = self.make_trash_decision_incremental(
                decision, choices, chosen,
                allow_none = (len(chosen) >= decision.min)
            )
            if latest is not None:
                choices.remove(latest)
                chosen.append(latest)
        return chosen
    
    def make_trash_decision_fromopen(self, decision):
        latest = False
        chosen = []
        choices = decision.choices()
        while len(chosen) < decision.max:
            latest = self.make_trash_decision_incremental(
                decision, choices, chosen,
                allow_none = (len(chosen) >= decision.min)
            )
            if latest is None:
                return chosen
            if latest is not None:
                choices.remove(latest)
                chosen.append(latest)
        return chosen
    
    def make_open_decision(self, decision):
        latest = False
        chosen = []
        choices = decision.choices()
        if len(choices)==0:
            return None
        latest = self.make_trash_decision_incremental(
            decision, choices, chosen,
            allow_none = False
        )
        return latest

    def make_discard_decision_incremental(self, decision, choices, allow_none=True):
        actions_sorted = [ca for ca in choices if self.cardpool[ca].isAction()]
        actions_sorted.sort(key=lambda a: self.cardpool[a].actions)
        plus_actions = sum([self.cardpool[ca].actions for ca in actions_sorted])
        wasted_actions = len(actions_sorted) - plus_actions - self.states[decision.player][0]
        victory_cards = [ca for ca in choices if self.cardpool[ca].isVictory() and
                         not self.cardpool[ca].isAction() and not self.cardpool[ca].isTreasure()]
        if wasted_actions > 0:
            return actions_sorted[0]
        elif len(victory_cards):
            return victory_cards[0]
        elif 1 in choices:
            return 1
        elif allow_none:
            return None
        else:
            priority_order = sorted(choices,
              key=lambda ca: (self.cardpool[ca].actions, self.cardpool[ca].cards, self.cardpool[ca].coins, self.cardpool[ca].treasure))
            return priority_order[0]

    def make_discard_decision(self, decision):
        # TODO: make this good.
        # This probably involves finding all distinct sets of cards to discard,
        # of size decision.min to decision.max, and figuring out how well the
        # rest of your hand plays out (including things like the Cellar bonus).

        # Start with 
        #   game = decision.game().simulated_copy() ...
        # to avoid cheating.

        latest = False
        chosen = []
        choices = decision.choices()
        while choices and latest is not None and len(chosen) < decision.max:
            latest = self.make_discard_decision_incremental(
                decision, choices,
                allow_none = (len(chosen) >= decision.min)
            )
            if latest is not None:
                choices.remove(latest)
                chosen.append(latest)
        return chosen
    
    def make_discard_fromopen_incremental(self, decision, choices, allow_none=True):
        if len(choices)==1:
            return None
        choices2 = choices[1:]
        victory_cards = [ca for ca in choices2 if self.cardpool[ca].isVictory()]
        if len(victory_cards):
            return victory_cards[0]
        elif 1 in choices:
            return 1
        elif allow_none:
            return None
        else:
            priority_order = sorted(choices,
              key=lambda ca: (self.cardpool[ca].actions, self.cardpool[ca].cards, self.cardpool[ca].coins, self.cardpool[ca].treasure))
            return priority_order[0]
    
    def make_discard_decision_fromopen(self, decision):
        latest = False
        chosen = []
        choices = decision.choices()
        while len(chosen) < decision.max:
            latest = self.make_discard_fromopen_incremental(
                decision, choices,
                allow_none = True
            )
            if latest is None:
                return chosen
            if latest is not None:
                choices.remove(latest)
                chosen.append(latest)
        return chosen
    
    def make_open_decision_Tr(self, decision):
        choices = decision.choices()
        if len(choices) == 1:
            return choices[0]
        choices = [ca for ca in choices if ca!=None]
        choices.sort(key=lambda x: self.cardpool[x].cost)
        without_gold = [ca for ca in choices if self.cardpool[ca].name!='Gold']
        if len(without_gold)>0:
            return without_gold[-1]
        elif len(choices)>0:
            return choices[0]
        return None
    
    def make_back_hand(self, decision):
        choices = decision.choices()
        actions_sorted = [ca for ca in choices if self.cardpool[ca].isAction()]
        treasures = [ca for ca in choices if self.cardpool[ca].isTreasure()]
        treasures.sort(key=lambda a: self.cardpool[a].treasure)
        victory_cards = [ca for ca in choices if self.cardpool[ca].isVictory()]
        actions_sorted.sort(key=lambda a: self.cardpool[a].actions)
        plus_actions = sum([self.cardpool[ca].actions for ca in actions_sorted])
        wasted_actions = len(actions_sorted) - plus_actions - self.states[decision.player][0]
        if wasted_actions>0:
            return actions_sorted[0]
        elif len(treasures)>0:
            return treasures[0]
        elif len(victory_cards)>0:
            return victory_cards[0]
        else:
            return actions_sorted[0]
        
    def make_back_discard(self, decision):
        choices = decision.choices()
        actions_sorted = [ca for ca in choices if self.cardpool[ca].isAction()]
        treasures = [ca for ca in choices if self.cardpool[ca].isTreasure()]
        treasures.sort(key=lambda a: self.cardpool[a].treasure)
        actions_sorted.sort(key=lambda a: self.cardpool[a].actions)
        if len(actions_sorted)>0:
            return actions_sorted[-1]
        elif (len(treasures)>0 and treasures[-1].treasure>2):
            return treasures[-1]
        else:
            return None
    
    def make_library_decision(self, decision):
        choices = decision.choices()
        actions_sorted = [ca for ca in self.hands[decision.player] if self.cardpool[ca].isAction()]
        actions_sorted.sort(key=lambda a: self.cardpool[a].actions)
        plus_actions = sum([self.cardpool[ca].actions for ca in actions_sorted])
        wasted_actions = len(actions_sorted) - plus_actions - self.states[decision.player][0]
        if wasted_actions>0:
            return choices[1]
        else:
            return choices[0]
    
    def make_random_decision(self, decision):
        """
        Choose a card.
        詳細未実装の仮置き場
        """
        choices = decision.choices()
        random.shuffle(choices)
        return choices[0]
    
class Decision(object):
    def __init__(self, Env, player):
        self.Env = Env
        self.player = player

class MultiDecision(Decision):
    def __init__(self, Env, player, min=0, max=INF):
        self.min=min
        self.max=max
        Decision.__init__(self, Env, player)

class limitedDecision(Decision):
    def __init__(self, Env, player, cost=0):
        self.cost=cost
        Decision.__init__(self, Env, player)

class ActDecision(Decision):
    def choices(self):
        return [None] + [card for card in self.Env.hands[self.player] if self.Env.cardpool[card].isAction()]
    def choose(self, card):
        player = self.player
        print('player:'+str(player) +'  '+ str(card))
        if card is None:
            print('play None')
            self.Env.change_state(player,a= -self.Env.state[player][0])
        else:
            #print('hand '+str(self.Env.hands[0])+'dis '+str(self.Env.discards[0])+'deck '+str(self.Env.decks[0])+'sta '+str(self.Env.states[0]))
            #print('hand '+str(self.Env.hands[1])+'dis '+str(self.Env.discards[1])+'deck '+str(self.Env.decks[1])+'sta '+str(self.Env.states[1]))
            #print('hand '+str(self.Env.hands[2])+'dis '+str(self.Env.discards[2])+'deck '+str(self.Env.decks[2])+'sta '+str(self.Env.states[2]))
            #print('hand '+str(self.Env.hands[3])+'dis '+str(self.Env.discards[3])+'deck '+str(self.Env.decks[3])+'sta '+str(self.Env.states[3]))
            self.Env.play_action(player, card)
            self.Env.cardpool[card].perform_action(self.Env)
            #print('play  ' + self.Env.cardpool[card].name)
            #print('hand '+str(self.Env.hands[0])+'dis '+str(self.Env.discards[0])+'deck '+str(self.Env.decks[0])+'sta '+str(self.Env.states[0]))
            #print('hand '+str(self.Env.hands[1])+'dis '+str(self.Env.discards[1])+'deck '+str(self.Env.decks[1])+'sta '+str(self.Env.states[1]))
            #print('hand '+str(self.Env.hands[2])+'dis '+str(self.Env.discards[2])+'deck '+str(self.Env.decks[2])+'sta '+str(self.Env.states[2]))
            #print('hand '+str(self.Env.hands[3])+'dis '+str(self.Env.discards[3])+'deck '+str(self.Env.decks[3])+'sta '+str(self.Env.states[3]))
    def __str__(self):
        player = self.player
        return "ActDecision (%d actions, %d buys, +%d coins)" %\
          (self.Env.states[player][0], self.Env.states[player][1], self.Env.states[player][2])
    
class PlayDecision(Decision):
    def choices(self):
        return [None] + [card for card in self.Env.hands[self.player] if self.Env.cardpool[card].isAction()]
    def choose(self, card):
        player = self.player
        if card is None:
            self.Env.change_state(player,a= -self.Env.state[player][0])
        else:
            self.Env.play_card(player, card)
            self.Env.cardpool[card].perform_action(self.Env)
    def __str__(self):
        player = self.player
        return "PlayDecision (%d actions, %d buys, +%d coins)" %\
          (self.Env.states[player][0], self.Env.states[player][1], self.Env.states[player][2])
    
class PlaytwiceDecision(Decision):
    def choices(self):
        return [None] + [card for card in self.Env.hands[self.player] if self.Env.cardpool[card].isAction()]
    def choose(self, card):
        player = self.player
        if card is None:
            self.Env.change_state(player,a= -self.Env.state[player][0])
        else:
            self.Env.play_action(player, card)
            card.perform_action(self.Env)
            card.perform_action(self.Env)
    def __str__(self):
        player = self.player
        return "PlaytwiceDecision (%d actions, %d buys, +%d coins)" %\
          (self.Env.states[player][0], self.Env.states[player][1], self.Env.states[player][2])
    
class PlayDecisionO(Decision):
    def choices(self):
        return [None] + [card for card in self.Env.opens if card.isAction()]
    def choose(self, card):
        if card is not None:
            card.perform_action(self.Env)
            self.Env.play_opentop(card)
    def __str__(self):
        player = self.player
        return "PlayDecision (%d actions, %d buys, +%d coins)" %\
          (self.Env.states[player][0], self.Env.states[player][1], self.Env.states[player][2])

class TrashDecision(MultiDecision):
    def choices(self):
        return sorted(list(self.Env.hands[self.player]))

    def choose(self, choices):
        for card in choices:
            self.Env.trash_card(self.player, card)

    def __str__(self):
        return "TrashDecision(%s, %s, %s)" % (self.Env.hands[self.player], self.min, self.max)

class DiscardDecision(MultiDecision):
    def choices(self):
        return sorted(list(self.Env.hands[self.player]))

    def choose(self, choices):
        for card in choices:
            self.Env.discard_card(self.player, card)
    
    def __str__(self):
        return "DiscardDecision" + str(self.Env.hands[self.player])
    
class TrashDecision_fromopen(MultiDecision):
    def choices(self):
        return [None] + sorted(list(self.Env.opens))

    def choose(self, choices):
        if choices is not None:
            for card in choices:
                self.Env.trash_card_from_open(card)

    def __str__(self):
        return "TrashDecision(%s, %s, %s)" % (self.Env.hands[self.player], self.min, self.max)
    
class BackDecision_discard(Decision):
    def choices(self):
        return [card for card in self.Env.discards[self.player]]
    def choose(self, card):
        if card is not None:
            self.Env.back_from_discard(self.player, card)
    def __str__(self):
        return "BackDecision" + str(self.Env.discards[self.player])
    
class BackDecision_Open(Decision):
    def choices(self):
        return [card for card in self.Env.opens]
    def choose(self, card):
        self.Env.back_open_card(self.player, card)
    def __str__(self):
        return "BackDecision" + str(self.Env.opens)
    
class BackDecision_hand(Decision):
    def choices(self):
        return [card for card in self.Env.hands[self.player]]
    def choose(self, card):
        if card is not None:
            self.Env.back_from_hand(self.player, card)
    def __str__(self):
        return "BackDecision" + str(self.Env.hands[self.player])

class DiscardDecision_fromopen(MultiDecision):
    def choices(self):
        return [None] + sorted(list(self.Env.opens))
    
    def choose(self, choices):
        for card in choices:
            self.Env.discard_card_from_open(self.player, card)
    
    def __str__(self):
        return "DiscardDecision" + str(self.Env.opens)

class GainDecision(limitedDecision):
    def choices(self):
        cost = self.cost
        return [None] + [card for card in self.Env.card_choices() if self.Env.cardpool[card].cost <= cost]
    def choose(self, card):
        if card is not None:
            self.Env.remove_card(card)
            self.Env.gain(self.player, card)

    def __str__(self):
        return "GainDecision (under %d coins)" %\
          (self.cost)

class GainDecision_hand(limitedDecision):
    def choices(self):
        cost = self.cost
        return [None] + [card for card in self.Env.card_choices() if self.Env.cardpool[card].cost <= cost]
    def choose(self, card):
        if card is not None:
            self.Env.remove_card(card)
            self.Env.gain_hand(self.player, card)
    
    def __str__(self):
        return "GainDecision (under %d coins)" %\
          (self.cost)
    
class GainDecision_Tr(limitedDecision):
    """for mine"""
    def choices(self):
        cost = self.cost
        return [card for card in self.Env.card_choices() if self.Env.cardpool[card].cost <= cost and self.Env.cardpool[card].isTreasure()]
    def choose(self, card):
        if card is not None:
            self.Env.remove_card(card)
            self.Env.gain_hand(self.player, card)
        else:
            self.Env.back_open_hand(self.player)
    
    def __str__(self):
        return "GainDecision (under %d coins)" %\
          (self.cost)

class OpenDecision(Decision):
    """
    手札を捨て札、または廃棄した上でそのカードの情報を参照する際、
    一時的に公開ゾーンに送り、処理終了後に改めて本来の位置に動かす方針
    """
    def choices(self):
        return sorted(list(self.Env.hands[self.player]))
    
    def choose(self, card):
        if card is not None:
            self.Env.show_card(self.player, card)
    
    def __str__(self):
        return "OpenDecision" + str(self.Env.hands[self.player])
    
class OpenDecision_Tr(Decision):
    """
    treasure only
    """
    def choices(self):
        treasures = [ca for ca in self.Env.hands[self.player] if self.Env.cardpool[ca].isTreasure()]
        return [None] + sorted(treasures)
    
    def choose(self, card):
        if card is not None:
            self.Env.show_card(self.player, card)
    
    def __str__(self):
        return "OpenDecision" + str(self.Env.hands[self.player])
    
class LibraryDecision(Decision):
    """
    for library action
    """
    def choices(self):
        return [None]+[self.Env.hands[self.player][-1]]
    
    def choose(self, card):
        if card is not None:
            self.Env.show_card(self.player, card)
    
    def __str__(self):
        return "LibraryDecision"


if __name__ == '__main__':
    e = Environment()
    for _ in range(1):
        e.reset()
         #while not e.terminal():
        for _ in range(20):
            player=e.turn()
            print('turn= '+ str(e.turn_count))
            e.act_card()
            actions = e.legal_actions()
            while player==e.turn():
                #print([e.action2str(a) for a in actions])
                e.play(random.choice(actions))
        #print(e)
        print(e.outcome())
