#パッケージのインポート
from game import State
# from dual_network import DN_INPUT_SHAPE
from math import sqrt
from tensorflow.keras.models import load_model
from pathlib import Path
import numpy as np


DN_INPUT_SHAPE = (3,3,2) #入力シェイプ
#パラメータの準備
PV_EVALUATE_COUNT = 50 #１推論当たりのシミュレーション回数（本家は１６００）

#推論
def predict(model, state):
    #推論のための入力データのシェイプの変換
    a, b, c = DN_INPUT_SHAPE
    x = np.array([state.pieces, state.enemy_pieces])
    x = x.reshape(c,a,b).transpose(1,2,0).reshape(1,a,b,c)

    #推論
    y = model.predict(x, batch_size=1)

    #方策の取得
    policies = y[0][0][list(state.legal_actions())] #合法手のみ
    policies /= sum(policies) if sum(policies) else 1 #合計１の確立分布に変換

    #価値の取得
    value = y[1][0][0]
    return policies, value

#ノードのリストを試行回数のリストに変換
def nodes_to_scores(nodes):
    scores = []
    for c in nodes:
        scores.append(c.n)
    return scores

#モンテカルロ木探索のスコアの取得
def pv_mcts_scores(model, state, temperature):
    #モンテカルロ木探索のノードの定義
    class node:
        #ノードの初期化
        def __init__(self, state, p):
            self.state = state
            self.p = p #方策
            self.w = 0 #価値
            self.n = 0 #試行回数
            self.child_nodes = None #子ノード群

        #局面の価値の計算
        def evaluate(self):
            #ゲーム終了時
            if self.state.is_done():
                #勝敗結果で価値を取得
                value = -1 if self.state.is_lose() else 0

                self.w += value
                self.n += 1
                return value
            
            #子ノードが存在しないとき
            if not self.child_nodes:
                #ニューラルネットワークの推論で方策と価値を取得
                policies, value = predict(model, self.state)

                self.w += value
                self.n += 1

                #子ノードの展開
                self.child_nodes = []
                for action, policy in zip(self.state.legal_actions(), policies):
                    self.child_nodes.append(node(self.state.next(action), policy))
                return value
            #子ノードが存在するとき
            else:
                #アーク評価値が最大の子ノードの評価で価値を取得
                value = -self.next_child_node().evaluate()

                self.w += value
                self.n += 1
                return value
            
        #アーク評価値が最大の子ノードを取得
        def next_child_node(self):
            #アーク評価値の計算
            C_PUCT = 1.0
            t = sum(nodes_to_scores(self.child_nodes))
            pucb_values = []
            for child_node in self.child_nodes:
                pucb_values.append((-child_node.w / child_node.n if child_node.n else 0.0) +
                                   C_PUCT * child_node.p * sqrt(t) / (1 + child_node.n))
            #アーク評価値が最大の子ノードを返す
            return self.child_nodes[np.argmax(pucb_values)]









    #現在の局面のノードの作成
    root_node = node(state, 0)

    #複数回の評価を実行
    for _ in range(PV_EVALUATE_COUNT):
        root_node.evaluate()

    #合法手の確立分布
    scores = nodes_to_scores(root_node.child_nodes)
    if temperature == 0: #最大値のみ１
        action = np.argmax(scores)
        scores = np.zeros(len(scores))
        scores[action] = 1
    else: #ボルツマン分布でバラつき付加
        scores = boltzman(scores, temperature)
    return scores


#モンテカルロ木探索で行動選択
def pv_mcts_action(model, temperature = 0):
    def pv_mcts_action(state):
        scores = pv_mcts_scores(model, state, temperature)
        return np.random.choice(state.legal_actions(), p = scores)
    return pv_mcts_action
#ボルツマン分布
def boltzman(xs, temperature):
    xs = [x**(1/temperature) for x in xs]
    return [x / sum(xs) for x in xs]
