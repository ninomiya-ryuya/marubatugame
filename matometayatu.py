#３目並べの作成
#game.pyの部分
import random
import math

# import pyodide

# await pyodide.loadPackage("numpy")

# import os  #tensorflowのエラーを防ぐため
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #tensorflowのエラーを防ぐため

class State:
    #初期化
    def __init__(self, pieces=None, enemy_pieces=None):
        #石の配置
        self.pieces = pieces if pieces != None else [0]*9
        self.enemy_pieces = enemy_pieces if enemy_pieces != None else [0]*9

    #石の数の取得
    def piece_count(self, pieces):
        count=0
        for i in pieces:
            if i==1:
                count += 1
        return count

    #負けかどうか
    def is_lose(self):
        #3並びかどうか
        def is_comp(x,y,dx,dy):
            for k in range(3):
                if y<0 or 2<y or x<0 or 2<x or self.enemy_pieces[x+y*3]==0:
                    return False
                x,y = x+dx,y+dy
            return True

        #負けかどうか
        if is_comp(0,0,1,1) or is_comp(0,2,1,-1):
            return True
        for i in range(3):
            if is_comp(0,i,1,0) or is_comp(i,0,0,1):
                return True
        return False

    #引き分けかどうか
    def is_draw(self):
        return self.piece_count(self.pieces) + self.piece_count(self.enemy_pieces)==9

    #ゲーム終了かどうか
    def is_done(self):
        return self.is_lose() or self.is_draw()

    #次の状態の取得
    def next(self,action):
        pieces = self.pieces.copy()
        pieces[action] = 1
        return State(self.enemy_pieces, pieces)

    #合法手のリストの取得
    def legal_actions(self):
        actions = []
        for i in range(9):
            if self.pieces[i] == 0 and self.enemy_pieces[i] == 0:
                actions.append(i)
        return actions

    #先手かどうか
    def is_first_player(self):
        return self.piece_count(self.pieces) == self.piece_count(self.enemy_pieces)

    #文字列表示
    def __str__(self):
        ox = ('o','x') if self.is_first_player() else ('x','o')
        str = ''
        for i in range(9):
            if self.pieces[i] == 1:
                str += ox[0]
            elif self.enemy_pieces[i] ==1:
                str += ox[1]
            else:
                str += '-'
            if i % 3 == 2:
                str  += '\n'
        return str
    
#ランダムで行動選択
def random_action(state):
    legal_actions = state.legal_actions()
    return legal_actions[random.randint(0,len(legal_actions)-1)]

#alphabetaで状態価値計算
def alpha_beta(state,alpha,beta):
    if state.is_lose():
        return -1
    if state.is_draw():
        return 0
    
    for action in state.legal_actions():
        score = -alpha_beta(state.next(action),-beta,-alpha)
        if score > alpha:
            alpha = score

        if alpha >= beta:
            return alpha
        
    return alpha

#alphabetaで行動選択
def alpha_beta_action(state):
    #合法手の状態価値の計算
    best_action = 0
    alpha = -float('inf')
    str = ['','']
    for action in state.legal_actions():
        score = -alpha_beta(state.next(action),-float('inf'),-alpha)
        if score > alpha:
            best_action = action
            alpha = score

    return best_action

#プレイアウト
def playout(state):
    if state.is_lose():
        return -1
    if state.is_draw():
        return 0
    #次の状態の状態価値
    return -playout(state.next(random_action(state)))

#最大値のインデックスを返す
def argmax(collection, key=None):
    return collection.index(max(collection))

#モンテカルロ木探索の行動選択
def mcts_action(state):
    #モンテカルロ木探索のノードの定義
    class Node:
        #ノードの初期化
        def __init__(self, state):
            self.state = state #状態
            self.w = 0 #累計価値
            self.n = 0 #試行回数
            self.child_nodes = None #子ノード群

        #局面の価値の計算
        def evaluate(self):
            #ゲーム終了時
            if self.state.is_done():
                #勝敗結果で価値を取得
                value = -1 if self.state.is_lose() else 0 #負け-1,引き分け0
                #累計価値と試行回数の更新
                self.w += value
                self.n += 1
                return value
            
            #子ノードが存在しない時
            if not self.child_nodes:
                #プレイアウトで価値を取得
                value = playout(self.state)
                #累計価値と試行回数の更新
                self.w += value
                self.n += 1

                #子ノードの展開
                if self.n == 10:
                    self.expand()
                return value
            
            #子ノードが存在する時
            else:
                #UCB1が最大の子ノードの評価で価値を取得
                value = -self.next_child_node().evaluate()
                
                self.w += value
                self.n += 1
                return value
        #子ノードの展開
        def expand(self):
            legal_actions = self.state.legal_actions()
            self.child_nodes = []
            for action in legal_actions:
                self.child_nodes.append(Node(self.state.next(action)))

        #UCB1が最大の子ノードの取得
        def next_child_node(self):
            #試行回数0の子ノードを返す
            for child_node in self.child_nodes:
                if child_node.n == 0:
                    return child_node
            #UCBIの計算
            t = 0
            for c in self.child_nodes:
                t += c.n
                ucb1_values =  []
                for child_node in self.child_nodes:
                    ucb1_values.append(-child_node.w/child_node.n+(2*math.log(t)/child_node.n)**0.5)
                #UCB1が最大のノードを返す
                return self.child_nodes[argmax(ucb1_values)]

        
    #現在の局面のノードの作成
    root_node = Node(state)
    root_node.expand()

    #100回のシミュレーションを実行
    for _ in range(100):
        root_node.evaluate()

    #試行回数の最大値を持つ行動を返す
    legal_actions = state.legal_actions()
    n_list = []
    for c in root_node.child_nodes:
        n_list.append(c.n)
    return legal_actions[argmax(n_list)]


#pv_msts.pyの部分
#パッケージのインポート
# from game import State
# from dual_network import DN_INPUT_SHAPE
from math import sqrt
# from tensorflow.keras.models import load_model
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

#pyxelpractice2.pyの部分
import pyxel
import numpy as np
# from game import State
# from pv_mcts import pv_mcts_action

# 畳み込み層のフィルター数と残差ブロックの数
DN_FILTERS = 128
DN_RESIDUAL_NUM = 16
DN_INPUT_SHAPE = (3, 3, 2)
DN_OUTPUT_SIZE = 9

class TicTacToeGame:
    def __init__(self):
        pyxel.init(240, 240)
        pyxel.window_title("〇×ゲーム")  # 代わりにタイトル設定
        
        # Numpyからモデルの重みを読み込む
        weights = np.load('./11_7_best_model_weights.npy', allow_pickle=True)
        
        # モデルを構築して重みをセットする（モデル構造に合わせて修正）
        self.model = self.build_model()
        self.model.set_weights(weights)
        
        self.state = State()
        self.next_action = pv_mcts_action(self.model, 0.0)
        self.game_over = False  # ゲームが終了したかどうかのフラグ
        self.game_over_frame = 0  # ゲーム終了時のフレーム数を記録
        pyxel.run(self.update, self.draw)

    def build_model(self):
        # 入力層
        input_data = np.zeros(DN_INPUT_SHAPE)  # ダミーデータで構造を再現
        # 畳み込み層
        x = conv(DN_FILTERS)(input_data)
        x = batch_norm()(x)
        x = relu(x)
        
        # 残差ブロック
        for _ in range(DN_RESIDUAL_NUM):
            x = residual_block(x)
        
        # プーリング層
        x = global_avg_pool(x)
        
        # 出力層
        policy_output = dense(DN_OUTPUT_SIZE, activation="softmax")(x)
        value_output = dense(1, activation="tanh")(x)
        
        return [policy_output, value_output]
        
    # ビルド関数で構築したモデルにNumpy重みを適用
    model_weights = np.load('./11_7_best_model_weights.npy', allow_pickle=True)
    model = build_model()
    
    # モデルの重みをセットするコード部分（仮の例）
    for layer in self.model.layers:
        if hasattr(layer, 'set_weights'):
            layer.set_weights(weights)  


    def update(self):
        if pyxel.btnp(pyxel.KEY_Q):
            pyxel.quit()

        if self.game_over:
            if pyxel.frame_count - self.game_over_frame > 180:  # 180フレーム後に終了
                pyxel.quit()
            return

        if pyxel.btnp(pyxel.MOUSE_LEFT_BUTTON):
            x = pyxel.mouse_x // 80
            y = pyxel.mouse_y // 80
            action = x + y * 3
            if self.state.is_first_player() and action in self.state.legal_actions():
                self.state = self.state.next(action)
                self.check_game_over()
                if not self.game_over:
                    self.ai_turn()

    def ai_turn(self):
        if not self.state.is_done():
            action = self.next_action(self.state)
            self.state = self.state.next(action)
            self.check_game_over()

    def check_game_over(self):
        # ゲーム終了を確認
        if self.state.is_done():
            self.game_over = True
            self.game_over_frame = pyxel.frame_count  # 現在のフレーム数を記録
            print("Game Over!!!")  # ターミナルに表示

    def draw(self):
        pyxel.cls(0)
        for i in range(3):
            pyxel.line(i * 80, 0, i * 80, 240, 7)  # 白色で縦線
            pyxel.line(0, i * 80, 240, i * 80, 7)  # 白色で横線

        for i in range(9):
            x = (i % 3) * 80 + 20
            y = (i // 3) * 80 + 20
            if self.state.pieces[i] == 1:
                pyxel.circ(x + 20, y + 20, 20, 8)  # 赤色の丸
            elif self.state.enemy_pieces[i] == 1:
                pyxel.line(x, y, x + 40, y + 40, 6)  # 青色の斜線
                pyxel.line(x + 40, y, x, y + 40, 6)  # 青色の斜線

        # ゲームが終了した場合のメッセージ表示
        if self.game_over:
            pyxel.text(70, 110, "Game Over!!!", 7)  # 白色で終了メッセージを表示

TicTacToeGame()
