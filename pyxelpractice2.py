import pyxel
from game import State
from pv_mcts import pv_mcts_action
from tensorflow.keras.models import load_model

class TicTacToeGame:
    def __init__(self):
        pyxel.init(240, 240, caption="〇×ゲーム")
        self.model = load_model('./model/best.h5')
        self.state = State()
        self.next_action = pv_mcts_action(self.model, 0.0)
        self.game_over = False  # ゲームが終了したかどうかのフラグ
        self.game_over_frame = 0  # ゲーム終了時のフレーム数を記録
        pyxel.run(self.update, self.draw)

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
            print("Game Over!!!")#ターミナルに表示

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
