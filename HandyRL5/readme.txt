対人戦用dominionの取扱説明書
(2024/6/26更新)
・動作環境
HandyRLが動かせる環境。詳細および必要なライブラリの導入は以下のHandyRL公式githubを参照
https://github.com/DeNA/HandyRL

・基本的な使い方
階層 ~/HandyRL5　まで移動し、以下のようなコマンドを打つことで起動する
python main.py --eval models/mix60000.pth:models/mix60000.pth:models/mix60000.pth:self 1 1
左の二つの数はそれぞれ　対戦回数/セット数を表す。対人戦ならば 1 1 で問題ない
models/mix60000.pthは対戦相手のモデル指定。models内にある別のモデルに置き換えることもできる。
加えて組み込みのアルゴリズムに変えることも可能。この場合、前後にmodels/~.pthをつける必要はない。下がその一覧
------------------------------
str  単純ステロ
str_m   番兵型
str_s   鍛冶屋型
str_w   魔女型
str_b   盗賊型
str_p   密猟者型
------------------------------
※各モデル/アルゴリズムは:で区切り、四人分指定する必要がある。
selfを一人以上指定しなければ対人モードにならない。

使用するカードセットの選択
デフォルトでは完全ランダム状態になってますが、自分で指定することも可能。
その場合、dominion.py内部の506行目、self.card_listの設定を変更してください。
self.setx (xは1~7)：465行目あたりに定義してあるセット。詳しくは私の卒論参照。
self.settest：検証用に作ったやつ。自由に変えてもらって大丈夫。
self.randomset()：デフォルト設定。完全にランダムにセットを選ぶ。

・表示について
一応どのプレイヤーが何を使用したか、何を購入したかはログのように表示されます。ただ、下記の選択時の表示に押し流されることが多いので
コマンドラインは広めに表示しておくことを推奨します。

・操作について
購入選択を行うときの画面
------------------------------
your id=0    ①

score
player0:9 player1:22 player2:10 player3:15    ②
field state
your state[action, buy, +coin]=[1, 1, 0]　　　③
 Copper:32
 Silver:33
 Gold:23
 Estate:6
 Duchy:6
 Province:12
 Curse:30
 Cellar:10
 Chapel:10
 Vassal:0
 Harbinger:10
 Moneylender:10
 TroneRoom:10
 Poacher:8
 Militia:5
 Gardens:0
 Artisan:9　　　　　　④

[1, 1, 0]　　　　⑤
coins=5　　　　⑥
 0:Copper:32 1:Silver:33 3:Estate:6 4:Duchy:6 6:Curse:30 7:Cellar:10 8:Chapel:10 13:Harbinger:10 17:Moneylender:10 18:TroneRoom:10 19:Poacher:8 20:Militia:5 33:None　　　⑦
input your choice
------------------------------
①：自分のid。実行ごとに変わる
②：現在の各playerの勝利点
③：自分の現在の残りアクション権、購入権、追加金量
④：盤面に残っているカードの名前とその残量
⑤：現在の自分の手札
⑥：自分が使える金量
⑦：選べる選択肢の　番号：名前：残量
上記のような表示の後に、自分が購入するカードを番号で選ぶ。
※必ず番号を半角数字で入力してください。

アクション関係の選択
例１　単一の選択
------------------------------
ActDecision (1 actions, 1 buys, +0 coins)　　①
your hands
 Copper Gold Copper Estate Militia　　②
choices
 33:None 20:Militia　　③
input your choice
------------------------------
①：選択する内容と必要な情報の表示
②：現在の自分の手札
③：選択肢
例２　複数枚の選択ができる
------------------------------
DiscardDecision(hands:[2, 0, 3, 0, 0], min:2, max:2)
choices
 0:Copper 0:Copper 0:Copper 2:Gold 3:Estate
input your choices separated by a space
------------------------------
例１と概ね一緒だが、複数枚選ぶときには一番下の指示のように半角スペースで区切って入力する。

・固有の細かい仕様
DiscardDecision（手札からカードを特定枚数捨てる）,TrashDecision（手札から特定枚数を廃棄する）について
minは選択最小数、maxは選択最大数を表す。
minが0であり、どのカードも選択しない場合には何も入力せずにenterを押すことで処理が可能（他ではバグるのでやらないでください）
何もしないという選択肢が取れる場合、基本的にNoneが選択肢に現れる（はず）
現れない場合は基本的に強制選択なので選択肢から選んでください

LibraryDecision（図書室使用時にアクションカードを脇に置くか手札に加えるかの判断）について
表示がわかりにくいが、Noneを選んだら手札に加える。カードを選んだら脇に置く。

一部の廃棄して~する、公開して~するという処理について
処理の関係上一旦公開ゾーンに置いてから、改めて本来の処理を行うという動きになっている。そのため表示がOpenDecisionになっているが、本来の処理は違う。

購入を一ターンに複数回行った場合、③の追加金量がマイナスになることがあるが、これはバグではない。
使用済み金量を追加金量から引くという処理の結果そういう表示になることがある。

一番最初に手札が無い謎の虚無ターンが存在しますが、これは初期デッキをランダム化するための処理であるため細かいことは考えずにNoneを選択して流してください。

注意
バグ回避用のシステムは最低限しか組み込んでないため、変な入力はやめてください。

お願い
一応バグがなさそうなことは確認してますが、まだ変なバグが残ってる可能性があるので見つけたら教えてください。
テストプレイが現状自分だけのため、ユーザーインターフェースがとんでもない不親切仕様になってる可能性があります。何か意見やアドバイス、質問などあれば遠慮なく連絡ください。
組み込むのが面倒だったため、カードの効果説明がどこにもありません。申し訳ないですが、自分で効果を調べるなり覚えるなりしてください。カードリストがあるサイトを乗っけておきます。
https://dominion758.jp/card-list/basic/
