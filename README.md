Face-Recognize

使用方法與套件:
我們採用keras 套件實作CNN 模組，用自定義的陣列方式讀取圖
片當作訓練資料。
步驟:
1. 讀檔:
使用image.open()讀檔，讀入的圖片資料會以陣列型
態保存，然後因為每張圖片的大小不一樣所以再進
行resize()，使每張圖片的大小都一樣方便陣列儲
存，也更好載入模組。
第33 行:目的是將RGB 的圖案轉成灰階影像。
第34 行:這裡是將array 的數值都統一處理成0<val<1
因為少了05,09 所以我們只好將迴圈設計成50 人13
個每人。最後用face_data 紀錄。(如下圖)
![Alt text](https://imgur.com/viJCU3S.jpg)

載入完的陣列可以直接拿來用ㄝ，當作Ｘ作為數據組，而Y則是對應每個Ｘ的label，所以就簡單地利用迴圈載入對應的陣列中
![Alt text](https://imgur.com/4twE3UO.jpg)

3. 建立模組訓練
接下來就是利用 接下來就是利用 keras套件，建立卷基層。
調整 learning rate、loss function、optimizer function。
最後用 model.fit(X,y)載入訓練數據，在完成了。
![Alt text](https://imgur.com/QIGCPWs.jpg)