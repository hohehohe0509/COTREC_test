# COTREC_test

資料集中item的ID(entity ID)需要從1開始，relation ID要從0開始
在kg.txt裡存在的三元組為單向的，相當於一個有向圖（需在程式碼轉換成無向圖)

目前的session切分方式為：
1. 根據userID分群
2. 再來根據時間戳做排序
3. 如果是同一個userID且時間不超過1天(+86400)則為同一個session
