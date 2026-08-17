#支座与主梁
rigidLink beam 202 286
rigidLink beam 202 287
rigidLink beam 202 288
rigidLink beam 202 289
rigidLink beam 280 284
rigidLink beam 280 285
rigidLink beam 280 290
rigidLink beam 280 291

#墩底与承台顶
rigidLink beam 105 108
rigidLink beam 105 107
rigidLink beam 106 109
rigidLink beam 106 110

#墩顶与桥梁
rigidLink beam  143 221
rigidLink beam  144 223
rigidLink beam  145 259
rigidLink beam  146 261

fix 91 1 1 1 1 1 1
fix 92 1 1 1 1 1 1
fix 93 1 1 1 1 1 1
fix 94 1 1 1 1 1 1
fix 95 1 1 1 1 1 1
fix 96 1 1 1 1 1 1
fix 97 1 1 1 1 1 1
fix 98 1 1 1 1 1 1
fix 99 1 1 1 1 1 1
fix 100 1 1 1 1 1 1
fix 101 1 1 1 1 1 1
fix 102 1 1 1 1 1 1
fix 103 1 1 1 1 1 1
fix 104 1 1 1 1 1 1

fix 11284 1 1 1 1 1 1
fix 11285 1 1 1 1 1 1
fix 11290 1 1 1 1 1 1
fix 11291 1 1 1 1 1 1
fix 11286 1 1 1 1 1 1
fix 11287 1 1 1 1 1 1
fix 11288 1 1 1 1 1 1
fix 11289 1 1 1 1 1 1
#支座
# ============================================================
# 右端支座
# ============================================================

# SX：X/Y活动，Z约束
element zeroLength 8001 11284 284 -mat 101 -dir 3
element zeroLength 8002 11290 290 -mat 101 -dir 3

# DX：X活动，Y/Z约束
element zeroLength 8003 11285 285 -mat 103 101 -dir 2 3
element zeroLength 8004 11291 291 -mat 103 101 -dir 2 3


# ============================================================
# 左端支座
# ============================================================

# SX：X/Y活动，Z约束
element zeroLength 8005 11286 286 -mat 101 -dir 3
element zeroLength 8006 11288 288 -mat 101 -dir 3

# DX：X活动，Y/Z约束
element zeroLength 8007 11287 287 -mat 103 101 -dir 2 3
element zeroLength 8008 11289 289 -mat 103 101 -dir 2 3




