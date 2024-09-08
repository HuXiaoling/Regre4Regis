## Regression for registration and the uncertainty

### Two-stage experimental results

#### Pre-training experimental results without uncertainty branch

| Setting | regress loss (l1) | mask loss | uncer loss | mask | x | y | z | States |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
|   5 channels  | 1 (0.0147)    | 0.5 (-0.7352) | ---   | &check;   | &check;   | &check;   | &check;   | Yes   |

#### Pre-training experimental results by using l2 regression loss

| Setting | regress loss (l2) | mask loss | uncer loss | mask | x | y | z | States |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 8 channels    | 1 (0.0363)    | 0.01 (-0.6640)    | --- (3.7952)  | &check;   | &check;   | &check;   | &check;   | Yes   |
| 8 channels    | 1 (0.0410)    | 0.05 (-0.7070)    | --- (5.1415)  | &check;   | &check;   | &check;   | &check;   | Yes   |
| 8 channels    | 1 (0.0254)    | 0.1  (-0.7179)    | --- (2.0328)  | &check;   | &check;   | &check;   | &check;   | Yes   |
| 8 channels    | 1 (0.0421)    | 0.5  (-0.7223)    | --- (6.4939)  | &check;   | &check;   | &check;   | &check;   | Yes   |
| 6 channels    | 1 (0.0443)    | 0.01 (-0.6363)    | --- (5.3125)  | &check;   | &check;   | &check;   | &check;   | Yes   |
| 6 channels    | 1 (0.0493)    | 0.05 (-0.6992)    | --- (5.7710)  | &check;   | &check;   | &check;   | &check;   | Yes   |
| 6 channels    | 1 (0.0235)    | 0.1  (-0.7237)    | --- (1.1944)  | &check;   | &check;   | &check;   | &check;   | Yes   |
| 6 channels    | 1 (0.0298)    | 0.5  (-0.7256)    | --- (1.4838)  | &check;   | &check;   | &check;   | &check;   | Yes   |

#### Finetune experimental results ($\lambda_{mask} = 0.5$)

| Setting | regress loss (l2) | mask loss | uncer loss | mask | x | y | z | States |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 8 channels    | 1 (0.1088)    | 0.5 (-0.7209) | Gaussian  0.1  (2.9452)   | &check;   | &check;   | &check;   |           | No    |
| 8 channels    | 1 (0.1081)    | 0.5 (-0.7045) | Gaussian  0.5  (2.9571)   | &check;   | &check;   | &check;   |           | No    |
| 8 channels    | 1 (0.2966)    | 0.5 (-0.7280) | Laplacian 0.01 (22.3965)  |           |           |           | &check;   | No    |
| 8 channels    | 1 (0.0244)    | 0.5 (-0.7107) | Laplacian 0.05 (1.2930)   | &check;   | &check;   | &check;   | &check;   | Yes   |
| 8 channels    | 1 (0.0248)    | 0.5 (-0.6961) | Laplacian 0.1  (1.1233)   | &check;   | &check;   | &check;   | &check;   | Yes   |
| 8 channels    | 1 (0.0227)    | 0.5 (-0.5428) | Laplacian 0.5  (0.8866)   | &check;   | &check;   | &check;   | &check;   | Yes   |
| 6 channels    | 1 (0.2925)    | 0.5 (-0.7231) | Gaussian  0.01 (3.4299)   |           |           |           |           | No    |
| 6 channels    | 1 (0.0279)    | 0.5 (-0.7230) | Gaussian  0.05 (0.9441)   | &check;   | &check;   | &check;   | &check;   | Yes   |
| 6 channels    | 1 (0.0262)    | 0.5 (-0.7177) | Gaussian  0.1  (0.8739)   | &check;   | &check;   | &check;   | &check;   | Yes   |
| 6 channels    | 1 (0.0262)    | 0.5 (-0.6996) | Gaussian  0.5  (0.8722)   | &check;   | &check;   | &check;   | &check;   | Yes   |
| 6 channels    | 1 (0.2881)    | 0.5 (-0.7168) | Laplacian 0.01 (8.6877)   |           |           |           |           | No    |
| 6 channels    | 1 (0.0243)    | 0.5 (-0.7212) | Laplacian 0.05 (0.9008)   | &check;   | &check;   | &check;   | &check;   | Yes   |
| 6 channels    | 1 (0.0246)    | 0.5 (-0.7174) | Laplacian 0.1  (0.9362)   | &check;   | &check;   | &check;   | &check;   | Yes   |
| 6 channels    | 1 (0.0243)    | 0.5 (-0.6252) | Laplacian 0.5  (0.9320)   | &check;   | &check;   | &check;   | &check;   | Yes   |

#### Finetune experimental results ($\lambda_{mask} = 0.1$)

| Setting | regress loss (l2) | mask loss | uncer loss | mask | x | y | z | States |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 8 channels    | 1 (0.1177)    | 0.1 (-0.7140) | Gaussian  0.01 (4.9129)   |           |           |           |           | No    |
| 8 channels    | 1 (0.0303)    | 0.1 (-0.6966) | Gaussian  0.05 (2.1743)   | &check;   | &check;   | &check;   | &check;   | Yes   |
| 8 channels    | 1 (0.0344)    | 0.1 (-0.5831) | Gaussian  0.1  (2.2953)   | &check;   | &check;   | &check;   | &check;   | Yes   |
| 8 channels    | 1 (0.0406)    | 0.1 (-0.6007) | Gaussian  0.5  (2.5143)   | &check;   | &check;   | &check;   | &check;   | Yes   |
| 8 channels    | 1 (0.0369)    | 0.1 (-0.7191) | Laplacian 0.01 (3.4481)   | &check;   | &check;   | &check;   | &check;   | Yes   |
| 8 channels    | 1 (0.0223)    | 0.1 (-0.6920) | Laplacian 0.05 (1.3373)   | &check;   | &check;   | &check;   | &check;   | Yes   |
| 8 channels    | 1 (0.0336)    | 0.1 (-0.6450) | Laplacian 0.1  (2.4943)   | &check;   | &check;   | &check;   | &check;   | Yes   |
| 8 channels    | 1 (0.0321)    | 0.1 (-0.5821) | Laplacian 0.5  (2.1881)   | &check;   | &check;   | &check;   | &check;   | Yes   |
| 6 channels    | 1 (0.2113)    | 0.1 (-0.7187) | Gaussian  0.01 (3.1466)   |           |           |           |           | No    |
| 6 channels    | 1 (0.1285)    | 0.1 (-0.7175) | Gaussian  0.05 (2.6510)   |           |           |           |           | No    |
| 6 channels    | 1 (0.0383)    | 0.1 (-0.7029) | Gaussian  0.1  (1.4070)   | &check;   | &check;   | &check;   | &check;   | Yes   |
| 6 channels    | 1 (0.0323)    | 0.1 (-0.6316) | Gaussian  0.5  (1.2556)   | &check;   | &check;   | &check;   | &check;   | Yes   |
| 6 channels    | 1 (0.1316)    | 0.1 (-0.7215) | Laplacian 0.01 (5.0410)   |           |           |           |           | No    |
| 6 channels    | 1 (0.1210)    | 0.1 (-0.7250) | Laplacian 0.05 (5.0588)   |           |           |           |           | No    |
| 6 channels    | 1 (0.0308)    | 0.1 (-0.6987) | Laplacian 0.1  (1.2880)   | &check;   | &check;   | &check;   | &check;   | Yes   |
| 6 channels    | 1 (0.1255)    | 0.1 (-0.7056) | Laplacian 0.5  (5.1822)   |           |           |           |           | No    |

#### Finetune experimental results ($\lambda_{mask} = 0.05$)

| Setting | regress loss (l2) | mask loss | uncer loss | mask | x | y | z | States |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 8 channels    | 1 (0.0370)    | 0.05 (-0.7097)    | Gaussian  0.01 (3.0875)   | &check;   | &check;   | &check;   | &check;   | Yes   |
| 8 channels    | 1 (0.0365)    | 0.05 (-0.6197)    | Gaussian  0.05 (2.6919)   | &check;   | &check;   | &check;   | &check;   | Yes   |
| 8 channels    | 1 (0.0330)    | 0.05 (-0.5606)    | Gaussian  0.1  (2.2717)   | &check;   | &check;   | &check;   | &check;   | Yes   |
| 8 channels    | 1 (0.0394)    | 0.05 (-0.5076)    | Gaussian  0.5  (2.4040)   | &check;   | &check;   | &check;   | &check;   | Yes   |
| 8 channels    | 1 (0.0345)    | 0.05 (-0.7103)    | Laplacian 0.01 (2.5869)   | &check;   | &check;   | &check;   | &check;   | Yes   |
| 8 channels    | 1 (0.0365)    | 0.05 (-0.5714)    | Laplacian 0.05 (2.5743)   | &check;   | &check;   | &check;   | &check;   | Yes   |
| 8 channels    | 1 (0.0333)    | 0.05 (-0.5778)    | Laplacian 0.1  (2.3617)   | &check;   | &check;   | &check;   | &check;   | Yes   |
| 8 channels    | 1 (0.0290)    | 0.05 (-0.4851)    | Laplacian 0.5  (2.0467)   | &check;   | &check;   | &check;   | &check;   | Yes   |
| 6 channels    | 1 (0.0981)    | 0.05 (-0.7091)    | Gaussian  0.01 (2.2814)   |           |           |           |           | No    |
| 6 channels    | 1 (0.0458)    | 0.05 (-0.6455)    | Gaussian  0.05 (1.6408)   | &check;   | &check;   | &check;   | &check;   | Yes   |
| 6 channels    | 1 (0.1151)    | 0.05 (-0.5485)    | Gaussian  0.1  (2.4986)   |           |           |           |           | No    |
| 6 channels    | 1 (0.0373)    | 0.05 (-0.5658)    | Gaussian  0.5  (1.5592)   | &check;   | &check;   | &check;   | &check;   | Yes   |
| 6 channels    | 1 (0.0396)    | 0.05 (-0.7143)    | Laplacian 0.01 (1.6604)   | &check;   | &check;   | &check;   | &check;   | Yes   |
| 6 channels    | 1 (0.1296)    | 0.05 (-0.6691)    | Laplacian 0.05 (5.3738)   |           |           |           |           | No    |
| 6 channels    | 1 (0.2083)    | 0.05 (-0.6752)    | Laplacian 0.1  (6.7715)   |           |           |           |           | No    |
| 6 channels    | 1 (0.0261)    | 0.05 (-0.6091)    | Laplacian 0.5  (1.7034)   | &check;   | &check;   | &check;   | &check;   | Yes   |

#### Finetune experimental results ($\lambda_{mask} = 0.01$)

| Setting | regress loss (l2) | mask loss | uncer loss | mask | x | y | z | States |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 8 channels    | 1 (0.0471)    | 0.01 (-0.5769)    | Gaussian  0.01 (2.9437)   | &check;   | &check;   | &check;   | &check;   | Yes   |
| 8 channels    | 1 (0.0243)    | 0.01 (-0.4585)    | Gaussian  0.05 (0.8756)   | &check;   | &check;   | &check;   | &check;   | Yes   |
| 8 channels    | 1 (0.0352)    | 0.01 (-0.5483)    | Gaussian  0.1  (2.2861)   | &check;   | &check;   | &check;   | &check;   | Yes   |
| 8 channels    | 1 (0.0288)    | 0.01 (-0.1618)    | Gaussian  0.5  (1.2280)   | &check;   | &check;   | &check;   | &check;   | Yes   |
| 8 channels    | 1 (0.0821)    | 0.01 (-0.5235)    | Laplacian 0.01 (7.0968)   |           |           |           |           | No    |
| 8 channels    | 1 (0.0237)    | 0.01 (-0.4981)    | Laplacian 0.05 (0.9300)   | &check;   | &check;   | &check;   | &check;   | Yes   |
| 8 channels    | 1 (0.0338)    | 0.01 (-0.5741)    | Laplacian 0.1  (2.4788)   | &check;   | &check;   | &check;   | &check;   | Yes   |
| 8 channels    | 1 (0.0247)    | 0.01 (-0.1832)    | Laplacian 0.5  (1.2784)   | &check;   | &check;   | &check;   | &check;   | Yes   |
| 6 channels    | 1 (0.0628)    | 0.01 (-0.5945)    | Gaussian  0.01 (1.8581)   |           |           |           |           | No    |
| 6 channels    | 1 (0.0219)    | 0.01 (-0.5944)    | Gaussian  0.05 (0.8058)   | &check;   | &check;   | &check;   | &check;   | Yes   |
| 6 channels    | 1 (0.0438)    | 0.01 (-0.6105)    | Gaussian  0.1  (1.5624)   | &check;   | &check;   | &check;   | &check;   | Yes   |
| 6 channels    | 1 (0.1529)    | 0.01 (-0.1553)    | Gaussian  0.5  (2.5679)   |           |           |           |           | No    |
| 6 channels    | 1 (0.1677)    | 0.01 (-0.6582)    | Laplacian 0.01 (6.8342)   |           |           |           |           | No    |
| 6 channels    | 1 (0.1201)    | 0.01 (-0.6867)    | Laplacian 0.05 (4.8883)   |           |           |           |           | No    |
| 6 channels    | 1 (0.1205)    | 0.01 (-0.6810)    | Laplacian 0.1  (4.2078)   |           |           |           |           | No    |
| 6 channels    | 1 (0.1129)    | 0.01 (-0.5902)    | Laplacian 0.5  (5.1991)   |           |           |           |           | No    |

#### Pre-training experimental results by using l1 regression loss

| Setting | regress loss (l1) | mask loss | uncer loss | mask | x | y | z | States |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 8 channels    | 1 (0.0168)    | 0.01 (-0.6171)    | --- (0.7625)  | &check;   | &check;   | &check;   | &check;   | Yes   |
| 8 channels    | 1 (0.1059)    | 0.05 (-0.7199)    | --- (54.7341) |           |           |           |           | No    |
| 8 channels    | 1 (0.1055)    | 0.1  (-0.7243)    | --- (48.8682) |           |           |           |           | No    |
| 8 channels    | 1 (0.1059)    | 0.5  (-0.7340)    | --- (61.4771) |           |           |           |           | No    |
| 6 channels    | 1 (0.1053)    | 0.01 (-0.6972)    | --- (56.8711) |           |           |           |           | No    |
| 6 channels    | 1 (0.1061)    | 0.05 (-0.7171)    | --- (61.7876) |           |           |           |           | No    |
| 6 channels    | 1 (0.0156)    | 0.1  (-0.7227)    | --- (0.6635)  | &check;   | &check;   | &check;   | &check;   | Yes   |
| 6 channels    | 1 (0.0276)    | 0.5  (-0.7320)    | --- (2.0480)  | &check;   | &check;   | &check;   | &check;   | yes   |

#### Finetune experimental results ($\lambda_{mask} = 0.5$)

| Setting | regress loss (l1) | mask loss | uncer loss | mask | x | y | z | States |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 6 channels    | 1 (0.0674)    | 0.5 (-0.7313) | Gaussian  0.01 (1.6465)   | &check;   | &check;   | &check;   | &check;   | yes   |
| 6 channels    | 1 (0.0592)    | 0.5 (-0.7298) | Gaussian  0.05 (1.1923)   | &check;   | &check;   | &check;   | &check;   | yes   |
| 6 channels    | 1 (0.0602)    | 0.5 (-0.7234) | Gaussian  0.1  (1.1049)   | &check;   | &check;   | &check;   | &check;   | yes   |
| 6 channels    | 1 (0.0367)    | 0.5 (-0.6771) | Gaussian  0.5  (1.0768)   | &check;   | &check;   | &check;   | &check;   | yes   |
| 6 channels    | 1 (0.0471)    | 0.5 (-0.7314) | Laplacian 0.01 (2.1758)   | &check;   | &check;   | &check;   | &check;   | yes   |
| 6 channels    | 1 (0.0442)    | 0.5 (-0.7289) | Laplacian 0.05 (1.5562)   | &check;   | &check;   | &check;   | &check;   | yes   |
| 6 channels    | 1 (0.0272)    | 0.5 (-0.7222) | Laplacian 0.1  (1.1024)   | &check;   | &check;   | &check;   | &check;   | yes   |
| 6 channels    | 1 (0.0255)    | 0.5 (-0.6958) | Laplacian 0.5  (1.0686)   | &check;   | &check;   | &check;   | &check;   | yes   |

#### Finetune experimental results ($\lambda_{mask} = 0.1$)

| Setting | regress loss (l1) | mask loss | uncer loss | mask | x | y | z | States |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 6 channels    | 1 (0.0259)    | 0.1 (-0.7160) | Gaussian  0.01 (1.0434)   | &check;   | &check;   | &check;   | &check;   | yes   |
| 6 channels    | 1 (0.0237)    | 0.1 (-0.7106) | Gaussian  0.05 (0.7888)   | &check;   | &check;   | &check;   | &check;   | yes   |
| 6 channels    | 1 (0.0260)    | 0.1 (-0.6655) | Gaussian  0.1  (0.9351)   | &check;   | &check;   | &check;   | &check;   | yes   |
| 6 channels    | 1 (0.0292)    | 0.1 (-0.4767) | Gaussian  0.5  (0.9934)   | &check;   | &check;   | &check;   |           | no    |
| 6 channels    | 1 (0.0218)    | 0.1 (-0.7222) | Laplacian 0.01 (1.0243)   | &check;   | &check;   | &check;   | &check;   | yes   |
| 6 channels    | 1 (0.0236)    | 0.1 (-0.7013) | Laplacian 0.05 (0.8906)   | &check;   | &check;   | &check;   | &check;   | yes   |
| 6 channels    | 1 (0.0237)    | 0.1 (-0.6962) | Laplacian 0.1  (0.9584)   | &check;   | &check;   | &check;   | &check;   | yes   |
| 6 channels    | 1 (0.0215)    | 0.1 (-0.6579) | Laplacian 0.5  (0.7976)   | &check;   | &check;   | &check;   | &check;   | yes   |

#### Finetune experimental results ($\lambda_{mask} = 0.01$)

| Setting | regress loss (l1) | mask loss | uncer loss | mask | x | y | z | States |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 8 channels    | 1 (0.0232)    | 0.01 (-0.6151)    | Gaussian  0.01 (0.7764)   | &check;   | &check;   | &check;   | &check;   | yes   |
| 8 channels    | 1 (0.0261)    | 0.01 (-0.4961)    | Gaussian  0.05 (0.9828)   | &check;   | &check;   | &check;   |           | no    |
| 8 channels    | 1 (0.0210)    | 0.01 (-0.4080)    | Gaussian  0.1  (0.6204)   | &check;   | &check;   | &check;   |           | no    |
| 8 channels    | 1 (0.0226)    | 0.01 (-0.0721)    | Gaussian  0.5  (0.7397)   | &check;   | &check;   | &check;   |           | no    |
| 8 channels    | 1 (0.0207)    | 0.01 (-0.6176)    | Laplacian 0.01 (0.7567)   | &check;   | &check;   | &check;   | &check;   | yes   |
| 8 channels    | 1 (0.0258)    | 0.01 (-0.5117)    | Laplacian 0.05 (1.2250)   | &check;   | &check;   | &check;   |           | no    |
| 8 channels    | 1 (0.0242)    | 0.01 (-0.4279)    | Laplacian 0.1  (1.1487)   | &check;   | &check;   | &check;   |           | no    |
| 8 channels    | 1 (0.0257)    | 0.01 (-0.3149)    | Laplacian 0.5  (1.1498)   | &check;   | &check;   | &check;   |           | no    |

### End2end experimental results

#### Finetune experimental results with both linear (least square) and non-linear (Bspline) deformation for $\lambda_{uncer} = 0.1s

| Setting | regress loss (l2) | mask loss | uncer loss | seg loss | mask | x | y | z | States |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 8 channels    | 1 | 0.5   | Laplacian 0.1     | 2     |   |   |   |   | Done      |
| 8 channels    | 1 | 0.5   | Laplacian 0.1     | 5     |   |   |   |   | Done      |
| 8 channels    | 1 | 0.5   | Laplacian 0.1     | 10    |   |   |   |   | Done      |
| 6 channels    | 1 | 0.5   | Laplacian 0.1     | 2     |   |   |   |   | Done      |
| 6 channels    | 1 | 0.5   | Laplacian 0.1     | 5     |   |   |   |   | Done      |
| 6 channels    | 1 | 0.5   | Laplacian 0.1     | 10    |   |   |   |   | Done      |

#### Finetune experimental results with both linear (least square) and non-linear (Bspline) deformation for $\lambda_{uncer} = 0.05s

| Setting | regress loss (l2) | mask loss | uncer loss | seg loss | mask | x | y | z | States |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 8 channels    | 1 | 0.5   | Laplacian 0.05    | 2     |   |   |   |   | Done      |
| 8 channels    | 1 | 0.5   | Laplacian 0.05    | 5     |   |   |   |   | Done      |
| 8 channels    | 1 | 0.5   | Laplacian 0.05    | 10    |   |   |   |   | Done      |
| 6 channels    | 1 | 0.5   | Laplacian 0.05    | 2     |   |   |   |   | Running   |
| 6 channels    | 1 | 0.5   | Laplacian 0.05    | 5     |   |   |   |   | Done      |
| 6 channels    | 1 | 0.5   | Laplacian 0.05    | 10    |   |   |   |   | Done      |

#### Finetune experimental results with both linear (least square) and non-linear (demon) deformations for $\lambda_{uncer} = 0.1$

| Setting | regress loss (l2) | mask loss | uncer loss | seg loss | mask | x | y | z | States |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 8 channels    | 1    | 0.5 | Laplacian 0.1   | 0.5    |  |  |  |  | Done      |
| 8 channels    | 1    | 0.5 | Laplacian 0.1   | 1      |  |  |  |  | Done      |
| 8 channels    | 1    | 0.5 | Laplacian 0.1   | 2      |  |  |  |  | Done      |
| 8 channels    | 1    | 0.5 | Laplacian 0.1   | 5      |  |  |  |  | Done      |
| 8 channels    | 1    | 0.5 | Laplacian 0.1   | 10     |  |  |  |  | Running   |
| 6 channels    | 1    | 0.5 | Laplacian 0.1   | 0.5    |  |  |  |  | Done      |
| 6 channels    | 1    | 0.5 | Laplacian 0.1   | 1      |  |  |  |  | Done      |
| 6 channels    | 1    | 0.5 | Laplacian 0.1   | 2      |  |  |  |  | Done      |
| 6 channels    | 1    | 0.5 | Laplacian 0.1   | 5      |  |  |  |  | Done      |
| 6 channels    | 1    | 0.5 | Laplacian 0.1   | 10     |  |  |  |  | Running   |

#### Finetune experimental results with both linear (least square) and non-linear (demon) deformations for $\lambda_{uncer} = 0.05$

| Setting | regress loss (l2) | mask loss | uncer loss | seg loss | mask | x | y | z | States |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 8 channels    | 1    | 0.5 | Laplacian 0.05   | 2      |  |  |  |  | Running  |
| 8 channels    | 1    | 0.5 | Laplacian 0.05   | 5      |  |  |  |  | Running  |
| 8 channels    | 1    | 0.5 | Laplacian 0.05   | 10     |  |  |  |  | Running  |
| 6 channels    | 1    | 0.5 | Laplacian 0.05   | 2      |  |  |  |  | Running  |
| 6 channels    | 1    | 0.5 | Laplacian 0.05   | 5      |  |  |  |  | Running  |
| 6 channels    | 1    | 0.5 | Laplacian 0.05   | 10     |  |  |  |  | Running  |

#### Finetune experimental results with both linear (least square) and non-linear (demon) deformations for $\lambda_{uncer} = 0.1$: using gaussian distribution

| Setting | regress loss (l2) | mask loss | uncer loss | seg loss | mask | x | y | z | States |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 8 channels    | 1    | 0.5 | Gaussian 0.05    | 5     |  |  |  |  | N/A       |
| 8 channels    | 1    | 0.5 | Gaussian 0.05    | 10    |  |  |  |  | N/A       |
| 8 channels    | 1    | 0.5 | Gaussian 0.1     | 5     |  |  |  |  | N/A       |
| 8 channels    | 1    | 0.5 | Gaussian 0.1     | 10    |  |  |  |  | N/A       |
| 6 channels    | 1    | 0.5 | Gaussian 0.05    | 5     |  |  |  |  | To run    |
| 6 channels    | 1    | 0.5 | Gaussian 0.05    | 10    |  |  |  |  | To run    |
| 6 channels    | 1    | 0.5 | Gaussian 0.1     | 5     |  |  |  |  | To run    |
| 6 channels    | 1    | 0.5 | Gaussian 0.1     | 10    |  |  |  |  | To run    |

#### Finetune experimental results with both linear (least square) and non-linear (demon) deformations for $\lambda_{uncer} = 0.1$: using l1 regression loss

| Setting | regress loss (l1) | mask loss | uncer loss | seg loss | mask | x | y | z | States |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 8 channels    | 1    | 0.5 | Laplacian 0.05   | 5     |  |  |  |  | N/A       |
| 8 channels    | 1    | 0.5 | Laplacian 0.05   | 10    |  |  |  |  | N/A       |
| 8 channels    | 1    | 0.5 | Laplacian 0.1    | 5     |  |  |  |  | N/A       |
| 8 channels    | 1    | 0.5 | Laplacian 0.1    | 10    |  |  |  |  | N/A       |
| 6 channels    | 1    | 0.5 | Laplacian 0.05   | 5     |  |  |  |  | Running   |
| 6 channels    | 1    | 0.5 | Laplacian 0.05   | 10    |  |  |  |  | To run    |
| 6 channels    | 1    | 0.5 | Laplacian 0.1    | 5     |  |  |  |  | To run    |
| 6 channels    | 1    | 0.5 | Laplacian 0.1    | 10    |  |  |  |  | To run    |

### End2end experimental results (incorporating uncertainty during training)

#### Finetune experimental results with both linear (least square + uncertainty) and non-linear (Bspline) deformations for $\lambda_{uncer} = 0.1$; finetune from non-linear models

| Setting | regress loss (l2) | mask loss | uncer loss | seg loss | mask | x | y | z | States |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 8 channels    | 1    | 0.5 | Laplacian 0.1   | 1      |  |  |  |  | Done  |
| 8 channels    | 1    | 0.5 | Laplacian 0.1   | 2      |  |  |  |  | Done  |
| 8 channels    | 1    | 0.5 | Laplacian 0.1   | 5      |  |  |  |  | Done  |
| 8 channels    | 1    | 0.5 | Laplacian 0.1   | 10     |  |  |  |  | Done  |
| 8 channels    | 1    | 0.5 | Laplacian 0.1   | 15     |  |  |  |  | Done  |
| 6 channels    | 1    | 0.5 | Laplacian 0.1   | 1      |  |  |  |  | Done  |
| 6 channels    | 1    | 0.5 | Laplacian 0.1   | 2      |  |  |  |  | Done  |
| 6 channels    | 1    | 0.5 | Laplacian 0.1   | 5      |  |  |  |  | Done  |
| 6 channels    | 1    | 0.5 | Laplacian 0.1   | 10     |  |  |  |  | Done  |
| 6 channels    | 1    | 0.5 | Laplacian 0.1   | 15     |  |  |  |  | Done  |