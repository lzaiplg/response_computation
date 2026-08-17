geomTransf PDelta 100 1 0 0;  # 承台转换
geomTransf PDelta 200 0 -1 0; # 桥墩转换
geomTransf PDelta 300 0 0 1;  # 箱梁转换



element elasticBeamColumn 91 105 95 $A25 $E3 $G3 $J25 $Iy25 $Iz25 100
element elasticBeamColumn 92 106 96 $A25 $E3 $G3 $J25 $Iy25 $Iz25 100
element nonlinearBeamColumn	250 143	139	$numIntgrPts1	23	200	-iter	$maxIters	$tol
element nonlinearBeamColumn	251	144	140	$numIntgrPts1	24	200	-iter	$maxIters	$tol
element nonlinearBeamColumn	252	146	142	$numIntgrPts1	24	200	-iter	$maxIters	$tol
element nonlinearBeamColumn	253	145	141	$numIntgrPts1	23	200	-iter	$maxIters	$tol
element nonlinearBeamColumn	254	139	135	$numIntgrPts1	23	200	-iter	$maxIters	$tol
element nonlinearBeamColumn	255	135	131	$numIntgrPts1	23	200	-iter	$maxIters	$tol
element nonlinearBeamColumn	256	131	127	$numIntgrPts1	23	200	-iter	$maxIters	$tol
element nonlinearBeamColumn	257	127	123	$numIntgrPts1	23	200	-iter	$maxIters	$tol
element nonlinearBeamColumn	258	123	119	$numIntgrPts1	23	200	-iter	$maxIters	$tol
element nonlinearBeamColumn	259	119	115	$numIntgrPts1	23	200	-iter	$maxIters	$tol
element nonlinearBeamColumn	260	115	111	$numIntgrPts1	23	200	-iter	$maxIters	$tol
element nonlinearBeamColumn	261	111	107	$numIntgrPts1	23	200	-iter	$maxIters	$tol
element nonlinearBeamColumn	262	140	136	$numIntgrPts1	24	200	-iter	$maxIters	$tol
element nonlinearBeamColumn	263	136	132	$numIntgrPts1	24	200	-iter	$maxIters	$tol
element nonlinearBeamColumn	264	132	128	$numIntgrPts1	24	200	-iter	$maxIters	$tol
element nonlinearBeamColumn	265	128	124	$numIntgrPts1	24	200	-iter	$maxIters	$tol
element nonlinearBeamColumn	266	124	120	$numIntgrPts1	24	200	-iter	$maxIters	$tol
element nonlinearBeamColumn	267	120	116	$numIntgrPts1	24	200	-iter	$maxIters	$tol
element nonlinearBeamColumn	268	116	112	$numIntgrPts1	24	200	-iter	$maxIters	$tol
element nonlinearBeamColumn	269	112	108	$numIntgrPts1	24	200	-iter	$maxIters	$tol
element nonlinearBeamColumn	270	142	138	$numIntgrPts1	24	200	-iter	$maxIters	$tol
element nonlinearBeamColumn	271	138	134	$numIntgrPts1	24	200	-iter	$maxIters	$tol
element nonlinearBeamColumn	272	134	130	$numIntgrPts1	24	200	-iter	$maxIters	$tol
element nonlinearBeamColumn	273	130	126	$numIntgrPts1	24	200	-iter	$maxIters	$tol
element nonlinearBeamColumn	274	126	122	$numIntgrPts1	24	200	-iter	$maxIters	$tol
element nonlinearBeamColumn	275	122	118	$numIntgrPts1	24	200	-iter	$maxIters	$tol
element nonlinearBeamColumn	276	118	114	$numIntgrPts1	24	200	-iter	$maxIters	$tol
element nonlinearBeamColumn	277	114	110	$numIntgrPts1	24	200	-iter	$maxIters	$tol
element nonlinearBeamColumn	278	141	137	$numIntgrPts1	23	200	-iter	$maxIters	$tol
element nonlinearBeamColumn	279	137	133	$numIntgrPts1	23	200	-iter	$maxIters	$tol
element nonlinearBeamColumn	280	133	129	$numIntgrPts1	23	200	-iter	$maxIters	$tol
element nonlinearBeamColumn	281	129	125	$numIntgrPts1	23	200	-iter	$maxIters	$tol
element nonlinearBeamColumn	282	125	121	$numIntgrPts1	23	200	-iter	$maxIters	$tol
element nonlinearBeamColumn	283	121	117	$numIntgrPts1	23	200	-iter	$maxIters	$tol
element nonlinearBeamColumn	284	117	113	$numIntgrPts1	23	200	-iter	$maxIters	$tol
element nonlinearBeamColumn	285	113	109	$numIntgrPts1	23	200	-iter	$maxIters	$tol
element	elasticBeamColumn	125	260	261	$A1	$E1	$G1	$J1	$Iy1	$Iz1	300
element	elasticBeamColumn	126	261	262	$A2	$E1	$G1	$J2	$Iy2	$Iz2	300
element	elasticBeamColumn	127	262	263	$A3	$E1	$G1	$J3	$Iy3	$Iz3	300
element	elasticBeamColumn	128	263	264	$A4	$E1	$G1	$J4	$Iy4	$Iz4	300
element	elasticBeamColumn	129	264	265	$A5	$E1	$G1	$J5	$Iy5	$Iz5	300
element	elasticBeamColumn	130	265	266	$A6	$E1	$G1	$J6	$Iy6	$Iz6	300
element	elasticBeamColumn	131	266	267	$A7	$E1	$G1	$J7	$Iy7	$Iz7	300
element	elasticBeamColumn	132	267	268	$A8	$E1	$G1	$J8	$Iy8	$Iz8	300
element	elasticBeamColumn	133	268	269	$A9	$E1	$G1	$J9	$Iy9	$Iz9	300
element	elasticBeamColumn	134	269	270	$A10	$E1	$G1	$J10	$Iy10	$Iz10	300
element	elasticBeamColumn	135	270	271	$A11	$E1	$G1	$J11	$Iy11	$Iz11	300
element	elasticBeamColumn	136	271	272	$A12	$E1	$G1	$J12	$Iy12	$Iz12	300
element	elasticBeamColumn	137	272	273	$A13	$E1	$G1	$J13	$Iy13	$Iz13	300
element	elasticBeamColumn	138	273	274	$A14	$E1	$G1	$J14	$Iy14	$Iz14	300
element	elasticBeamColumn	139	274	275	$A15	$E1	$G1	$J15	$Iy15	$Iz15	300
element	elasticBeamColumn	140	275	276	$A16	$E1	$G1	$J16	$Iy16	$Iz16	300
element	elasticBeamColumn	141	276	277	$A17	$E1	$G1	$J17	$Iy17	$Iz17	300
element	elasticBeamColumn	142	277	278	$A18	$E1	$G1	$J18	$Iy18	$Iz18	300
element	elasticBeamColumn	143	279	278	$A19	$E1	$G1	$J19	$Iy19	$Iz19	300
element	elasticBeamColumn	144	280	279	$A20	$E1	$G1	$J20	$Iy20	$Iz20	300
element	elasticBeamColumn	145	281	280	$A20	$E1	$G1	$J20	$Iy20	$Iz20	300
element	elasticBeamColumn	146	259	260	$A1	$E1	$G1	$J1	$Iy1	$Iz1	300
element	elasticBeamColumn	147	258	259	$A2	$E1	$G1	$J2	$Iy2	$Iz2	300
element	elasticBeamColumn	148	257	258	$A3	$E1	$G1	$J3	$Iy3	$Iz3	300
element	elasticBeamColumn	149	256	257	$A4	$E1	$G1	$J4	$Iy4	$Iz4	300
element	elasticBeamColumn	150	255	256	$A5	$E1	$G1	$J5	$Iy5	$Iz5	300
element	elasticBeamColumn	151	254	255	$A6	$E1	$G1	$J6	$Iy6	$Iz6	300
element	elasticBeamColumn	152	253	254	$A7	$E1	$G1	$J7	$Iy7	$Iz7	300
element	elasticBeamColumn	153	252	253	$A8	$E1	$G1	$J8	$Iy8	$Iz8	300
element	elasticBeamColumn	154	251	252	$A9	$E1	$G1	$J9	$Iy9	$Iz9	300
element	elasticBeamColumn	155	250	251	$A10	$E1	$G1	$J10	$Iy10	$Iz10	300
element	elasticBeamColumn	156	249	250	$A11	$E1	$G1	$J11	$Iy11	$Iz11	300
element	elasticBeamColumn	157	248	249	$A12	$E1	$G1	$J12	$Iy12	$Iz12	300
element	elasticBeamColumn	158	247	248	$A13	$E1	$G1	$J13	$Iy13	$Iz13	300
element	elasticBeamColumn	159	246	247	$A14	$E1	$G1	$J14	$Iy14	$Iz14	300
element	elasticBeamColumn	160	245	246	$A15	$E1	$G1	$J15	$Iy15	$Iz15	300
element	elasticBeamColumn	161	244	245	$A16	$E1	$G1	$J16	$Iy16	$Iz16	300
element	elasticBeamColumn	162	243	244	$A17	$E1	$G1	$J17	$Iy17	$Iz17	300
element	elasticBeamColumn	163	242	243	$A18	$E1	$G1	$J18	$Iy18	$Iz18	300
element	elasticBeamColumn	164	241	242	$A19	$E1	$G1	$J19	$Iy19	$Iz19	300
element	elasticBeamColumn	165	240	241	$A19	$E1	$G1	$J19	$Iy19	$Iz19	300
element	elasticBeamColumn	166	239	240	$A18	$E1	$G1	$J18	$Iy18	$Iz18	300
element	elasticBeamColumn	167	238	239	$A17	$E1	$G1	$J17	$Iy17	$Iz17	300
element	elasticBeamColumn	168	237	238	$A16	$E1	$G1	$J16	$Iy16	$Iz16	300
element	elasticBeamColumn	169	236	237	$A15	$E1	$G1	$J15	$Iy15	$Iz15	300
element	elasticBeamColumn	170	235	236	$A14	$E1	$G1	$J14	$Iy14	$Iz14	300
element	elasticBeamColumn	171	234	235	$A13	$E1	$G1	$J13	$Iy13	$Iz13	300
element	elasticBeamColumn	172	233	234	$A12	$E1	$G1	$J12	$Iy12	$Iz12	300
element	elasticBeamColumn	173	232	233	$A11	$E1	$G1	$J11	$Iy11	$Iz11	300
element	elasticBeamColumn	174	231	232	$A10	$E1	$G1	$J10	$Iy10	$Iz10	300
element	elasticBeamColumn	175	230	231	$A9	$E1	$G1	$J9	$Iy9	$Iz9	300
element	elasticBeamColumn	176	229	230	$A8	$E1	$G1	$J8	$Iy8	$Iz8	300
element	elasticBeamColumn	177	228	229	$A7	$E1	$G1	$J7	$Iy7	$Iz7	300
element	elasticBeamColumn	178	227	228	$A6	$E1	$G1	$J6	$Iy6	$Iz6	300
element	elasticBeamColumn	179	226	227	$A5	$E1	$G1	$J5	$Iy5	$Iz5	300
element	elasticBeamColumn	180	225	226	$A4	$E1	$G1	$J4	$Iy4	$Iz4	300
element	elasticBeamColumn	181	224	225	$A3	$E1	$G1	$J3	$Iy3	$Iz3	300
element	elasticBeamColumn	182	223	224	$A2	$E1	$G1	$J2	$Iy2	$Iz2	300
element	elasticBeamColumn	183	222	223	$A1	$E1	$G1	$J1	$Iy1	$Iz1	300
element	elasticBeamColumn	184	221	222	$A1	$E1	$G1	$J1	$Iy1	$Iz1	300
element	elasticBeamColumn	185	220	221	$A2	$E1	$G1	$J2	$Iy2	$Iz2	300
element	elasticBeamColumn	186	219	220	$A3	$E1	$G1	$J3	$Iy3	$Iz3	300
element	elasticBeamColumn	187	218	219	$A4	$E1	$G1	$J4	$Iy4	$Iz4	300
element	elasticBeamColumn	188	217	218	$A5	$E1	$G1	$J5	$Iy5	$Iz5	300
element	elasticBeamColumn	189	216	217	$A6	$E1	$G1	$J6	$Iy6	$Iz6	300
element	elasticBeamColumn	190	215	216	$A7	$E1	$G1	$J7	$Iy7	$Iz7	300
element	elasticBeamColumn	191	214	215	$A8	$E1	$G1	$J8	$Iy8	$Iz8	300
element	elasticBeamColumn	192	213	214	$A9	$E1	$G1	$J9	$Iy9	$Iz9	300
element	elasticBeamColumn	193	212	213	$A10	$E1	$G1	$J10	$Iy10	$Iz10	300
element	elasticBeamColumn	194	211	212	$A11	$E1	$G1	$J11	$Iy11	$Iz11	300
element	elasticBeamColumn	195	210	211	$A12	$E1	$G1	$J12	$Iy12	$Iz12	300
element	elasticBeamColumn	196	209	210	$A13	$E1	$G1	$J13	$Iy13	$Iz13	300
element	elasticBeamColumn	197	208	209	$A14	$E1	$G1	$J14	$Iy14	$Iz14	300
element	elasticBeamColumn	198	207	208	$A15	$E1	$G1	$J15	$Iy15	$Iz15	300
element	elasticBeamColumn	199	206	207	$A16	$E1	$G1	$J16	$Iy16	$Iz16	300
element	elasticBeamColumn	200	205	206	$A17	$E1	$G1	$J17	$Iy17	$Iz17	300
element	elasticBeamColumn	201	204	205	$A18	$E1	$G1	$J18	$Iy18	$Iz18	300
element	elasticBeamColumn	202	203	204	$A19	$E1	$G1	$J19	$Iy19	$Iz19	300
element	elasticBeamColumn	203	203	202	$A20	$E1	$G1	$J20	$Iy20	$Iz20	300
element	elasticBeamColumn	204	202	201	$A20	$E1	$G1	$J20	$Iy20	$Iz20	300

# 101: 竖向支撑（刚度很大）
uniaxialMaterial Elastic 101 1.0e10

# 103: DX型支座的横向约束方向
uniaxialMaterial Elastic 103 1.0e10

