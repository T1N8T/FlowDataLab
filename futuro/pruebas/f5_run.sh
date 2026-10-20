#!/bin/bash
# tanda f5: variantes de LGB (CPU) y de la red (GPU) + segundo paso de sesión + stack + suavizado anidado
cd "$(dirname "$0")/../.."
P=.venv/bin/python; S=futuro/pruebas/f5_stack.py; N=futuro/pruebas/notas
for v in f5_clear_binary f5_clear_xendcg f5_clear_l63; do $P futuro/pruebas/f5_clear_et.py $v > $N/$v.out 2>&1 & done
CUDA_VISIBLE_DEVICES=0 $P futuro/m_nn_future.py --clear --bs 256 --name nn_future_clear_bs256 > $N/nn_future_clear_bs256.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 $P futuro/m_nn_future.py --clear --d 128 --heads 8 --name nn_future_clear_h8 > $N/nn_future_clear_h8.out 2>&1 &
# lo que ya está entrenado: no espera a nadie
$P $S ses f5_clear_et10 f5_clear_et nn_future_clear nn_future_clear_d128 nn_future linear_future f4_two_stage
$P $S stack raw lgb_future_clear10 f5_clear_et10 nn_future_clear nn_future_clear_d128 linear_future f4_two_stage
$P $S ses stackf_raw_avg stackf_raw_meta
$P $S stack ses lgb_future_clear10_ses f5_clear_et10_ses nn_future_clear_ses nn_future_clear_d128_ses linear_future_ses f4_two_stage_ses
$P $S suav f5_clear_et10_ses nn_future_clear_ses stackf_raw_avg_ses stackf_raw_meta_ses stackf_ses_avg stackf_ses_meta
echo FASE1
wait
$P $S ses f5_clear_binary f5_clear_xendcg f5_clear_l63 nn_future_clear_bs256 nn_future_clear_h8
$P $S stack all lgb_future_clear10_ses f5_clear_et10_ses nn_future_clear_ses nn_future_clear_d128_ses linear_future_ses f4_two_stage_ses f5_clear_binary_ses f5_clear_xendcg_ses f5_clear_l63_ses nn_future_clear_bs256_ses nn_future_clear_h8_ses
$P $S suav stackf_all_avg stackf_all_meta
echo FIN
