#!/bin/bash
# Launch all experiments in parallel across 3 GPUs.
# Estimated wall time: ~9 hours.
#
# GPU0: MVTec 8 classes  (carpet grid leather tile wood bottle cable capsule)  150 epochs
# GPU1: MVTec 7 classes  (hazelnut metal_nut pill screw toothbrush transistor zipper) 150 epochs
# GPU2: Contamination sweep  (5 MVTec classes + MMS)  5 rates x 50 epochs

cd "$(dirname "$0")"

echo "Launching GPU0 (MVTec 8 classes)..."
bash run_tinyglass_mvtec_gpu0.sh > ../logs/gpu0_mvtec.log 2>&1 &
PID0=$!

echo "Launching GPU1 (MVTec 7 classes)..."
bash run_tinyglass_mvtec_gpu1.sh > ../logs/gpu1_mvtec.log 2>&1 &
PID1=$!

echo "Launching GPU2 (Contamination sweep)..."
bash run_contamination.sh 2 > ../logs/gpu2_contamination.log 2>&1 &
PID2=$!

echo ""
echo "All launched. PIDs: GPU0=$PID0  GPU1=$PID1  GPU2=$PID2"
echo "Monitor with:"
echo "  tail -f ../logs/gpu0_mvtec.log"
echo "  tail -f ../logs/gpu1_mvtec.log"
echo "  tail -f ../logs/gpu2_contamination.log"
echo ""
echo "After completion, run from project root:"
echo "  python plot_results.py"

wait $PID0 $PID1 $PID2
echo "All done at $(date)"
