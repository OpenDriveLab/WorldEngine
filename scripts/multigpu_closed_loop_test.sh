# export WORLDENGINE_ROOT="/path/to/your/WorldEngine"

bash projects/SimEngine/scripts/run_ray_distributed_testing.sh \
    $WORLDENGINE_ROOT/projects/AlgEngine/configs/worldengine/e2e_vadv2_50pct.py \
    $WORLDENGINE_ROOT/data/alg_engine/ckpts/e2e_vadv2_50pct_ep8.pth \
    e2e_vadv2_50pct \
    navtest_failures \
    NR