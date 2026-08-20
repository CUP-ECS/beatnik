Python command run to generate gold files:
```
python examples/run_adaptive_mesh_bubble.py \
  --A 0.3 --g 1.0 --mu 0.002 --eps 0.025 \
  --viscosity-mode laplace-beltrami --br-approximation direct \
  --adaptive-dt --no-dynamic-remesh --refine-every 0 \
  --source-quadrature vertex \
  --icosphere-subdivisions 3 --steps 2000 \
  --checkpoint-every-steps 25 --no-video --checkpoint-dir results3
```
