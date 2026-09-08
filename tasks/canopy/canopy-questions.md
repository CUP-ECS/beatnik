Hi Steve,

I’m working on the far-field portion of the C++ port of the zmodel. The treecode solver carries the regularization factor (“blob”) directly into the Barnes-Hutt Taylor series expansion, implemented in the _expansion_batch function in treecode.py.

Canopy, our fast multipole solver, does not regularize the far-field but it regularizes the near-field. The problem is that with an epsilon value of 0.025 and a bubble of radius 0.25, Canopy must force nearly the entire domain into the near-field to achieve more than 1e-2 accuracy. This eliminates the O(n) advantage of the FMM.

Do you know of any methods to integrate the regularization factor into the Laplace (1/r) FMM kernel? We can use the Barnes-Hutt method as a fallback, but we prefer the full FMM method because it adds more interesting nobs to tune in the far-field portion of the solve.

Please let me know,
Jason

---

Hi Jason,

The issue isn't that the far field needs regularizing; rather,  it's that Canopy's far-field operator is blob-unaware, while our treecode's is blob-aware. That's the whole difference.

Against the bare 1/r² kernel your Laplace M2L expands, dropping the blob costs a relative error of ~(3/2)(δ/r)². To hold 1e-2, you then need every far interaction at r ≳ δ·√(3/(2·tol)) ≈ 12δ ≈ 0.30, which is bigger than the 0.25 bubble — so nearly everything falls into the near field. It's not a MAC you can tune around; the blob-unaware far field forces it.

One thing to be clear on: if Canopy is a classical spherical-harmonic (Greengard–Rokhlin) FMM, you can't fold δ into its existing M2L. The leading blob correction is isotropic and ~1/r³, and the only isotropic harmonic functions in 3D are const and 1/r, so no finite solid-harmonic expansion of a single source represents it. You need a different far-field operator, not a patch to the harmonic one.

You barely need any of it, though: the blob correction decays as (δ/r)^{2(k+1)} once you keep k correction orders — one order pulls the required separation from ~12δ to ~4δ, two orders to ~2.5δ, back inside ordinary FMM well-separatedness. Here are your options, roughly least-effort to most:

Quick unblock, no change to Canopy's core. Keep your bare-1/r² far field and add a separate 1–2 term Cartesian-Taylor correction pass for the blob on the octree Canopy already built; this is essentially a cut-down _expansion_batch. One correction term makes the residual (δ/r)⁴ and shrinks the regularization near-field from ~0.30 to ~0.09, which gets O(N) back immediately.
Upgrade our blob-aware treecode to a blob-aware FMM. _expansion_batch already computes the blob-aware Cartesian Taylor moments (G/D/Q, δ carried in). A Cartesian-Taylor FMM reuses those verbatim and just adds the M2M/M2L/L2L translations; the far field is then regularized automatically at every order. This is the Duan–Krasny Cartesian-treecode → FMM path; this is likely the best reuse of what you have.
Black-box FMM (Fong & Darve, JCP 228, 2009). Chebyshev interpolation of the kernel per box; this is kernel-independent, so it takes r/(r²+δ²)^{3/2} directly, no singular/harmonic split. The interpolation order is the far-field knob you want. Least new derivation if you do not want to hand-build translations.
Kernel-independent FMM (Ying, Biros & Zorin, JCP 196, 2004). This would be the equivalent-density version, also black-box in the kernel. More to stand up than bbFMM,  but it's very robust.

I'd start with 1 to unblock, then decide between 2 and 3 for the permanent fix depending on how much you want to hand-derive vs. interpolate. I can send the exact Taylor-coefficient recurrences from _expansion_batch and sketch the M2L for the Cartesian version, and that is the only piece the treecode doesn't already have.

Best regards, Steve

---

Hi Steve, 

Thank you for the explanation! You conveyed the problem in much better words. Patrick, let's discuss how to approach this considering the four options.

Steve, yes, it would be extremely helpful if you sent the exact Taylor-coefficient recurrences from _expansion_batch and sketched the M2L for the Cartesian version.

Thanks,
Jason

---

Hi Jason,

Here are the exact coefficients _expansion_batch implements with some guidance to turn them into an M2L. The main point is that the procedure is Cartesian derivatives of a single scalar

φ(r) = (|r|² + blob)^{−1/2},    K(r) = r/(|r|²+blob)^{3/2} = −∇φ(r).

Blob-awareness costs nothing structurally. The blob sits inside w = |r|² + blob and rides along at every derivative order. That is the only reason the treecode's far field is accurate where your harmonic M2L isn't. The FMM just needs the same tensors in its translation operator.

Notation below: r ≡ rv (target minus box/cell center), w = |r|²+blob, and the radial ladder

P_m = w^{−(2m+1)/2},   so P_0 = w^{−1/2} = φ, P_1 = 1/w15, P_2 = 1/w25, P_3 = 1/w35.

1. The ladder rule. Every derivative in the file comes from one identity:

∂a P_m = −(2m+1) r_a P{m+1}.

Because ∂_a w = 2 r_a, differentiating never leaves the ladder; it just advances m and drops an r_a. Blob is inert under it — it only ever appears in w.

2. The exact tensors in _expansion_batch. Let b_k = ∂^k φ (multi-index k). Applying the ladder rule:

b_∅            = P_0
(∂_a)          = −r_a P_1
(∂_a∂_b)       = −δ_ab P_1 + 3 r_a r_b P_2
(∂_a∂_b∂_c)    = 3(δ_ab r_c + δ_ac r_b + δ_bc r_a) P_2 − 15 r_a r_b r_c P_3

Since K = −∇φ, the code's three arrays are exactly minus these, shifted by one index:

K  (rv/w15)            =  r_a P_1                                   = −(∂_a φ)
dK                     =  δ_ab P_1 − 3 r_a r_b P_2                  = −(∂_a∂_b φ)
ddK                    = −3(δ_ab r_c + δ_ac r_b + δ_bc r_a) P_2 + 15 r_a r_b r_c P_3 = −(∂_a∂_b∂_c φ)

(You can line these up term-by-term with lines 59, 62, and 68–77.) So the "monopole/dipole/quadrupole" expansion is: contract the (n+1)-th derivative tensor of φ with the n-th moment and cross with the strength,  which is what the cross(...)/einsum lines do.

3. Arbitrary order (if you ever want p > 2). The ladder provides a closed Cartesian recurrence, and there is no need to hand-expand. Solving the identity w ∂_i φ = −r_i φ order by order:

w · b_{k+e_i} = −r_i b_k − k_i b_{k−e_i} − 2 Σ_j k_j r_j b_{k+e_i−e_j} − Σ_j k_j (k_j−1) b_{k+e_i−2e_j}

with e_i the unit multi-index in direction i, and any b with a negative component ≡ 0. Every coefficient at order |k|+1 comes from orders |k| and |k|−1; blob enters only through the leading w. I checked that this reproduces b_∅, b_{e_i}, b_{2e_i} above. Then, divide by k! where you fold them into moments.

4. M2L sketch (Cartesian version). The velocity is vector-valued, but the cross product is bilinear, so peel it off: with [K×γ]i = ε{ilm} K_l γ_m and K_l = −∂l φ, the far field is three scalar-φ FMM passes (one per strength component γ_m), recombined by ε{ilm} at the end. Everything below is then the ordinary Cartesian-Taylor FMM for the scalar kernel φ, blob-aware because w carries blob.

Moments (upward pass). Your _build already computes them: G, D, Q = Σγ, Σ d⊗γ, Σ d⊗d⊗γ with d = y − c_B, i.e. the order 0/1/2 multipole moments M_q^B = Σ_{j∈B} d_j^q/q! · s_j about box center c_B. Nothing new here.

M2L (the one new operator). For a well-separated source box B → target box A with center separation R = c_A − c_B, the local coefficients about c_A are

ℓ_p^A = Σ_q  [ (−1)^{|q|} b_{p+q}(R) ] · M_q^B ,

where b_n(R) = ∂^n φ(R) are the same tensors from §2/§3, now evaluated once per box-pair at R (not per target). The blob is in w = |R|²+blob, so the translation is regularized automatically.

Reuse shortcut: _expansion_batch(R, G, D, Q, blob, order) already IS the p = 0 local coefficient (the value at the box center). To get a real FMM, keep that as ℓ_0 and add the p ≥ 1 coefficients, which are just the higher b_{p+q}(R) — same ladder, one or two more m's. So you're extending the function you have, not replacing it.

M2M / L2L (kernel-independent, no blob). Both are plain binomial Taylor shifts by the center offset s:

M2M:  M_q^{parent} = Σ_{q'≤q} (s^{q−q'}/(q−q')!) M_{q'}^{child}
L2L:  ℓ_p^{child}  = Σ_{p'≥p} (s^{p'−p}/(p'−p)!) ℓ_{p'}^{parent}

These do not see the kernel at all, so they are identical to any Cartesian FMM.

Evaluate. At target x in box A: u_scalar-pass(x) = Σ_p (x−c_A)^p/p! · ℓ_p^A, then apply ε_{ilm} across the three passes.

5. The order that you actually need. Because the blob correction dies as (δ/R)^{2(k+1)}, order p = 2 (exactly what _expansion_batch already uses)  is enough at standard FMM admissibility (R ≳ 2–3 box widths). You do not need higher order to beat the regularization problem; you need box-to-box translation (M2L/L2L) so the far field is O(N) instead of the treecode's O(N log N). The M2L above is the only genuinely new code; moments and the tensors are already written.

The only place that needs care is the sign and normalization convention between the moment definition and the (−1)^{|q|} multiplier. To verify the conventions, confirm that ℓ_0 reproduces _expansion_batch for a single source box and target box, and then proceed to the higher order coefficients.

Best regards, Steve