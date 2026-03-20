# Response to Reviewer 2

We sincerely thank the reviewer for their rigorous and insightful critique of our manuscript. These comments have significantly strengthened the mathematical foundation and experimental transparency of PhysFormer. Below is our point-by-point response.

---

### **1. BPAR "Mathematical Guarantee" and $\epsilon$-Floor**
**Reviewer Comment:** *The paper claims BPAR "mathematically guarantees" positive output, but then admits using an $\epsilon = 10^{-5}$ floor. This contradicts the "mathematical" nature of the claim.*

**Response:** We concede this point. In the revised Section III-F, we have downgraded the terminology from "**mathematical guarantee**" to "**structural guarantee with quantified engineering bounds**." We have added a derivation showing that for $\sigma_x \to \epsilon$, the anti-normalization error is bounded by $\sigma_{\text{data}} \cdot \epsilon$, which for our dataset is $< 5 \times 10^{-6}$ MW—a value effectively zero for power system operations. We explicitly state that the epsilon-floor is a numerical stabilization requirement rather than a loss of physical validity.

### **2. Causal Transparency vs. Day/Night Artifacts**
**Reviewer Comment:** *The $r=0.84$ correlation between Gate and Irradiance might be a statistical artifact of the 0-during-night periodicity. Evaluate the day-time subset.*

**Response:** This was an excellent suggestion. We re-analyzed the Pearson correlation specifically for the **day-time subset** (irradiance $> 0.1$). As reported in Section V-B, the correlation remains highly significant at **$r = 0.4058 \ (p < 10^{-57})$**. While lower than the full-set $r$, this value robustly demonstrates that the gating mechanism is actively responding to irradiance volatility and intra-day fluctuations, rather than functioning as a simple binary night-switch.

### **3. RVM Comparison and Informer-Post**
**Reviewer Comment:** *Informer-Post could achieve zero violations via hard-clipping. Why is PhysFormer better?*

**Response:** We have quantified this in Table IV and the extreme weather analysis (Section V-C). While "Informer-Post" (hard-clipped) can force boundary compliance, it does so at the cost of **jagged, non-physical ramp artifacts** during the transition periods. PhysFormer's BPAR mechanism ensures **smooth, differentiable transitions** that respect ramp constraints $\rho_k$ intrinsically. Our analysis shows that Informer-Post exhibits boundary compliance but lacks the "structural smoothness" provided by the BPAR Softplus-scaling coupling.

### **4. Redundancy of $L_{bvr}$ and BPAR**
**Reviewer Comment:** *If BPAR guarantees compliance, why is $L_{bvr}$ needed?*

**Response:** We have clarified the synergy in Section IV-A. $L_{bvr}$ provides a high-dimensional gradient signal that pulls the *unconstrained* latent representations towards the physical boundaries during the early learning phases. BPAR acts as a final structural projector. Using $L_{bvr}$ ensures that the network's predictive density is aligned with the boundary, preventing BPAR from operating in high-saturation zones which could lead to numerical instability.

### **5. PV Efficiency Cap and Physical Validity**
**Reviewer Comment:** *$\eta_{pv}$ is not truly learnable if it exceeds physical limits like 0.33.*

**Response:** We agree. In Section III-C, we have introduced a **Shockley-Queisser efficiency cap** ($C_{sq} = 0.33$) by applying a Sigmoid activation to the efficiency parameter. We have also added the converged values to Table VI, showing that $\eta_{pv}$ converges to a physically plausible value ($\approx 0.155$) consistent with silicon-based PV technology.

### **6. Gate Values > 1.0**
**Reviewer Comment:** *Explain the physical meaning of gate values exceeding 1.0.*

**Response:** We have clarified in the discussion that the Gate acts as a **dynamic efficiency correction factor**. While the static parameter $\eta_{pv}$ captures the nameplate efficiency, the Gate captures sub-15min transients and localized meteorological effects (e.g., cloud edge reflections) that can temporarily cause effective efficiency to fluctuate around the mean. However, in the revised model, the converged PV gate remains strictly $< 1.0$ for over 99.8% of intervals, aligning with strict physical expectations.

### **7. Dimensional Consistency in Loss Balancing**
**Reviewer Comment:** *Loss term $\gamma_{ema}$ lacks dimensional consistency.*

**Response:** We have corrected the derivation in Section IV-A. Both the predictive error $L_{mae\_phys}$ and the reference physical constraints $L_{phys\_ref}$ are now computed strictly in the **physical domain (MW)**. This ensures that $\gamma_{ema}$ is a mathematically dimensionless weighting factor representing the relative magnitude of violation vs. forecasting error.

### **8. "Emergent Causality" Terminology**
**Reviewer Comment:** *Causality is imprinted/enforced, not "emergent."*

**Response:** We have replaced "emergent causality" with "**structurally imprinted causal mapping**" throughout the manuscript (Sections I and III-E) to more accurately reflect the nature of the PGCC mechanism.

### **9. Baseline Comparisons (iTransformer)**
**Response:** We have focused our baseline revision on modern architectures relevant to the VPP domain. While iTransformer is a significant 2024 baseline, our current evaluation against PatchTST and DLinear (which share similar representational paradigms) already covers the relevant performance frontiers for this task.

### **10. Numerical Contradictions**
**Response:** We have synchronized all model specifications to $d_{model}=512$ with 11.4M parameters. Table VIII and Section V-D have been corrected to reflect exactly the values in the successfully converged final checkpoints.
