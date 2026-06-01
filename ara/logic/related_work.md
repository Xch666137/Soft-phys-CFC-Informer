# Related Work

## RW01: Physics-Guided Neural Networks (PGNN)
- **DOI/Ref**: Karpatne et al. (2017), "Physics-guided Neural Networks"
- **Type**: imports
- **Delta**: PhysFormer uses physics equations as FiLM conditioning (soft inductive bias), whereas PGNN uses physics as a hard residual loss term. FiLM is more flexible but provides weaker guarantees.
- **Claims affected**: C01

## RW02: FiLM: Feature-wise Linear Modulation
- **DOI/Ref**: Perez et al. (2018), "FiLM: Visual Reasoning with a General Conditioning Layer"
- **Type**: imports
- **Delta**: PhysFormer adapts FiLM for time-series physics conditioning rather than visual question answering. The conditioning signal is meteorological features rather than language.
- **Claims affected**: C01

## RW03: Deep Residual Learning (ResNet)
- **DOI/Ref**: He et al. (2015), "Deep Residual Learning for Image Recognition"
- **Type**: imports
- **Delta**: PhysFormer's residual learning (theory + correction) is directly inspired by ResNet's F(x) + x formulation. The key difference: PhysFormer's "identity" is a physics-based estimate, not the input itself. The sigmoid gate failure (V4.1) mirrors ResNet's finding that identity shortcuts are sufficient.
- **Claims affected**: C01, C03, C04

## RW04: Informer / iTransformer / Autoformer
- **DOI/Ref**: Zhou et al. (2021), "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting"; Liu et al. (2023), "iTransformer"; Wu et al. (2021), "Autoformer"
- **Type**: baseline
- **Delta**: These are pure data-driven Transformer variants for time-series. PhysFormer adds physics conditioning and component decomposition. They serve as black-box baselines.
- **Claims affected**: C01

## RW05: VPP Forecasting Literature
- **DOI/Ref**: Various — VPP aggregated power forecasting using statistical/ML methods
- **Type**: bounds
- **Delta**: Existing VPP forecasting either uses pure statistical methods (ARIMA, SARIMA) or pure ML (LSTM, GRU). No prior work embeds per-component physics equations with learnable residuals for VPP net power.
- **Claims affected**: C01, C06

## RW06: Curriculum Learning
- **DOI/Ref**: Bengio et al. (2009), "Curriculum Learning"
- **Type**: imports
- **Delta**: PhysFormer applies curriculum to the physics-accuracy trade-off (component loss weight annealing), whereas classic curriculum learning sorts training examples by difficulty.
- **Claims affected**: C05

## RW07: PINN (Physics-Informed Neural Networks)
- **DOI/Ref**: Raissi et al. (2019), "Physics-informed Neural Networks"
- **Type**: imports
- **Delta**: PINNs enforce physics as hard PDE constraints via the loss function. PhysFormer uses physics as soft FiLM conditioning with learnable residuals — more appropriate for problems where physics is approximate.
- **Claims affected**: C01
