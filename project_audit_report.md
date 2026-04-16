# BTC Predictor: Component Logic Refinement & Hardening Audit

**Date**: 2026-04-15
**Benchmark**: Core Logic Refinement (Focus on mathematical hardening, temporal alignment, and data resilience)
**Classification**: High-Complexity Agentic Forecasting System

---

## 1. Executive Summary

The project is a sophisticated "HUD-centric" forecasting system. The objective of this audit is to transition from the current "functional-but-brittle" state to a **Refined V2** by hardening the core logic and eliminating mathematical bottlenecks. 

| Component | Health | Refinement Tier | Primary Risk |
| :--- | :--- | :--- | :--- |
| **ETL Pipeline** | YELLOW | Logic Hardening | Feature Saturation (Wiki Views) & Mismatch (RSS). |
| **Neural Core** | GREEN | Logic Hardening | Momentum vs. Noise Anchor imbalance during volatility. |
| **MLOps** | RED | Lifecycle Sync | Lack of structured operational memory for agents. |
| **UX/UI** | GREEN | Polished | Theme is stable; requires refined performance metrics. |

---

## 2. Refinement Audit

### A. Data Engineering (Resilience & Alignment)
**Current State**: Sequential ingestion with custom normalization anchors.
- **Refinement F-01 (Adaptive Scaling)**: Transition from `WIKI_VIEWS_FIXED_MAX = 200_000` to an **Adaptive Maximum** anchor. This uses historical peak views but allows for dynamic adjustment to prevent signal saturation during hyper-interest cycles.
- **Refinement F-02 (API Resilience)**: The `MarketAdapter` requires a standard retry-and-backoff protocol to survive transient API failures without returning empty DataFrames.
- **Flaw P-03 (Sentiment Alignment)**: The current `DataOrchestrator` implementation of RSS sentiment creates a training-inference mismatch. This must be refined to isolate sentiment influence strictly to the `t-0` live row.

### B. Algorithmic Core (Grounding & Trajectory)
**Current State**: Stacked LSTM with MC Dropout. Inception Grounding via linear decay.
- **Refinement F-03 (Momentum-Favored Grounding)**: The `GROUNDING_FACTOR` currently applies a flat 0.5 anchor. To improve predictive accuracy, this will be refined to favor **Neural Momentum** during high-volatility events. This allows the model to "speak through the noise" of high-variance live trades.
- **Logical Proposal**: Transition from linear decay to **Confidence-Weighted Decay**. The standard deviation (`std`) from MC Dropout will regulate how quickly the anchor vanishes.

### C. Operational Sync (MLOps refinement)
**Current State**: Script-driven deployments with fragmented logging.
- **Refinement F-04 (Persistence Synchronization)**: Transition from manual history-tracking to an agent-driven **Synchronized History**. The `/mlops_expert` will manage the `pipeline_history.json` as a primary tool for detecting drift and determining recalibration necessity.

---

## 3. Flaw manifest (Hardening Backlog)

| ID | Component | Severity | Refinement Path |
| :--- | :--- | :--- | :--- |
| **P-03** | ETL | HIGH | **Sentiment Isolation**: Rewrite `DataOrchestrator` hybrid signal logic to protect historical rows. |
| **P-01** | ETL | HIGH | **Gap Recovery**: Finalize `_stitch_yesterday_gap` to eliminate ffill biases. |
| **F-01** | ETL | MEDIUM | **Adaptive Max**: Implement dynamic normalization in `MarketAdapter`. |
| **F-03** | Logic | MEDIUM | **Momentum Grounding**: Update `ForecastingFacade` to favor model trajectory over noise. |

---

## 4. Refinement Missions (Agent Action Items)

### Mission 1: The "Resilience Protocol" (Agent: `/mlops_expert`)
- **Objective**: Harden the data ingestion layer and operational memory.
- **Requirement**: Execute `scripts/sync_pipeline_state.py` and implement the `@retry` decorator across the `MarketAdapter`.

### Mission 2: The "Neural Trajectory" (Agent: `/ds_expert`)
- **Objective**: Implement Momentum-Favored Grounding.
- **Requirement**: Update `ForecastingFacade.get_forecast` to prioritize Model Momentum during volatility spikes.
- **Success Condition**: Neural Bias audit logs reflect clear trajectory preservation during 5%+ daily moves.

### Mission 3: The "Schema Warden" (Agent: `/logic_expert`)
- **Objective**: Verify and Enforce.
- **Requirement**: Audit Gate 6 completions and cross-reference `pipeline_history.json` for all Work Orders.
- **Success Condition**: Zero unmapped ETL runs in the history log.
