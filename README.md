# The "Omni" World Model: A Comprehensive Research & Engineering Plan

**Vision:** To create a production-ready, explainable AI system that models the world as a dynamic, hierarchical temporal knowledge graph. This model will be capable of calculating calibrated probabilities of future events, forecasting causal ripple effects, and providing interpretable reasoning for its predictions.

---
## Phase 0: Foundation & Research Validation

**Objective:** To establish a robust foundation for the project by addressing data quality, entity ambiguity, and creating a rigorous evaluation framework before any large-scale data processing begins.

### 0.1. Entity Resolution & Disambiguation
*   **Problem:** GDELT entities ("Police", "Apple") are ambiguous.
*   **Implementation:**
    *   Develop a Named Entity Linking (NEL) module in `src/utils/entity_linker.py`.
    *   Use spaCy for initial Named Entity Recognition (NER).
    *   Employ a pre-trained model like **BLINK** or a similar transformer-based model to disambiguate entities by linking them to canonical identifiers in a knowledge base (e.g., **Wikidata, DBpedia**).
    *   Store the canonical mappings in a local database (e.g., `data/resources/entity_mappings.db`).
    *   Incorporate temporal and location context into the disambiguation logic.

### 0.2. Data Quality Assessment Framework
*   **Problem:** GDELT data has known noise and reliability issues.
*   **Implementation:**
    *   Create a `src/utils/data_quality.py` module.
    *   Implement functions for:
        *   **Source Credibility Scoring:** Develop a heuristic or model to score the reliability of news sources.
        *   **Temporal Consistency Checks:** Detect and flag contradictory events (e.g., "peace deal signed" followed immediately by "declaration of war").
        *   **Outlier Detection:** Identify and analyze events with extreme `GoldsteinScale` or `AvgTone` scores.
    *   Generate data quality reports during the ingestion phase.

### 0.3. Benchmark Dataset Selection
*   **Problem:** The original plan lacked standard benchmarks for evaluation.
*   **Implementation:**
    *   Download and prepare several standard temporal knowledge graph forecasting datasets in `data/benchmarks/`.
    *   **Key Datasets:** **ICEWS14/ICEWS05-15**, a pre-processed **GDELT** subset, **MusTQ** for multi-hop reasoning, and the **TGB (Temporal Graph Benchmark)**.
    *   These will be used for model validation and comparison against state-of-the-art (SOTA) results.

---
## Phase 1: Enhanced Data Pipeline

**Objective:** To build an incremental, quality-aware data pipeline that ingests GDELT data and enriches it with temporal and semantic information.

### 1.1. Temporal Validity & Event Encoding
*   **Problem:** Events are not instantaneous; they have a period of relevance.
*   **Implementation:**
    *   Modify the data schema to include `valid_from`, `valid_until`, and `confidence_score` for each event.
    *   Create an `src/utils/event_encoder.py` module.
    *   Implement a richer event representation beyond simple CAMEO codes, potentially using **Contextual Self-Supervised Contrastive Learning** to generate event embeddings.

### 1.2. Incremental Learning Architecture
*   **Problem:** The model must learn continuously as new data arrives.
*   **Implementation:**
    *   Design the training pipeline around an **Online Continual Graph Learning (OCGL)** paradigm.
    *   Implement a `src/training/incremental_learner.py` module.
    *   Use sliding time windows for training, an experience replay buffer to mitigate catastrophic forgetting, and neighborhood expansion control to manage memory.

---
## Phase 2: Advanced Graph Construction

**Objective:** To construct a hierarchical, heterogeneous temporal graph that captures the complex relationships and dynamics of world events.

### 2.1. Hierarchical & Heterogeneous Graph Structure
*   **Problem:** A flat graph misses crucial semantic hierarchies.
*   **Implementation:**
    *   Use a graph library that supports heterogeneous types (e.g., DGL or PyG).
    *   Define typed nodes (e.g., `PERSON`, `ORGANIZATION`, `LOCATION`, `EVENT`) and typed edges.
    *   Incorporate hierarchies (e.g., `Government` -> `Police` -> `French Police`).

### 2.2. Dynamic Graph Snapshots & Feature Engineering
*   **Problem:** Storing full graph snapshots is inefficient; features must be sophisticated.
*   **Implementation:**
    *   Use **differential encoding** to store only the changes (diffs) between graph snapshots.
    *   Create a `src/graph/feature_extractor.py` module to compute advanced features:
        *   **Temporal Centrality:** How node importance evolves.
        *   **Graph Motifs:** Detect common interaction patterns (e.g., triangles of cooperation).
        *   **Structural Holes:** Identify entities that bridge disconnected communities.

---
## Phase 3: Advanced GNN Architecture & Training

**Objective:** To build and train a state-of-the-art temporal GNN capable of multi-task learning, causal inference, and handling new entities.

### 3.1. Model Architecture: Temporal Graph Transformer
*   **Problem:** The previously planned GraphSAGE is outdated for complex temporal data.
*   **Implementation:**
    *   Implement a **Temporal Graph Transformer** architecture (e.g., **TGAT**).
    *   The architecture will follow a `Temporal Encoding → Graph Attention → Temporal Attention → Decoder` flow.
    *   Integrate a **Temporal Point Process** to model the intensity and timing of event occurrences.

### 3.2. Multi-Task & Contrastive Learning
*   **Problem:** Training only on link prediction is inefficient.
*   **Implementation:**
    *   Frame the training as a **multi-task learning** problem with a shared encoder and task-specific decoders.
    *   **Primary Task:** Link Prediction.
    *   **Auxiliary Tasks:** Event Type Prediction, Temporal Prediction (when?), Intensity Prediction.
    *   Incorporate a **Temporal Reasoning with Contrastive Learning (TRCL)** component to mitigate noise by distinguishing historically repeated events from random co-occurrences.

### 3.3. Causal Inference & Cold Start Handling
*   **Problem:** The model must understand causality and handle new entities.
*   **Implementation:**
    *   Add a `src/analysis/causal_inference.py` module to perform **Granger Causality** tests and explore **Structural Causal Models (SCM)** for counterfactual reasoning.
    *   Add a `src/models/cold_start_handler.py` module using **meta-learning** or attribute-based GNN features to handle predictions for new, unseen entities.

---
## Phase 4: Enhanced Oracle & Query Engine

**Objective:** To build a sophisticated query engine that can understand complex natural language questions and generate explained, calibrated probability scores.

### 4.1. Advanced NLP & Feature Engineering
*   **Problem:** Basic parsing is insufficient for complex queries.
*   **Implementation:**
    *   Enhance `p4_query_parser.py` to handle **multi-hop reasoning** and classify question types (factual, counterfactual).
    *   Use advanced temporal expression normalization libraries (**SUTime**, **HeidelTime**).
    *   Expand `p4_feature_extraction.py` to include features from the causal inference module, graph topology features, and uncertainty quantification.

### 4.2. Calibrated & Explainable Probabilities
*   **Problem:** A raw probability score is not enough.
*   **Implementation:**
    *   Use **temperature scaling** or other calibration methods to ensure probability scores are reliable.
    *   Generate confidence intervals for predictions using bootstrapping.
    *   Integrate **SHAP (SHapley Additive exPlanations)** to generate "because..." style explanations for predictions.

---
## Phase 5: Production & MLOps (New Phase)

**Objective:** To build a production-ready, scalable, and maintainable system.

*   **MLOps Pipeline:** Implement Level 2 MLOps maturity using tools like **MLflow** or **Kubeflow**. This includes CI/CD for pipelines, a model registry, a feature store, and continuous training triggers.
*   **Distributed Training:** Implement graph partitioning (e.g., using **METIS**) and distributed training (e.g., using **PyTorch DDP**) in a `src/distributed/` module to handle the full GDELT scale.
*   **Scalability Architecture:** Use **Kafka** for real-time data streaming, **Redis** for caching, and **FAISS** for approximate nearest neighbor search for entity matching.
*   **Monitoring:** Implement dashboards using **Prometheus + Grafana** to monitor model performance drift, feature distribution shifts, and system health.

---
## Phase 6: Explainability & Interpretability (New Phase)

**Objective:** To make the model's predictions transparent and trustworthy.

*   **GNN Explainability:** Implement techniques like **GNNShap** to identify the most influential edges and nodes for a given prediction.
*   **User Interface:** Move beyond a CLI to an interactive dashboard (e.g., using **D3.js** or **Plotly Dash**) that visualizes the explanatory subgraphs and attention weights.

---
## Evaluation, Research & Risk Management

*   **Evaluation Framework:** Define a comprehensive set of metrics (MRR, Hits@k, ECE for calibration) and perform ablation studies to test components. Compare against SOTA baselines like **TiRGN, TRCL, and TGAT**.
*   **Experimentation Infrastructure:** Use **Weights & Biases (W&B)** for experiment tracking and **Optuna** for Bayesian hyperparameter optimization.
*   **Reproducibility:** Enforce strict random seed management, use **Docker** for environment specification, and **DVC (Data Version Control)** for datasets.
*   **Risk Mitigation:** Address GDELT data quality by exploring alternative sources (**ACLED, Phoenix**). Manage scope creep with an agile MVP approach.
*   **Publication Strategy:** Target top-tier ML conferences (**NeurIPS, ICML**) and domain-specific venues (**KDD, WWW**).

---
## Enhanced Requirements & Dependencies

*A comprehensive `requirements.txt` will be created based on the tools mentioned above (e.g., `torch`, `torch_geometric`, `dvc`, `mlflow`, `optuna`, `shap`, `faiss-cpu`, etc.).*

---
## Timeline & Resource Estimates

*   **Total Estimated Timeline:** 6-9 months with 1-2 full-time researchers.
*   **Computational Needs:** Multi-GPU setup (4-8 GPUs) for training; auto-scaling cloud infrastructure for production.
*   **Storage Needs:** 1-2 TB for GDELT data and model artifacts.
