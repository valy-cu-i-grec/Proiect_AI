# Project CelLab: Cellular Automata Simulator

**CelLab** is a comprehensive system that allows users to describe cellular automata in natural language, simulates their evolution over multiple generations, and analyzes the system's behavior to answer complex state questions.

---

## Deliverable 1: Team Identification and Technical Solutions

### Team Members
*   **Leonte Alexandru-Valentin**
*   **Istrate Andrei-Daniel**
*   **Ciocan Ionut-Gabriel**
*   **Ionescu Alexandru-Dragos**

### Proposed Technical Stack
To meet the requirements of simulating complex grids, parsing natural language, and eventually integrating neural networks (Deliverable 3), we have chosen a **Python-based architecture**.

1.  **Core Simulation Engine:** `Python` + `NumPy`
    *   *Why:* Cellular automata heavily rely on 2D matrix operations. `NumPy` provides highly optimized, C-level performance for matrix transformations, which is critical for simulating large generations and complex rules (like Rule 30 or Brian's Brain).
2.  **Graphical User Interface (GUI):** `PyQt6` (Desktop Application)
    *   *Why:* Ensures a strict, native desktop experience without relying on web browsers or local servers. `PyQt6` provides powerful widgets (like `QGraphicsView`) capable of rendering large, dynamic 2D grids efficiently, alongside complex input panels for parameters and natural language text.
3.  **Natural Language Processing (NLP):** `spaCy` / `OpenAI API`
    *   *Why:* Translating natural language into executable automata rules requires robust text parsing. We will extract entities and map them to JSON/Python dictionaries to feed into the Core Engine.
4.  **Machine Learning / RL Integration:** `PyTorch` / `scikit-learn`
    *   *Why:* Native integration within the Python ecosystem makes generating datasets from our Core Engine and training models seamless.
5.  **Data Storage:** `JSON` / `HDF5`
    *   *Why:* For storing intermediate generation states, user-defined rules, and caching datasets for ML training efficiently.

---

### Architecture Diagrams

#### C1: System Context Diagram
Shows how the user interacts with the CelLab system.


#### C2: Container Diagram
Details the internal modules of the CelLab application.

---

### Analytical Capabilities (The 5 Questions)
The core engine is designed to resolve system state inquiries:
*   **State after N iterations:** Resolved via **Reinforcement Learning (RL) / Neural Networks**. Instead of calculating every intermediate step, we will train an RL agent/NN model on a significant dataset of generated automata to predict the exact grid state after N iterations.
*   **Stable state detection:** Comparing $Matrice_N$ with $Matrice_{N-1}$.
*   **Cycle detection:** Implementing state hashing (`hash(matrix.tobytes())`) stored in a `HashSet`. If a generated hash already exists, a cycle is confirmed.
*   **Target configuration search:** Sub-matrix pattern matching.
*   **Cell counting:** Vectorized aggregations using `NumPy`.

### AI Collaboration Transparency
All interactions with conversational AI assistants used during the development of this project are strictly documented.
*   Logs are maintained in the `/ai_logs` directory.
*   AI-generated code snippets are explicitly commented in the source files with references to the respective chat logs.
