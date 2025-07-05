# PRD: TARS Simulation Platform - Research Prototype & Educational Tool

## 1. Product overview

### 1.1 Document title and version

- PRD: TARS Simulation Platform - Research Prototype & Educational Tool
- Version: 2.0

### 1.2 Product summary

This document outlines the requirements for the TARS Simulation Platform, a research tool designed to validate and demonstrate the "Trust-Aware Reinforcement Selector (TARS) for Robust Federated Learning" framework. The platform serves dual purposes: first, as a rigorous scientific instrument to reproducibly generate the results, tables, and graphs presented in the research paper; and second, as a highly interactive and visual educational tool for demonstrating how the TARS defense mechanism works in real-time.

The system simulates a complete Federated Learning environment including a central server, multiple clients (honest and Byzantine), the four specified attack strategies (label flipping, sign flipping, Gaussian noise, and pretense attacks), and a comprehensive suite of defense mechanisms. The core innovation is the TARS agent, which uses trust-aware reinforcement learning to dynamically select optimal aggregation rules. The platform features both a graphical user interface (GUI) for interactive exploration and a command-line interface for batch experiments, providing deep insight into the TARS decision-making process while enabling rigorous scientific validation.

## 2. Goals

### 2.1 Business goals

- Establish a mathematically principled and empirically validated solution for trustworthy federated learning in adversarial settings
- Demonstrate superior performance compared to existing Byzantine-robust aggregation rules (FedAvg, Krum, Trimmed Mean/Median, FLTrust, SARA)
- Produce a high-quality, extensible open-source research tool that serves as a foundation for future federated learning security research
- Enable reproducible research through rigorous experiment validation and result verification

### 2.2 User goals

- **Research Validation:** Configure and run FL simulations to validate specific performance claims from the research paper
- **Interactive Demonstration:** Visualize the training process in real-time to demonstrate the adaptive nature of TARS to academic audiences
- **Deep Inspection:** Inspect the internal state of the TARS agent (trust scores, Q-values, rule selection) to understand decision-making processes
- **Educational Use:** Provide an intuitive tool for teaching federated learning security concepts to students and peers

### 2.3 Non-goals

- Production-ready, enterprise-scale FL system deployment (focus is on simulation and validation)
- Real-world, distributed clients over network infrastructure
- Consumer-grade GUI design (functional and informative interface prioritized over aesthetics)
- Support for datasets beyond MNIST and CIFAR-10

## 3. User personas

### 3.1 Key user types

- **Primary Researcher:** Individual who developed the TARS framework and needs validation, demonstration, and extension capabilities
- **Academic Reviewer:** Peer reviewers and researchers who want to understand, verify, or learn from the TARS implementation
- **Student/Educator:** Academic instructors and students studying federated learning security concepts

### 3.2 Basic persona details

- **Shafiq, the Primary Researcher:** Deeply familiar with TARS mathematical foundations, needs reliable batch experiment execution for paper-ready graphs and dynamic visualization tools for academic presentations. Values reproducibility, observability, and extensibility above all.
- **Dr. Academic, the Peer Reviewer:** Wants to quickly understand and verify TARS claims, needs clear visual demonstrations and access to experimental data for validation.
- **Sarah, the Graduate Student:** Learning federated learning concepts, needs intuitive interface to explore trust mechanisms and see real-time attack/defense dynamics.

### 3.3 Role-based access

- **System Administrator:** Full access to simulation configuration, dataset management, and system monitoring
- **Researcher:** Access to experiment configuration, result analysis, and algorithm parameter tuning
- **Observer:** Read-only access to simulation results, trust score visualization, and performance metrics

## 4. Functional requirements

### 4.1 Simulation Engine (Priority: High)

- Support MNIST and CIFAR-10 datasets with IID and non-IID partitioning
- Simulate FL process over configurable number of rounds (1-50)
- Support configurable percentage of Byzantine clients (0-50%)
- Configurable client count (5-50 clients)

### 4.2 Defense & Attack Arsenal (Priority: High)

- **Baseline Aggregation Rules:** FedAvg, Krum, Trimmed Mean, Median, FLTrust, SARA
- **Attack Patterns:** All 4 specified attacks - label flipping, sign flipping, Gaussian noise, and pretense attacks
- **TARS Implementation:** Complete trust score calculation, temporal smoothing, state encoding, and trust-regularized reward function

### 4.3 TARS Agent Core (Priority: High)

- Trust score calculation using loss divergence, cosine similarity, and gradient magnitude
- Temporal trust smoothing with configurable decay factor β
- Q-learning policy with ε-greedy exploration for aggregation rule selection
- Real-time trust score monitoring and historical tracking

### 4.4 Interactive GUI (Priority: High)

- Configuration panel for all key simulation parameters
- Live-updating plots for model accuracy, loss, and trust scores
- Real-time view of TARS agent's internal state and decision process
- Trust table displaying per-client trust evolution
- Aggregation rule selection history with visual indicators

### 4.5 Batch Experimentation (Priority: High)

- Command-line interface for headless experiment execution
- YAML/JSON configuration file support
- Automated result export in CSV, JSON, and visualization-ready formats
- Statistical analysis with confidence intervals and significance testing

## 5. User experience

### 5.1 Entry points & first-time user flow

- **GUI Mode:** Launch application → configuration panel → parameter selection → start simulation → live visualization
- **Batch Mode:** Command-line execution with config files → automated experiment runs → result analysis
- **Educational Mode:** Pre-configured scenarios demonstrating different attack/defense combinations

### 5.2 Core experience

- **Experiment Configuration:** Intuitive parameter selection through dropdown menus and input fields for dataset, attack type, client configuration, and TARS hyperparameters
- **Live Visualization:** Real-time accuracy/loss curves, trust score evolution, and agent behavior observation during simulation execution
- **Agent Inspection:** Direct observation of aggregation rule selection, Q-table evolution, and trust score calculations in real-time
- **Result Analysis:** Comprehensive post-simulation analysis with exportable graphs, statistical summaries, and comparative performance metrics

### 5.3 Advanced features & edge cases

- Batch experiment execution for statistical significance testing
- Custom attack pattern implementation through extensible interface
- Advanced Q-learning variants (double Q-learning, experience replay)
- Multi-objective trust scoring with user-defined weight configurations
- Interactive parameter adjustment during simulation execution

### 5.4 UI/UX highlights

- Split-panel interface: configuration on left, live results on right
- Color-coded trust scores for immediate visual feedback
- Interactive plots with zoom, pan, and data point inspection
- Progress indicators and real-time performance metrics
- Export functionality for publication-ready figures

## 6. Narrative

A machine learning researcher launches the TARS simulation platform to validate research claims and prepare for an academic presentation. Using the GUI, they configure a 30-round experiment with 10 clients, 20% Byzantine attackers, and MNIST dataset with non-IID partitioning. As the simulation begins, real-time plots immediately show model accuracy and individual client trust scores. When Byzantine clients launch coordinated sign-flipping attacks at round 10, the researcher watches trust scores plummet in real-time while TARS automatically detects the threat. The Q-learning agent, observing declining performance through the interactive dashboard, switches from FedAvg to robust Krum aggregation, causing accuracy to recover toward the target 97.7%. The researcher exports the final results, generating publication-ready figures that match the paper's claims while providing visual evidence of TARS's adaptive defense capabilities.

## 7. Success metrics

### 7.1 User-centric metrics

- **Reproducibility:** Reproduce primary accuracy graphs from research paper within 2% margin across multiple runs
- **Usability:** New users can configure and run simulation with less than 5 minutes of instruction
- **Educational Value:** Clear visualization of trust mechanisms and attack/defense dynamics for academic demonstration

### 7.2 Business metrics

- **Research Validation:** Successful validation of paper claims contributing to publication acceptance
- **Community Adoption:** Clear, extensible codebase suitable for follow-up research projects
- **Academic Impact:** Tool adoption by federated learning security research community

### 7.3 Technical metrics

- **Accuracy Targets:** Consistent achievement of 97.7% (±2%) on MNIST and 80.5% (±2%) on CIFAR-10 under 20% Byzantine attacks
- **Optimality Targets:** >93% optimal rule selection frequency on MNIST, >89% on CIFAR-10
- **Stability:** 10/10 successful completion rate for standard 30-round simulations without crashes
- **Performance:** Standard 30-round MNIST simulation completes in under 10 minutes

## 8. Technical considerations

### 8.1 Integration points

- **PyTorch:** All machine learning components and neural network operations
- **GUI Framework:** PySimpleGUI for interactive interface (lightweight and functional)
- **Visualization:** matplotlib embedded within GUI for real-time plotting
- **Data Handling:** torchvision for official MNIST/CIFAR-10 dataset access
- **Configuration:** YAML/JSON support for experiment parameter management

### 8.2 Data storage & privacy

- Public datasets (MNIST, CIFAR-10) downloaded from official torchvision sources via HTTPS
- Local storage in data/ directory with automatic dataset management
- No personally identifiable information used or stored
- Federated learning privacy principles maintained (no raw data sharing)

### 8.3 Scalability & performance

- Strategy Pattern architecture for attacks and defenses enabling easy module addition
- Efficient vectorized operations for trust score calculation across multiple clients
- Memory-optimized Q-table implementation using defaultdict for sparse state spaces
- Performance target: 30-round simulation completion in under 10 minutes on standard hardware

### 8.4 Potential challenges

- GUI responsiveness during intensive ML computations requiring background processing
- Real-time visualization performance optimization for smooth user experience
- Q-learning convergence tuning across different experimental scenarios
- Attack detectability calibration across all four attack patterns

## 9. Milestones & sequencing

### 9.1 Project estimate

- **Size:** Medium
- **Timeline:** 4-6 weeks for complete implementation and validation

### 9.2 Team size & composition

- **Team Size:** 1-2 developers
- **Roles:** Python Developer, ML Engineer (can be combined in single person)

### 9.3 Suggested phases

- **Phase 1: Foundational Setup** (1 week)

  - Core simulation engine with PyTorch integration
  - Basic client/server architecture
  - Dataset loading and partitioning

- **Phase 2: TARS Core Implementation** (1.5 weeks)

  - Trust scoring mechanism with temporal smoothing
  - Q-learning agent with ε-greedy policy
  - All 4 attack pattern implementations
  - Baseline aggregation rule integration

- **Phase 3: GUI and Real-time Visualization** (1.5 weeks)

  - PySimpleGUI interface development
  - Live plotting with matplotlib integration
  - Real-time trust score and performance monitoring
  - Interactive parameter configuration

- **Phase 4: Batch Processing and Validation** (1 week)
  - Command-line interface for headless experiments
  - Result export and statistical analysis
  - Research paper result validation
  - Documentation and testing

## 10. User stories

### 10.1. Interactive experiment configuration and execution

- **ID:** TARS-001
- **Description:** As a researcher, I want to configure and run complete simulation experiments through an intuitive GUI so that I can quickly test different scenarios and demonstrate TARS capabilities.
- **Acceptance criteria:**
  - GUI allows selection of dataset (MNIST/CIFAR-10), client count, Byzantine percentage
  - Attack type selection from all 4 patterns (label flipping, sign flipping, Gaussian, pretense)
  - Data distribution configuration (IID/non-IID)
  - One-click simulation start with immediate visual feedback
  - Real-time progress tracking with estimated completion time

### 10.2. Live performance and trust monitoring

- **ID:** TARS-002
- **Description:** As a researcher, I want to visualize model performance and agent behavior in real-time so that I can observe and demonstrate the adaptive nature of TARS defense mechanisms.
- **Acceptance criteria:**
  - Live-updating "Test Accuracy vs. Round" and "Test Loss vs. Round" graphs
  - Real-time client trust score table with color-coded trust levels
  - Dynamic display of current aggregation rule selected by TARS agent
  - Interactive plots allowing zoom, pan, and data point inspection
  - Trust score evolution visualization with temporal smoothing effects

### 10.3. Headless experimentation for reproducibility

- **ID:** TARS-003
- **Description:** As a researcher, I want to run batch experiments headlessly with configuration files so that I can generate reproducible results for publication.
- **Acceptance criteria:**
  - Command-line execution with YAML/JSON configuration files
  - Automated CSV export containing round-by-round history (accuracy, loss, chosen rule, trust scores)
  - Statistical analysis script generating mean and standard deviation across multiple runs
  - Publication-ready plot generation with confidence intervals
  - Batch execution support for multiple experimental scenarios

### 10.4. Comprehensive baseline comparison

- **ID:** TARS-004
- **Description:** As a researcher, I want to run simulations using baseline defense mechanisms so that I can generate comprehensive comparison data against TARS.
- **Acceptance criteria:**
  - Configuration support for static aggregation rules (always use specific rule)
  - SARA (UCB-based) agent implementation as alternative to TARS
  - Identical experimental conditions across all baseline comparisons
  - Automated benchmarking execution against all baseline methods
  - Comparative result visualization and statistical significance testing

### 10.5. Trust mechanism inspection and analysis

- **ID:** TARS-005
- **Description:** As an academic reviewer, I want to inspect the internal workings of trust calculation and Q-learning decisions so that I can verify and understand the TARS algorithm.
- **Acceptance criteria:**
  - Real-time trust score breakdown showing loss divergence, cosine similarity, and gradient magnitude components
  - Q-table state visualization with action-value evolution over time
  - Trust score history export for detailed offline analysis
  - Aggregation rule selection frequency statistics and timing analysis
  - Interactive trust threshold adjustment with immediate visual feedback

### 10.6. Educational demonstration capabilities

- **ID:** TARS-006
- **Description:** As an academic instructor, I want to use pre-configured scenarios to demonstrate federated learning security concepts so that I can effectively teach students about attack/defense dynamics.
- **Acceptance criteria:**
  - Pre-configured demonstration scenarios showcasing each attack type
  - Step-by-step simulation control (pause, resume, step-through)
  - Clear visual indicators of attack onset and defense activation
  - Explanatory tooltips and help text for all GUI components
  - Simplified interface mode hiding advanced parameters for educational use

### 10.7. Result export and publication support

- **ID:** TARS-007
- **Description:** As a researcher, I want to export simulation results in multiple formats so that I can conduct statistical analysis and create publication-quality figures.
- **Acceptance criteria:**
  - Export capabilities in CSV, JSON, and plot-ready formats
  - Statistical metrics including mean, standard deviation, confidence intervals
  - Publication-ready figure generation (PNG, PDF, SVG formats)
  - Trust score evolution data with per-client and aggregate statistics
  - Automated report generation summarizing experimental outcomes

### 10.8. Extensibility and customization

- **ID:** TARS-008
- **Description:** As a developer, I want to extend the platform with new attacks or aggregation rules so that I can customize experiments for specific research needs.
- **Acceptance criteria:**
  - Modular attack interface enabling custom attack pattern implementation
  - Extensible aggregation rule interface for new Byzantine-robust methods
  - Plugin architecture supporting easy integration of new components
  - Clear documentation for adding new attacks, defenses, and visualization components
  - Configuration schema validation ensuring parameter compatibility

### 10.9. Performance optimization and scalability

- **ID:** TARS-009
- **Description:** As a system administrator, I want the platform to handle research-scale experiments efficiently so that I can evaluate various federated learning scenarios with optimal performance.
- **Acceptance criteria:**
  - Support for 5-50 clients with linear performance scaling
  - Trust score calculation latency under 100ms per client update
  - Smooth real-time visualization during extended experiments
  - Memory usage optimization for prolonged simulation runs
  - Background processing ensuring GUI responsiveness during computation

### 10.10. Security and privacy compliance

- **ID:** TARS-010
- **Description:** As a privacy-conscious researcher, I want to ensure the platform maintains federated learning privacy principles so that the simulation accurately reflects real-world privacy constraints.
- **Acceptance criteria:**
  - Client raw data never transmitted to server or other clients
  - Only model parameters (state_dict) shared between participants
  - Trust scores calculated using only received model updates
  - No data leakage through trust scoring or aggregation mechanisms
  - Privacy-preserving experiment logging and result export

## 11. Security & Privacy by Design

### 11.1 Threat model

- **Adversarial Goal:** Byzantine attackers aim to degrade global model accuracy through coordinated manipulation
- **Attack Vectors:** Malicious clients control model update transmission and can implement sophisticated poisoning strategies
- **Security Assumptions:** Central server and TARS agent remain secure and uncompromised throughout experiments

### 11.2 Security requirements

- Trust scoring mechanism designed to be resistant to gaming and manipulation attempts
- All external data downloaded from official, trusted sources (torchvision) over HTTPS connections
- Experimental isolation ensuring no cross-contamination between simulation runs

## 12. Error Handling & Resilience

### 12.1 Graceful failure handling

- Clear error messages for dataset download failures with suggested remediation steps
- Graceful simulation termination for mathematical errors (NaN values) with detailed error state logging
- Automatic recovery mechanisms for temporary computation failures

### 12.2 Input validation

- Comprehensive GUI input validation with real-time feedback and constraint enforcement
- "Start Simulation" button disabled until all required inputs pass validation
- Configuration file schema validation for batch experiment execution

## 13. Release & Dissemination Plan

### 13.1 Open source licensing

- MIT License for maximum research community accessibility and adoption
- Clear attribution requirements for academic use and derivative works

### 13.2 Documentation standards

- Comprehensive README.md with installation, usage, and extension instructions
- Well-commented codebase with detailed docstrings for all major functions and classes
- Academic usage examples and tutorial notebooks for educational adoption

### 13.3 Repository management

- Public GitHub repository with organized issue tracking and contribution guidelines
- Continuous integration for automated testing and code quality assurance
- Release management with semantic versioning for research reproducibility

---

**This hybrid PRD combines the practical focus and realistic timeline of the new version with the comprehensive structure and technical depth of the original, creating an optimal roadmap for TARS platform development that serves both research validation and educational demonstration needs.**
