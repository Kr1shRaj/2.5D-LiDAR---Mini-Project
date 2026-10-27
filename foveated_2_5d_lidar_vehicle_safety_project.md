# Foveated 2.5D LiDAR Elevation Mapping for Vehicle Safety

## 1. Project Overview

We are building a software-based **foveated 2.5D LiDAR elevation-mapping system** intended to support safer vehicle perception. The main idea is to turn LiDAR point-cloud data into a compact elevation map that preserves important road and obstacle information while avoiding the cost of representing every part of the scene at the same spatial resolution.

Our initial application is vehicle safety: identifying potentially hazardous road features and obstacles early enough to support time-critical driving decisions. The project is a research prototype, not a production-ready collision-avoidance system.

## 2. Why This Problem Matters

Vehicles need a reliable representation of the environment to detect road geometry and potential hazards.

- **Dense 3D LiDAR processing** captures detailed spatial information, but processing and storing large point clouds can require significant computation and memory.
- **Conventional 2D maps** are efficient but can lose important height information, such as road elevation changes, curbs, steep steps and some obstacles.
- **Uniform-resolution 2.5D maps** retain elevation information, but use the same cell size throughout the map even when some regions need more detail than others.
- **Dynamic road scenes** contain moving vehicles, cyclists and pedestrians. Combining observations over time without appropriate updates can leave stale information or ghost trails.

These challenges motivate a mapping approach that balances spatial detail, computational cost and the preservation of safety-relevant features.

## 3. What Is 2.5D Elevation Mapping?

A 2D map represents horizontal position, typically using X and Y coordinates. A 3D point cloud represents X, Y and Z for each measured point. A 2.5D elevation map uses a 2D grid but stores height-related information in each cell.

A cell may store:
- Estimated surface or ground elevation
- Elevation variation or roughness
- Obstacle or occupancy information
- Observation confidence and update time
- Optional semantic labels, when available

This representation can be more compact than a dense 3D point cloud while retaining useful terrain information. However, a single elevation value per cell cannot represent every vertically overlapping structure. Bridges, overhangs and multiple surfaces may require additional layers or explicit limitations.

## 4. What Does “Foveated” Mean?

Foveated mapping allocates spatial detail according to where it is most useful. It is inspired by human vision, which sees the region of focus in greater detail than the periphery.

For our vehicle-safety use case:
- **Near the vehicle:** Use fine spatial resolution to represent road edges, curbs, potholes and nearby obstacles.
- **Farther from the vehicle:** Use coarser resolution where fine detail may not be necessary.
- **Important regions at any distance:** Allow local refinement when a safety-relevant feature is detected or suspected.

Distance alone should not determine importance. A distant pedestrian, vehicle or road hazard may still need to be preserved. The system must therefore balance range, geometric risk indicators and semantic importance where those signals are available.

## 5. Proposed System

The planned pipeline is:

1. **LiDAR input:** Read real-world point-cloud data from a recorded dataset or, if available, a live sensor.
2. **Preprocessing:** Filter invalid/noisy points and transform points into a consistent coordinate frame.
3. **3D-to-2.5D conversion:** Project point-cloud measurements into a spatial grid and estimate elevation and other cell properties.
4. **Foveated resolution allocation:** Use finer cells in nearby and important regions and coarser cells in less critical regions.
5. **Temporal map maintenance:** Update cell estimates over successive scans and reduce the influence of stale observations. Dynamic-object filtering may be added if reliable labels or detection methods are available.
6. **Hazard-related prediction:** Explore a model that uses map features to classify regions as traversable, potentially hazardous, obstacle-containing or unknown.
7. **Visualization and evaluation:** Display the input point cloud, generated map, resolution regions, predictions and performance measurements.

The map is the main output of the project. The prediction model is a downstream component that can help demonstrate whether the map preserves information useful for a vehicle-safety task.

## 6. How We Plan to Build It

### Phase 1 — Establish a baseline

- Select a suitable real-world LiDAR dataset with sequential scans and, preferably, semantic annotations.
- Start with a small sequence or spatial region rather than processing the entire dataset.
- Implement a conventional uniform-resolution 2.5D elevation grid.
- Visualize the map and verify that coordinate conversion and elevation estimation are correct.

### Phase 2 — Implement foveated mapping

- Divide the surrounding area into distance bands or use a suitable variable-resolution structure.
- Assign smaller cells to the near field and larger cells to the far field.
- Add local refinement for significant elevation discontinuities or important obstacles where feasible.
- Ensure coordinate conversion and aggregation work correctly across cell sizes.

Illustrative starting parameters—not final requirements—could be 5 cm cells in the near field, 10–20 cm in the middle field and 25–50 cm farther away. The final values must be chosen based on sensor resolution, dataset quality, computing resources and evaluation results.

### Phase 3 — Maintain the map over time

- Process scans in sequence.
- Store observation timestamps and confidence for map cells.
- Update elevation and occupancy as new scans arrive.
- Investigate stale observations and ghost trails.
- Add dynamic-object filtering only after establishing a reliable baseline.

### Phase 4 — Add a prediction model

The faculty has suggested using a model with real-world data. A feasible first target is **terrain/obstacle-related classification**, rather than trying to predict accidents directly.

Potential model inputs include:
- Mean and maximum cell elevation
- Elevation variance, slope and roughness
- Height discontinuities
- Point density
- Distance from the vehicle
- Semantic class, if reliable labels are available

A simple baseline such as a decision tree, Random Forest or gradient-boosted model may be sufficient initially. If semantic labels are available, they can help establish reference classes. The model should be trained and evaluated on separate sequences or areas to reduce data leakage.

The system should report predictions as candidate hazards or traversability estimates—not as guaranteed accident predictions.

### Phase 5 — Evaluate against baselines

Compare the following configurations on the same data:

1. Uniform-resolution 2.5D mapping.
2. Foveated 2.5D mapping.
3. Foveated 2.5D mapping with the prediction model.

Measure:
- **Memory consumption:** Does foveation reduce map storage?
- **Processing latency:** How long does each scan take to process?
- **Map quality:** How accurately are elevations and important structures preserved?
- **Hazard/obstacle preservation:** Are important obstacles retained after resolution reduction?
- **Prediction quality:** If a model is used, report appropriate metrics such as precision, recall, F1-score and confusion matrix.
- **Temporal consistency:** Are stale observations or ghost trails reduced?

Use the same input sequences and comparable settings for fair comparisons. Report measured results rather than assuming that the foveated approach will always be faster or more accurate.

## 7. Real-World Data and Testing

A practical starting point is a public, recorded automotive LiDAR dataset such as **SemanticKITTI** or the **KITTI** dataset family. These provide real sensor recordings that can be replayed frame by frame.

It is important to distinguish:
- **Recorded real-world data replay:** Real sensor observations processed offline or frame by frame.
- **Live sensor input:** Point clouds arriving directly from a physical LiDAR device.

Recorded data can validate the algorithm against real-world observations, but it is not the same as a live vehicle deployment. If the faculty requires live sensor input, the team will need access to a suitable LiDAR sensor or a device that can provide live point clouds.

A small, well-chosen sequence is enough to begin development. Check dataset terms, file formats, calibration, annotations and available ground truth before choosing the final subset.

## 8. Expected Deliverables

- A working 3D point-cloud to 2.5D elevation-map pipeline.
- A foveated resolution-allocation module.
- A visualization showing the input cloud and resulting map.
- A baseline comparison of uniform and foveated mapping.
- A small prediction model, if feasible, with documented evaluation.
- Performance results covering computational cost and preservation of important spatial features.
- A report explaining assumptions, limitations and possible future deployment.

## 9. Scope and Safety Limitations

The project aims to demonstrate a mapping and perception concept, not to certify a collision-avoidance system.

- A 2.5D map can lose information about vertically separated or overlapping objects unless additional layers are used.
- LiDAR performance can be affected by sensor range, weather, surface reflectivity, occlusion and point density.
- Terrain or obstacle classification does not automatically predict collisions. Accident prevention requires additional components such as object tracking, motion prediction, vehicle dynamics and a validated decision-making system.
- The prototype must not be used to control a real vehicle or make safety-critical decisions without extensive validation.

## 10. One-Sentence Summary

**We are developing a foveated 2.5D LiDAR elevation-mapping system that uses real-world point-cloud data to preserve safety-relevant road and obstacle information while reducing unnecessary mapping computation, with an optional prediction model to assess potential hazards.**
