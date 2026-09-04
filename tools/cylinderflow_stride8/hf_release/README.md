---
license: apache-2.0
pretty_name: MeshGraphNets CylinderFlow phase-0 stride-8 (75 frames)
task_categories:
- time-series-forecasting
tags:
- fluid-dynamics
- computational-fluid-dynamics
- graph-neural-networks
- hdf5
- meshgraphnets
- text2pde
---

# MeshGraphNets CylinderFlow: phase-0 stride-8, 75 frames

This is a temporal derivative of the CylinderFlow data released with [MeshGraphNets](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets). It is not an independently simulated flow family.

Each of the 1,100 original 600-frame trajectories is sampled once with the phase-zero rule `raw_uvp[0:600:8]`. The HDF5 retains all 75 selected frames with raw indices `0, 8, ..., 592` and physical spacing `0.08` when the original solver output spacing is `0.01`. Trajectory-level membership preserves the upstream split: 1,000 Train trajectories and 100 Validation trajectories. Test fields were not accessed and no Test data is included.

The `feature/cylinderflow-stride8-1plus64` Text2PDE workflow reads exactly one sample per trajectory: the first 65 stored frames, corresponding to raw indices `0, 8, ..., 512`. Frame 0 is the clean condition and frames 1 through 64 are the prediction targets. It does not create sliding windows or alternate temporal phases. The remaining 10 stored frames stay in the release for provenance and reuse.

## Files

- `data/cylinderflow_stride8_75frames.h5`: complete 75-frame HDF5, 1,772,387,753 bytes, SHA-256 `d416be274e03a5d77f1cf2dffc4be8abcfc63ff9d6bf5a9cca19a44b43b36533`.
- `metadata/cylinderflow_stride8_75frames_manifest.json`: trajectory identities, split membership, raw-frame mapping, and Train-only statistics.
- `metadata/dataset_audit.json`: exact source-to-derived checks and data identity.
- `metadata/train_normal_stats.pkl`: Text2PDE-compatible `[u_mean,u_std,v_mean,v_std,p_mean,p_std]` pickle.
- `LICENSE`, `NOTICE`, and `TEST_NOT_ACCESSED.txt`: license, attribution, transformation, and access-boundary records.

The HDF5 root format is `dgn4cfd.mgn_cylinderflow_temporal_stride.v1`. Each `trajectory_XXXX` group contains:

- `uvp`: `[75, N, 3]` float32 with channels `(u, v, p)`;
- `mesh_pos`: `[N, 2]` float32;
- `cells`: `[C, 3]` int32 triangle connectivity;
- `node_type`: `[N]` int32 MeshGraphNets boundary labels.

Node and cell counts vary by trajectory. Load groups lazily and use a batch size of one unless a mesh-aware collator is supplied.
The file uses HDF5 2.0 object layouts and requires an h5py build linked against
HDF5 2.0 or newer (the accompanying Text2PDE environment pins `h5py==3.16.0`).

## Normalization

The released normalizer uses every one of the 75 stored frames from the 1,000 Train trajectories only:

| field | mean | standard deviation |
|---|---:|---:|
| `u` | 0.5661443793145688 | 0.57014025360096 |
| `v` | 0.0006225839965382098 | 0.1436773068500826 |
| `p` | 0.04184413450776907 | 0.34627331301859265 |

Validation fields do not contribute to these statistics.

## Provenance and references

MeshGraphNets' upstream README states that its release contains the full datasets. This derivative follows the upstream [Apache License 2.0](https://github.com/google-deepmind/deepmind-research/blob/master/LICENSE) and preserves attribution in `NOTICE`.

- Tobias Pfaff et al., [Learning Mesh-Based Simulation with Graph Networks](https://arxiv.org/abs/2010.03409), ICLR 2021.
- [MeshGraphNets dataset and code release](https://github.com/google-deepmind/deepmind-research/blob/master/meshgraphnets/README.md).
- [Text2PDE: Latent Diffusion Models for Accessible Physics Simulation](https://arxiv.org/abs/2410.01153).

Please cite the upstream MeshGraphNets work when using this derivative and cite Text2PDE when using the accompanying Text2PDE training/evaluation workflow.
